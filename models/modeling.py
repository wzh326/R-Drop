from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import paddle
import paddle.nn as nn
import numpy as np

import paddle.nn.functional as F

from paddle.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2D, LayerNorm

from scipy import ndimage

import models.configs as configs

from .modeling_resnet import ResNetV2

import sys

logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return paddle.to_tensor(weights)


def swish(x):
    m=paddle.nn.Sigmoid()
    return x * m(x)


def ttp(tensor):
    param=paddle.create_parameter(shape=tensor.shape,
                                  dtype=str(tensor.numpy().dtype),
                                  default_initializer=paddle.nn.initializer.Assign(tensor))
    return param

ACT2FN = {"gelu": paddle.nn.functional.gelu, "relu": paddle.nn.functional.relu, "swish": swish}


class Attention(nn.Layer):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(axis=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + [self.num_attention_heads, self.attention_head_size]
        x = paddle.reshape(x,shape=new_x_shape)
        return paddle.transpose(x,perm=[0, 2, 1, 3])

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = paddle.matmul(query_layer, key_layer,transpose_y=True)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = paddle.matmul(attention_probs, value_layer)
        context_layer = paddle.transpose(context_layer,perm=[0, 2, 1, 3])
        new_context_layer_shape = context_layer.shape[:-2] + [self.all_head_size]
        context_layer = paddle.reshape(context_layer,shape=new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Layer):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self._init_weights() 
        self.fc1 = paddle.nn.Linear( config.hidden_size,config.transformer['mlp_dim'],weight_attr=self.weight_attr,bias_attr=self.bias_attr)
        self.fc2 = paddle.nn.Linear(config.transformer['mlp_dim'],config.hidden_size,weight_attr=self.weight_attr,bias_attr=self.bias_attr)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])   

    def _init_weights(self):
        self.bias_attr = paddle.ParamAttr(
                                            name=None,
                                            initializer=paddle.nn.initializer.Normal(std=1e-6))     
        self.weight_attr = paddle.ParamAttr(
                                                name=None,
                                                initializer=paddle.nn.initializer.XavierUniform())
    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Layer):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = (img_size,img_size)

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = (config.patches["size"])

            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
                                         width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2D(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        pe=paddle.zeros((1, n_patches+1, config.hidden_size))
        self.position_embeddings = paddle.create_parameter(shape=pe.shape,
                                                            dtype=str(pe.numpy().dtype),
                                                            default_initializer=paddle.nn.initializer.Assign(pe))
        ct=paddle.zeros((1, 1, config.hidden_size))
        self.cls_token =paddle.create_parameter(shape=ct.shape,
                                                dtype=str(ct.numpy().dtype),
                                                default_initializer=paddle.nn.initializer.Assign(ct))
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = paddle.expand(self.cls_token,shape=[B, -1, -1])

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = paddle.transpose(x,perm=[0,2,1])
        x = paddle.concat((cls_tokens, x), axis=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Layer):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, epsilon=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, epsilon=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)

        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with paddle.no_grad():
            query_weight = paddle.reshape(np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]),shape=[self.hidden_size, self.hidden_size])
            key_weight = paddle.reshape(np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]),shape=[self.hidden_size, self.hidden_size])
            value_weight = paddle.reshape(np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]),shape=[self.hidden_size, self.hidden_size])
            out_weight = paddle.reshape(np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]),shape=[self.hidden_size, self.hidden_size])

            query_bias = paddle.reshape(np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]),shape=[-1])
            key_bias = paddle.reshape(np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]),shape=[-1])
            value_bias = paddle.reshape(np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]),shape=[-1])
            out_bias = paddle.reshape(np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]),shape=[-1])

            self.attn.query.weight=ttp(copy.deepcopy(query_weight))
            self.attn.key.weight=ttp(copy.deepcopy(key_weight))
            self.attn.value.weight=ttp(copy.deepcopy(value_weight))
            self.attn.out.weight=ttp(copy.deepcopy(out_weight))
            self.attn.query.bias=ttp(copy.deepcopy(query_bias))
            self.attn.key.bias=ttp(copy.deepcopy(key_bias))
            self.attn.value.bias=ttp(copy.deepcopy(value_bias))
            self.attn.out.bias=ttp(copy.deepcopy(out_bias))

            mlp_weight_0 = ttp(np2th(weights[pjoin(ROOT, FC_0, "kernel")]))
            mlp_weight_1 = ttp(np2th(weights[pjoin(ROOT, FC_1, "kernel")]))
            mlp_bias_0 = ttp(np2th(weights[pjoin(ROOT, FC_0, "bias")]))
            mlp_bias_1 = ttp(np2th(weights[pjoin(ROOT, FC_1, "bias")]))

            self.ffn.fc1.weight=ttp(copy.deepcopy(mlp_weight_0))
            self.ffn.fc2.weight=ttp(copy.deepcopy(mlp_weight_1))
            self.ffn.fc1.bias=ttp(copy.deepcopy(mlp_bias_0))
            self.ffn.fc2.bias=ttp(copy.deepcopy(mlp_bias_1))

            self.attention_norm.weight=ttp(copy.deepcopy(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")])))
            self.attention_norm.bias=ttp(copy.deepcopy(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")])))
            self.ffn_norm.weight=ttp(copy.deepcopy(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")])))
            self.ffn_norm.bias=ttp(copy.deepcopy(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")])))


class Encoder(nn.Layer):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.LayerList()
        self.encoder_norm = LayerNorm(config.hidden_size, epsilon=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        hidden_state=[]
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            hidden_state.append(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        hidden_state.pop(-1)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights,hidden_state


class Transformer(nn.Layer):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)

        encoded, attn_weights,hidden_state = self.encoder(embedding_output)
        return encoded, attn_weights,hidden_state


class VisionTransformer(nn.Layer):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False,alpha=0.3):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.head = Linear(config.hidden_size, num_classes)
        self.alpha=alpha
    def forward(self, x, labels=None):
        x1, attn_weights1,hidden_state1 = self.transformer(x)
        logits = self.head(x1[:, 0])

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(paddle.reshape(logits,shape=[-1, self.num_classes]), paddle.reshape(labels,shape=[-1]))
            x2, attn_weights2,hidden_state2 = self.transformer(x)
            newlogits = self.head(x2[:, 0])
            loss2 = loss_fct(paddle.reshape(newlogits,shape=[-1, self.num_classes]),paddle.reshape(labels,shape=[-1]))
            loss+=loss2
            
            p = F.log_softmax(paddle.reshape(logits,shape=[-1, self.num_classes]), axis=-1)
            p_tec = F.softmax(paddle.reshape(logits,shape=[-1, self.num_classes]), axis=-1)
            q = F.log_softmax(paddle.reshape(newlogits,shape=[-1, self.num_classes]), axis=-1)
            q_tec = F.softmax(paddle.reshape(newlogits,shape=[-1, self.num_classes]), axis=-1)
            kl_loss = F.kl_div(p, q_tec, reduction='none').sum()
            reverse_kl_loss = F.kl_div(q, p_tec, reduction='none').sum()


            loss +=self.alpha * (kl_loss + reverse_kl_loss)
            return loss
        else:
            return logits, attn_weights1
    def load_from(self, weights):
        with paddle.no_grad():
            if self.zero_head:
                hw=paddle.zeros_like(self.head.weight)
                self.head.weight=ttp(hw)
                hb=paddle.zeros_like(self.head.bias)                                       
                self.head.bias=ttp(hb)
            else:
                self.head.weight=ttp(copy.deepcopy(np2th(weights["head/kernel"]).t()))

                self.head.bias=ttp(copy.deepcopy(np2th(weights["head/bias"]).t()))

            self.transformer.embeddings.patch_embeddings.weight=ttp(copy.deepcopy(np2th(weights["embedding/kernel"], conv=True)))
            self.transformer.embeddings.patch_embeddings.bias=ttp(copy.deepcopy(np2th(weights["embedding/bias"])))
            self.transformer.embeddings.cls_token=ttp(copy.deepcopy(np2th(weights["cls"])))
            self.transformer.encoder.encoder_norm.weight=ttp(copy.deepcopy(np2th(weights["Transformer/encoder_norm/scale"])))
            self.transformer.encoder.encoder_norm.bias=ttp(copy.deepcopy(np2th(weights["Transformer/encoder_norm/bias"])))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.shape == posemb_new.shape:
                self.transformer.embeddings.position_embeddings=copy.deepcopy(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.shape, posemb_new.shape))
                ntok_new = posemb_new.shape[1]

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape([gs_old, gs_old, -1])

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings=ttp(copy.deepcopy(np2th(posemb)))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight=ttp(copy.deepcopy(np2th(weights["conv_root/kernel"], conv=True)))
                gn_weight = paddle.reshape(np2th(weights["gn_root/scale"]),shape=[-1])
                gn_bias = paddle.reshape(np2th(weights["gn_root/bias"]),shape=[-1])
                self.transformer.embeddings.hybrid_model.root.gn.weight=ttp(copy.deepcopy(gn_weight))
                self.transformer.embeddings.hybrid_model.root.gn.bias=ttp(copy.deepcopy(gn_bias))

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'testing': configs.get_testing(),
}