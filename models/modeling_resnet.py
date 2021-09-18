

from os.path import join as pjoin

from collections import OrderedDict  # pylint: disable=g-importing-member

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import copy

def ttp(tensor):
    param=paddle.create_parameter(shape=tensor.shape,
                                  dtype=str(tensor.numpy().dtype),
                                  default_initializer=paddle.nn.initializer.Assign(tensor))
    return param


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return paddle.to_tensor(weights)

def var_mean(input, dim=None, unbiased=True, keepdim=False):
    var = paddle.var(input, axis=dim, 
                     unbiased=unbiased, keepdim=keepdim)
    mean = paddle.mean(input, 
                       axis=dim, 
                       keepdim=keepdim)
    return var, mean

class StdConv2d(nn.Conv2D):

    def forward(self, x):
        w = self.weight
        v, m = var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / paddle.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Layer):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU()

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

    def load_from(self, weights, n_block, n_unit):
        conv1_weight = ttp(np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True))
        conv2_weight = ttp(np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True))
        conv3_weight = ttp(np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True))

        gn1_weight = ttp(np2th(weights[pjoin(n_block, n_unit, "gn1/scale")]))
        gn1_bias = ttp(np2th(weights[pjoin(n_block, n_unit, "gn1/bias")]))

        gn2_weight = ttp(np2th(weights[pjoin(n_block, n_unit, "gn2/scale")]))
        gn2_bias = ttp(np2th(weights[pjoin(n_block, n_unit, "gn2/bias")]))

        gn3_weight = ttp(np2th(weights[pjoin(n_block, n_unit, "gn3/scale")]))
        gn3_bias = ttp(np2th(weights[pjoin(n_block, n_unit, "gn3/bias")]))

        self.conv1.weight=ttp(copy.deepcopy(conv1_weight))
        self.conv2.weight=ttp(copy.deepcopy(conv2_weight))
        self.conv3.weight=ttp(copy.deepcopy(conv3_weight))

        self.gn1.weight=ttp(copy.deepcopy(paddle.reshape(gn1_weight,shape=[-1])))
        self.gn1.bias=ttp(copy.deepcopy(paddle.reshape(gn1_bias,shape=[-1])))

        self.gn2.weight=ttp(copy.deepcopy(paddle.reshape(gn2_weight,shape=[-1])))
        self.gn2.bias=ttp(copy.deepcopy(paddle.reshape(gn2_bias,shape=[-1])))

        self.gn3.weight=ttp(copy.deepcopy(paddle.reshape(gn3_weight,shape=[-1])))
        self.gn3.bias=ttp(copy.deepcopy(paddle.reshape(gn3_bias,shape=[-1])))

        if hasattr(self, 'downsample'):
            proj_conv_weight = ttp(np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True))
            proj_gn_weight = ttp(np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")]))
            proj_gn_bias = ttp(np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")]))

            self.downsample.weight=ttp(copy.deepcopy(proj_conv_weight))
            self.gn_proj.weight=ttp(copy.deepcopy(paddle.reshape(proj_gn_weight,shape=[-1])))
            self.gn_proj.bias=ttp(copy.deepcopy(paddle.reshape(proj_gn_bias,shape=[-1])))

class ResNetV2(nn.Layer):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        # The following will be unreadable if we split lines.
        # pylint: disable=line-too-long
        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU()),
            ('pool', nn.MaxPool2D(kernel_size=3, stride=2, padding=0))
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),    
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))

    def forward(self, x):
        x = self.root(x)
        x = self.body(x)
        return x
