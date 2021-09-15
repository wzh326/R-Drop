from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np

import paddle
import paddle.distributed as dist

from tqdm import tqdm

import paddle.amp as amp
#from apex import amp
#from apex.parallel import DistributedDataParallel as DDP

from models.modeling import VisionTransformer, CONFIGS

from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader


import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """估计和储存平均预测与当前预测结果"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#输出预测结果与标签的准确率
def simple_accuracy(preds, labels):
    return (preds == labels).mean()

#保存模型
def save_model(args, model,optimizer):
    model_to_save = model.module if hasattr(model, 'module') else model

    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.pdparams" % args.name)
    optimizer_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.pdopt" % args.name)

    paddle.save(model_to_save.state_dict(), model_checkpoint)
    paddle.save(optimizer.state_dict(), optimizer_checkpoint)

    logger.info("Saved model  checkpoint to [DIR: %s]", args.output_dir)

def init_optimizer(args,model):
    optimizer = paddle.optimizer.Momentum(
                            learning_rate=args.learning_rate,
                            parameters=model.parameters(),
                            momentum=0.9,
                            weight_decay=args.weight_decay,
                            grad_clip=paddle.nn.ClipGradByGlobalNorm(args.max_grad_norm))
    return optimizer

def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]

    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.pdparams" % args.name)
    optimizer_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.pdopt" % args.name)

    num_classes = 10 if args.dataset == "cifar10" else 100
    if args.dataset == "imagenet":
        num_classes=1000

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes,alpha=args.alpha)

    optimizer=init_optimizer(args,model)

    if args.model_load:
        try:
            model.set_state_dict(paddle.load(model_checkpoint))
            print("model already load")
            optimizer.set_state_dict(paddle.load(optimizer_checkpoint))
            print("optimizer already load")
        except:
            pass
    else:
        model.load_from(np.load(args.pretrained_dir))

    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)

    return args, model,optimizer


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.backward())
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)
    if args.n_gpu > 0:
        paddle.seed(args.seed)


def valid(args, model, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = paddle.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        x, y = batch
        with paddle.no_grad():
            logits = model(x)[0]

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = paddle.argmax(logits, axis=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    return accuracy


def train(args, model,optimizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader, test_loader = get_loader(args)

    t_total = args.num_steps

    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(args.learning_rate, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(args.learning_rate, warmup_steps=args.warmup_steps, t_total=t_total)

    #scale loss
    if args.fp16:
        scaler=amp.GradScaler(init_loss_scaling=2**15)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    paddle.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    optimizer.clear_gradients()

    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    while True:
        model.train()
        
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        
        for step, batch in enumerate(epoch_iterator):
            x, y = batch       
            if args.fp16:
                with amp.auto_cast():                   
                    loss = model(x, y)
            else:
                loss=model(x,y)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                scaled = scaler.scale(loss)
                scaled.backward()
            else:
                loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item()*args.gradient_accumulation_steps)
                
                if args.fp16:
                    scaler.minimize(optimizer, scaled)
                else:
                    optimizer.step()
                    
                scheduler.step()
                optimizer.clear_gradients()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
                if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                    accuracy = valid(args, model, test_loader, global_step)
                    print("Accuracy:",accuracy)
                    if best_acc < accuracy:
                        save_model(args, model,optimizer)
                        best_acc = accuracy
                    model.train()

                if global_step % t_total == 0:
                    break
        losses.reset()
        if global_step % t_total == 0:
            break

    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")
    print("Best Accuracy:",best_acc)


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100","imagenet"], default="cifar10",
                        help="Which downstream task.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=384, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=12, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=50, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=1e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=1000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--alpha", default=0.3, type=float,
                        help="alpha for kl loss")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', type=bool,default=False,
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--model_load', type=bool, default=False,
                        help="model_load")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:     #不进行并行式训练
        #运行的设备
        device=paddle.device.set_device("gpu" if paddle.device.is_compiled_with_cuda() else "cpu")       
        #环境中GPU的数量
        env = dist.ParallelEnv()
        args.n_gpu = env.world_size
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        paddle.device.set_device(args.local_rank)
        device = paddle.device.set_device("gpu", args.local_rank)
        dist.init_parallel_env()
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))


    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model,optimizer = setup(args)



    # Training
    train(args, model,optimizer)


if __name__ == "__main__":
    main()