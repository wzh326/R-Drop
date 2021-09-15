import logging
import argparse
import numpy as np
import os
import paddle
import paddle.distributed as dist
from utils.data_utils import get_loader
from models.modeling import VisionTransformer
from tqdm import tqdm



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



def valid(args):

    train_loader,test_loader = get_loader(args)
    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes,alpha=args.alpha)    
    if args.model_load:
        model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.pdparams" % args.name)
        model.set_state_dict(paddle.load(model_checkpoint))
        print("model already load")
    else:
        model_checkpoint = os.path.join(args.model)
        model.set_state_dict(paddle.load(model_checkpoint))
        print("model already load")

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
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    return accuracy


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--img_size", default=384, type=int,
                        help="Resolution size")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--model_load', type=bool, default=False,
                        help="pre train model")
    parser.add_argument('--model', type=str, default=False,
                        help="self training model")
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

    valid(args)

if __name__ == "__main__":
    main()