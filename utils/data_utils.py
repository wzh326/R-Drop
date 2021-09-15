import logging

import paddle

import paddle.vision.transforms as transforms
import paddle.vision.datasets as datasets

from paddle.io import DataLoader,RandomSampler, SequenceSampler

logger = logging.getLogger(__name__)

#实现torch.utils.data.DistributedSampler
class DistributedSampler(paddle.io.DistributedBatchSampler):
    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 seed=0,
                 drop_last=False):
        super().__init__(
            dataset=dataset,
            batch_size=1,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last)

def get_loader(args):
    if args.local_rank not in [-1, 0]:
        paddle.distributed.barrier()

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if args.dataset == "cifar10":
        trainset = datasets.Cifar10(mode='train',
                                    download=True,
                                    transform=transform_train)
        testset = datasets.Cifar10(mode='test',
                                   download=True,
                                   transform=transform_test) if args.local_rank in [-1, 0] else None
        
    elif args.dataset == "cifar100":
        trainset = datasets.Cifar100(mode='train',
                                     download=True,
                                     transform=transform_train)
        testset = datasets.Cifar100(mode='test',
                                    download=True,
                                    transform=transform_test) if args.local_rank in [-1, 0] else None
    if args.local_rank == 0:
        paddle.distributed.barrier()

    
    tra_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    train_sampler=paddle.io.BatchSampler(sampler=tra_sampler,
                                        batch_size=args.train_batch_size,
                                        drop_last=True)
    tes_sampler =  SequenceSampler(testset)
    test_sampler=paddle.io.BatchSampler(sampler=tes_sampler,
                                        batch_size=args.eval_batch_size,
                                        drop_last=True)

    train_loader = DataLoader(trainset,
                              batch_sampler=train_sampler,
                              num_workers=0)
    test_loader = DataLoader(testset,
                             batch_sampler=test_sampler,
                             num_workers=0) if testset is not None else None

    return train_loader, test_loader