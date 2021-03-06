import logging

import paddle

import paddle.vision.transforms as transforms
import paddle.vision.datasets as datasets

from paddle.io import DataLoader,RandomSampler, SequenceSampler,DistributedBatchSampler,Dataset

from paddle.distributed import fleet

logger = logging.getLogger(__name__)

class MyDataset(Dataset):
    def __init__(self, trainset):
        self.num_samples = len(trainset)
        self.trainset=trainset
    def __getitem__(self,idx):

        if(isinstance(idx,int)):
            print(1)
            image,label=self.trainset[idx]
        else:
            print(2)
            image,label=self.trainset[idx[0]]

        return image,label
    def __len__(self):
        return self.num_samples


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
        train_dataset = MyDataset(trainset)

        testset = datasets.Cifar100(mode='test',
                                    download=True,
                                    transform=transform_test) if args.local_rank in [-1, 0] else None
        test_dataset = MyDataset(testset)

    if args.local_rank == 0:
        fleet.barrier_worker()

    
    tra_sampler = RandomSampler(train_dataset)if args.local_rank == -1 else DistributedBatchSampler(train_dataset,
                                                                                                    batch_size=1,
                                                                                                    shuffle=True)

    train_sampler=paddle.io.BatchSampler(sampler=tra_sampler,
                                        batch_size=args.train_batch_size,
                                        drop_last=True)
    tes_sampler =  SequenceSampler(test_dataset)
    test_sampler=paddle.io.BatchSampler(sampler=tes_sampler,
                                        batch_size=args.eval_batch_size,
                                        drop_last=True)

    train_loader = DataLoader(train_dataset,
                              batch_sampler=train_sampler,
                              num_workers=0)
    test_loader = DataLoader(test_dataset,
                             batch_sampler=test_sampler,
                             num_workers=0) if testset is not None else None

    return train_loader, test_loader