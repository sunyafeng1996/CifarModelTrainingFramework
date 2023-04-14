from torch.utils.data import DataLoader
from torchvision import datasets, transforms

'''此函数返回一个指定数据集的DataLoaer'''
def GetDataLoader(dataset,train_bs,test_bs,dataset_path):
    if dataset == 'CIFAR10':
        train_dataset=datasets.CIFAR10(dataset_path,train=True, download=True,\
            transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ]))
        test_dataset=datasets.CIFAR10(dataset_path,train=False, download=True,\
            transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ]))
    elif dataset == 'CIFAR100':
        train_dataset=datasets.CIFAR100('./real_datasets/cifar100',train=True, download=True,\
            transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ]))
        test_dataset=datasets.CIFAR100('./real_datasets/cifar100',train=False, download=True,\
            transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ]))
    train_loader=DataLoader(train_dataset,train_bs,shuffle=True)
    test_loader=DataLoader(test_dataset,test_bs,shuffle=True)
    
    return train_loader,test_loader