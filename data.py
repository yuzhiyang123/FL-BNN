import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
import random
import torch
from collections import defaultdict
import json
from PIL import Image
import numpy as np

_DATASETS_MAIN_PATH = '/mnt/HD2/yyz/Datasets'
_dataset_path = {
    'cifar10': os.path.join(_DATASETS_MAIN_PATH, 'CIFAR10'),
    'cifar100': os.path.join(_DATASETS_MAIN_PATH, 'CIFAR100'),
    'stl10': os.path.join(_DATASETS_MAIN_PATH, 'STL10'),
    'mnist': os.path.join(_DATASETS_MAIN_PATH, 'MNIST'),
    'imagenet': {
        'train': os.path.join(_DATASETS_MAIN_PATH, 'ImageNet/train'),
        'val': os.path.join(_DATASETS_MAIN_PATH, 'ImageNet/val')
    },
    'femnist': {
        'train': os.path.join(_DATASETS_MAIN_PATH, 'femnist/data/niid_train_'),
        'val': os.path.join(_DATASETS_MAIN_PATH, 'femnist/data/niid_test_')
    },
    'celeba': {
        'train': os.path.join(_DATASETS_MAIN_PATH, 'celeba/data/niid_train_'),
        'val': os.path.join(_DATASETS_MAIN_PATH, 'celeba/data/niid_test_')
    }
}


class MyDataset(data.Dataset):
    def __init__(self, base_dataset):
        super(MyDataset).__init__()
        self.base_dataset = base_dataset
        self.path = os.path.join(_DATASETS_MAIN_PATH, 'celeba/data/raw/img_align_celeba')
    
    def __len__(self):
        # return 500
        return len(self.base_dataset['y'])

    def __getitem__(self, i):
        img_name = self.base_dataset['x'][i]
        img = Image.open(os.path.join(self.path, img_name))
        img = img.resize((84,84)).convert('RGB')
        return (torch.Tensor(np.array(img).transpose(2, 0, 1)), self.base_dataset['y'][i])
        # return (torch.Tensor(self.base_dataset['x'][i]), self.base_dataset['y'][i])


class dataset_index(data.Dataset):
    def __init__(self, base_dataset, indexes=None):
        self.base_dataset = base_dataset
        self.indexes = indexes

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, i):
        index_ori = self.indexes[i]
        return (self.base_dataset[index_ori])


def distribute(dataset, distribution, numclients, numclasses):
    if distribution == 'noniid':
        nc = 3
        n = int(len(dataset) / (nc * numclients))
        distribution = []
        for _ in range(numclients):
            distribution.append([0] * numclasses)
        tmp = [i for i in range(numclasses)] * int(numclients * nc / numclasses)
        random.seed(0)
        random.shuffle(tmp)
        i = 0
        for _tmp in tmp:
            distribution[i // nc][_tmp] += n
            i += 1
        # print(distribution)

    elif distribution == 'unbalanced':
        distribution = []
        for _ in range(20):
            distribution.append([120]*10)
        for _ in range(40):
            distribution.append([60]*10)
        for _ in range(40):
            distribution.append([30]*10)
    elif distribution is None or not (len(distribution) - 1 == numclients):
        n = int(len(dataset) / (numclasses * numclients))
        distribution = []
        for _ in range(numclients):
            distribution.append([n] * numclasses)
    indexes_ = [[]]
    for _ in range(len(distribution) - 1):
        indexes_.append([])
    index = 0
    for _, y in dataset:
        for i in range(len(distribution)):
            if distribution[i][y] > 0:
                distribution[i][y] -= 1
                (indexes_[i]).append(index)
                break
        index += 1
    return indexes_


def get_dataset(name, split='train', transform=None,
                target_transform=None, download=True, distribution=None, numclients=4, dataset_path=None):
    train = (split == 'train')
    if name == 'cifar10':
        dataset = datasets.CIFAR10(root=_dataset_path['cifar10'],
                                   train=train,
                                   transform=transform,
                                   target_transform=target_transform,
                                   download=download)
        numclasses = 10
    elif name == 'cifar100':
        dataset = datasets.CIFAR100(root=_dataset_path['cifar100'],
                                    train=train,
                                    transform=transform,
                                    target_transform=target_transform,
                                    download=download)
        numclasses = 100
    elif name == 'imagenet':
        path = _dataset_path[name][split]
        dataset = datasets.ImageFolder(root=path,
                                       transform=transform,
                                       target_transform=target_transform)
        numclasses = 21841
    elif name == 'MNIST':
        dataset = datasets.MNIST(root=_dataset_path['mnist'],
                                 train=train,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))
                                 ]),
                                 download=download)
        numclasses = 10
    dataset_ = []
    if name == 'femnist' or name == 'celeba':
        path = _dataset_path[name][split]
        if dataset_path is not None:
            path = path+dataset_path
        c, g, data = read_dir(path)
        print(len(c))
        if distribution == 'noniid':
            k = 0
            d_ = []
            for u in c:
                d = MyDataset(data[u])
                d_.append(d)
                k += d.__len__()
                if k > 500:
                    k = 0
                    dataset_.append(torch.utils.data.ConcatDataset(d_))
                    d_ = []
        elif distribution == 'unbalanced':
            k = 0
            d_ = []
            for u in c:
                d = MyDataset(data[u])
                d_.append(d)
                k += 1
                if k < 37 or (k % 2 == 0 and k < 107) or k % 4 ==2:
                    dataset_.append(torch.utils.data.ConcatDataset(d_))
                    d_ = []
        else:
            for u in c:
                dataset_.append(MyDataset(data[u]))
    else:
        indexes_ = distribute(dataset=dataset, distribution=distribution, numclients=numclients, numclasses=numclasses)
        for indexes in indexes_:
            dataset_.append(dataset_index(dataset, indexes=indexes))
    print('Dataset loaded')
    return dataset_

def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda : None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data
