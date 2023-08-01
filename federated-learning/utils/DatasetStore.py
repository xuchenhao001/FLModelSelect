import logging
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from utils.sampling import get_user_indices
from utils.util import ColoredLogger

logging.setLoggerClass(ColoredLogger)
logging.getLogger("werkzeug").setLevel(logging.ERROR)
logger = logging.getLogger(__file__)


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.targets = torch.Tensor([self.dataset.targets[idx] for idx in idxs])

    def classes(self):
        return torch.unique(self.targets)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        data, label = self.dataset[self.idxs[item]]
        return data, label


class LocalDataset:
    def __init__(self):
        self.initialized = False
        self.dataset_name = ""
        self.dataset_train = None
        self.dataset_test = None
        self.image_shape = None
        self.dict_users = None
        self.test_users = None
        self.dis_acc_test = False

    def init_local_dataset(self, dataset_name, dataset_size, num_users, dis_acc_test, noniid_zipf_s,
                           dataset_size_variant):
        dataset_train = None
        dataset_test = None
        real_path = os.path.dirname(os.path.realpath(__file__))
        # load dataset and split users
        if dataset_name == 'mnist':
            trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            data_path = os.path.join(real_path, "../../data/mnist/")
            dataset_train = datasets.MNIST(data_path, train=True, download=True, transform=trans)
            dataset_test = datasets.MNIST(data_path, train=False, download=True, transform=trans)

        elif dataset_name == 'fmnist':
            trans = transforms.Compose([transforms.ToTensor()])
            data_path = os.path.join(real_path, "../../data/fmnist/")
            dataset_train = datasets.FashionMNIST(data_path, train=True, download=True, transform=trans)
            dataset_test = datasets.FashionMNIST(data_path, train=False, download=True, transform=trans)

        elif dataset_name == 'cifar10':
            trans = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            data_path = os.path.join(real_path, "../../data/cifar10/")
            dataset_train = datasets.CIFAR10(data_path, train=True, download=True, transform=trans)
            dataset_test = datasets.CIFAR10(data_path, train=False, download=True, transform=trans)

        elif dataset_name == 'cifar100':
            trans = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025))])
            data_path = os.path.join(real_path, "../../data/cifar100/")
            dataset_train = datasets.CIFAR100(data_path, train=True, download=True, transform=trans)
            dataset_test = datasets.CIFAR100(data_path, train=False, download=True, transform=trans)

        dict_users, test_users = get_user_indices(dataset_name, dataset_train, dataset_test, dataset_size,
                                                  dataset_size_variant, num_users, noniid_zipf_s)

        self.dataset_name = dataset_name
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.image_shape = dataset_train[0][0].shape
        self.dict_users = dict_users
        self.test_users = test_users
        self.dis_acc_test = dis_acc_test
        self.initialized = True

    def load_train_dataset(self, idx, local_bs):
        split_ds = DatasetSplit(self.dataset_train, self.dict_users[idx])
        return DataLoader(split_ds, batch_size=local_bs, shuffle=True)

    def load_test_dataset(self, idxs, local_test_bs):
        if self.dis_acc_test:
            split_ds = DatasetSplit(self.dataset_test, idxs)
            return DataLoader(split_ds, batch_size=local_test_bs)
        else:
            return DataLoader(self.dataset_test, batch_size=local_test_bs)
