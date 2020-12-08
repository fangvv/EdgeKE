'''
获得caifar_10数据
'''
import torch
import torchvision
import torchvision.transforms as transforms
from functions import  my_functions


def get_data(train_batch_size=128, test_batch_size=128):

    Project_dir = my_functions.get_project_dir()
    root_dir = Project_dir + "/datasets/data/"

    transform_train = transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])

    transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])

    train_set = torchvision.datasets.CIFAR100(root=root_dir, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = train_batch_size, shuffle=True, num_workers=3)

    test_set = torchvision.datasets.CIFAR100(root=root_dir, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = test_batch_size, shuffle=False, num_workers=3)

    return train_loader, test_loader