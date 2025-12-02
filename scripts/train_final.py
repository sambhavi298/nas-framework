import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from nas.models.final_model import FinalModel

GENOTYPE = ['conv_3x3', 'skip', 'conv_3x3', 'conv_3x3', 'conv_3x3', 'conv_3x3']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])

    trainset = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    testset = datasets.CIFAR10("./data", train=False, download=True, transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader