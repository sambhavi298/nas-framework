import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import mobilenet_v2
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

def load_data():
    train = CIFAR10(root="./data", train=True, download=False, transform=ToTensor())
    val   = CIFAR10(root="./data", train=False, download=False, transform=ToTensor())
    return (
        DataLoader(train, batch_size=128, shuffle=True),
        DataLoader(val, batch_size=128, shuffle=False),
    )

def train():
    train_loader, val_loader = load_data()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = mobilenet_v2(num_classes=10).to(device)
    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(2):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()

        print("epoch:", epoch, "val_acc:", validate(model, val_loader, device))

def validate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

if __name__ == "__main__":
    train()
