import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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


def train_final():
    train_loader, test_loader = get_loaders()

    model = FinalModel(GENOTYPE).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(20):
        model.train()
        total = 0
        correct = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            _, predicted = pred.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        train_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/20 :: Train Acc = {train_acc:.2f}%")

    # Final test accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            _, predicted = pred.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

    print(f"Final Test Accuracy = {100 * correct / total:.2f}%")

    torch.save(model.state_dict(), "final_model.pth")


if __name__ == "__main__":
    train_final()