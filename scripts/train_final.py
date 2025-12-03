import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from nas.models.final_model import FinalModel

GENOTYPE = ['conv_3x3', 'skip', 'conv_3x3', 'conv_3x3', 'conv_3x3', 'conv_3x3']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_loaders(batch_size=96):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10("./data", train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10("./data", train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


def train_final():
    train_loader, test_loader = get_loaders()

    model = FinalModel(GENOTYPE).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Improved Optimizer: SGD with Momentum
    optimizer = optim.SGD(model.parameters(), lr=0.025, momentum=0.9, weight_decay=3e-4)
    
    # Scheduler: Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    epochs = 100
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total = 0
        correct = 0
        train_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = pred.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        scheduler.step()
        train_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} :: Train Loss = {train_loss/len(train_loader):.4f} | Train Acc = {train_acc:.2f}%")

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

    acc = 100 * correct / total
    print(f"Final Test Accuracy = {acc:.2f}%")

    torch.save(model.state_dict(), "final_model.pth")


if __name__ == "__main__":
    train_final()