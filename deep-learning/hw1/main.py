import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import sys
import itertools

"""
check if device supports mps
"""
mps_device = None
# check for apple silicon support
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
else:
    print("MPS device not found.")

class MultiPerceptron(nn.Module):
    def __init__(self):
        super(MultiPerceptron, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        return self.layers(x)

def part1():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    train_dataset = datasets.CIFAR10(root="../datasets", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="../datasets", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = MultiPerceptron()
    model.to(mps_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    training_losses = []

    for epoch in range(20):
        for images, labels in train_loader:
            images = images.to(mps_device)
            labels = labels.to(mps_device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        training_losses.append(loss.item())
        print(f"Epoch [{epoch+1}/{20}], Loss: {loss.item():.4f}")


    # plot relevant information
    plt.plot(np.arange(20), training_losses)
    # run validation accuracy
    true_labels = []
    predictions = []

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
                images = images.to(mps_device)
                labels = labels.to(mps_device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.view(-1).tolist())
                true_labels.extend(labels.view(-1).tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average="macro")

    conf_matrix = confusion_matrix(true_labels, predictions)

    print(f'Precision: {precision}\nRecall: {recall}\nF1 Score: {f1}')
    print(f'Confusion Matrix:\n{conf_matrix}')

def part2():
    pass
if __name__ == "__main__":
    if len(sys.argv) < 2 :
        print("Usage: python3 main.py <part1 or part2>")
        exit()

    partname = sys.argv[1]

    match partname:
        case "part1":
            part1()
        case "part2":
            part2()
        case _:
            print("Usage: python3 main.py <part1 or part2>")
