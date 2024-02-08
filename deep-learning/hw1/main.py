import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import PowerTransformer, StandardScaler
import sys
import itertools
from datetime import datetime
import pandas as pd

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
class WiderPerceptron(nn.Module):
    def __init__(self):
        super(WiderPerceptron, self).__init__()
        self.layers = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(3072, 2048),  # Increased width in the first hidden layer
                    nn.ReLU(),
                    nn.Linear(2048, 1024),  # Increased width in the second hidden layer
                    nn.ReLU(),
                    nn.Linear(1024, 512),   # Increased width in the third hidden layer
                    nn.ReLU(),
                    nn.Linear(512, 256),    # Added a fourth hidden layer (increased depth)
                    nn.ReLU(),
                    nn.Linear(256, 10)      # Output layer remains the same
                )
    def forward(self, x):
        return self.layers(x)

def part1(enhanced, epochs):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    train_dataset = datasets.CIFAR10(root="../datasets", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="../datasets", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = MultiPerceptron() if not enhanced else WiderPerceptron()
    model.to(mps_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    training_losses = []

    time = datetime.now()
    for epoch in range(epochs):
        for images, labels in train_loader:
            images = images.to(mps_device)
            labels = labels.to(mps_device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        training_losses.append(loss.item())
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    end_time = datetime.now()

    # plot relevant information
    plt.plot(np.arange(epochs), training_losses)
    plt.show()
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

    print(f"training took {end_time - time}")

def part2():
    pass

if __name__ == "__main__":
    if len(sys.argv) < 2 :
        print("Usage: python3 main.py <part1 or part2>")
        exit()

    partname = sys.argv[1]

    match partname:
        case "part1":
            part1(True if sys.argv[2] == "enhanced" else False, int(sys.argv[3])) # true for enhanced, false for normal
        case "part2":
            part2()
        case _:
            print("Usage: python3 main.py <part1 or part2>")
