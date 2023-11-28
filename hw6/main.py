import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import time

"""
check if device supports mps
"""
global mps_device
# check for apple silicon support
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")  # temporarily disabled mps
else:
    print("MPS device not found.")


# create dataloader class for housing.csv, housing-valid.csv
class HousingDataset(Dataset):
    def __init__(self, csvfile, transform=None, qty=True):
        self.hdata = pd.read_csv(csvfile)
        self.qty = qty
        self.transform = transform
        self.processData()

    def processData(self):
        if self.qty:
            # trim data to area, bedrooms, bathrooms, stories, parking
            self.hdata = self.hdata[
                ["area", "bedrooms", "bathrooms", "stories", "parking", "price"]
            ]

        # normalize pf dataframe
        self.hdata = (self.hdata - self.hdata.min()) / (
            self.hdata.max() - self.hdata.min()
        )

    def __len__(self):
        return len(self.hdata)

    def __getitem__(self, i):
        sample = self.hdata.iloc[i]

        outcome = sample.pop("price")
        features = sample.values.astype(float)

        features = torch.tensor(features, dtype=torch.float32, device=mps_device)
        outcome = torch.tensor(outcome, dtype=torch.float32, device=mps_device)

        if self.transform:
            features = self.transform(features)

        return features, outcome


class HousingModel1(nn.Module):
    def __init__(self):
        super(HousingModel1, self).__init__()
        self.fc1 = nn.Linear(12, 32)  # input layer
        self.fc2 = nn.Linear(32, 1)  # output layer (price)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x.squeeze(-1)


class HousingModel2(nn.Module):
    def __init__(self):
        super(HousingModel2, self).__init__()
        self.fc1 = nn.Linear(12, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x.squeeze(-1)


def ntrain(model, epochs, optimizer, loss_function, dataloader, valid_data):
    valid_loader = DataLoader(
        valid_data,
        batch_size=128,
    )
    for epoch in range(epochs + 1):
        model.train()
        for features, outcomes in dataloader:
            ypred = model(features)
            loss = loss_function(ypred, outcomes)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % (epochs // 10) == 0:
            # calculate and print validation accuracy
            model.eval()
            with torch.no_grad():
                valid_loss = 0
                for vf, vo in valid_loader:
                    vypred = model(vf)
                    valid_loss += loss_function(vypred, vo).item()

                valid_loss /= len(valid_loader)

            print(f"Epoch {epoch}, Loss: {loss.item()}, Validation Loss: {valid_loss}")

    return model


class CifarModel1(nn.Module):
    def __init__(self):
        super(CifarModel1, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 32 * 3, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class CifarModel2(nn.Module):
    def __init__(self):
        super(CifarModel2, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 32 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)  # Flatten
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def part1(model_complexity: bool) -> None:
    """
    Create a fully connected neural network to predict price based on all parameters in the housing dataset.

    model_complexity: false chooses the less complex model.
    """
    model = HousingModel2() if model_complexity else HousingModel1()

    valid_dataset = HousingDataset(
        "/Users/boop/code/python/intro-to-ml-hw/hw5/data/Housing-valid.csv", qty=False
    )
    dataset = HousingDataset(
        "/Users/boop/code/python/intro-to-ml-hw/hw5/data/Housing.csv", qty=False
    )  # grabs all data points
    dataLoader = DataLoader(dataset, batch_size=128)
    model.to(mps_device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    trained_model = ntrain(model, 1000, optimizer, loss_fn, dataLoader, valid_dataset)


def part2(model_complexity: bool):
    # load cifar10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, transform=transform, download=True
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=100, shuffle=True
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=100, shuffle=False
    )

    # define & move model
    model = CifarModel2() if model_complexity else CifarModel1()
    model = CNN()
    model.to(mps_device)

    # setup optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    # custom training loop
    start_time = time.time()
    for epoch in range(20):
        for images, labels in train_loader:
            images = images.to(mps_device)
            labels = labels.to(mps_device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{20}], Loss: {loss.item():.4f}")
    end_time = time.time()

    # evaluate model performance
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(mps_device)
            labels = labels.to(mps_device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = (correct / total) * 100

    total_time = end_time - start_time

    print(f"took {total_time:.2f} seconds to train, and", end=" ")
    print(f"final model accuracy is {accuracy:.2f}%")


if __name__ == "__main__":
    part2(True)
