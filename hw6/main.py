import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

"""
check if device supports mps
"""
global mps_device
# check for apple silicon support
if torch.backends.mps.is_available():
    mps_device = torch.device("cpu")  # temporarily disabled mps
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


"""
grab the unpickled dataset
"""


def unpickle(file):
    import pickle

    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def part1(model_complexity: bool) -> None:
    """
    Create a fully connected neural network to predict price based on all parameters in the housing dataset.
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


def part2():
    pass


if __name__ == "__main__":
    part2()
