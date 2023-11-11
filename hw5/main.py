#! /opt/homebrew/bin/python3.11 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt

global mps_device
# check for apple silicon support
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
else:
    print ("MPS device not found.")

class LinearModel(nn.Module):
    def __init__(self, nFeatures):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(nFeatures,1)
    
    def forward(self, x):
        return self.linear(x).squeeze(-1)

class PolynomialModel(nn.Module):
    def __init__(self):
        super(PolynomialModel, self).__init__()
        self.w2 = nn.Parameter(torch.randn(1))
        self.w1 = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return self.w2 * x**2 + self.w1*x + self.b


class HousingDataset(Dataset):
    def __init__(self, csvfile, transform=None, qty = True):
        self.hdata = pd.read_csv(csvfile)
        self.qty = qty
        self.transform = transform
        self.processData()

    def processData(self):
        if self.qty:
            # trim data to area, bedrooms, bathrooms, stories, parking
            self.hdata = self.hdata[["area", "bedrooms", "bathrooms", "stories", "parking", "price"]]

        # normalize pf dataframe 
        self.hdata = (self.hdata-self.hdata.min())/(self.hdata.max()-self.hdata.min())
    
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

def train(model, epochs, optimizer, loss_function, t_u, t_c):
    for epoch in range(epochs+1):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(t_u)

        # Compute and print loss
        loss = loss_function(y_pred, t_c)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    return model

def ntrain(model, epochs, optimizer, loss_function, dataloader, valid_data):
    valid_loader = DataLoader(valid_data, batch_size=128)
    for epoch in range(epochs+1):
        model.train()
        for features, outcomes in dataloader:
            ypred = model(features)
            loss = loss_function(ypred, outcomes)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch % (epochs // 5) == 0:
            # calculate and print validation accuracy
            model.eval()
            with torch.no_grad():
                valid_loss = 0
                for vf, vo in valid_loader:
                    vypred = model(vf)
                    valid_loss += loss_function(vypred, vo).item()

                valid_loss /= len(valid_loader)

            print(f'Epoch {epoch}, Loss: {loss.item()}, Validation Loss: {valid_loss}')


    return model

def main():
    """
    Part 1: 
    """
    # input data, not sure why the temp conversions are inaccurate
    # t_c = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0], dtype=torch.float32,device=mps_device)
    # t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4], dtype=torch.float32, device=mps_device)

    # # reshape data
    # t_c = t_c.view(-1,1)
    # t_u = t_u.view(-1,1)
    
    # model = PolynomialModel()
    # model.to(mps_device)
    # loss_fn = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # trained_model = train(model, 5000, optimizer, loss_fn, t_u, t_c)

    # print(trained_model.state_dict())

    """
    train the housing dataset on the following parameters. area, bedrooms, bathrooms, parking, stories
    """
    # load the validation dataset for all future runs 
    valid_dataset = HousingDataset("/Users/boop/code/python/intro-to-ml-hw/hw5/data/Housing-valid.csv", qty=False)

    # dataset = HousingDataset("/Users/boop/code/python/intro-to-ml-hw/hw5/data/Housing.csv")
    # dataLoader = DataLoader(dataset, 128, shuffle=True)
    # model = LinearModel(5)
    # model.to(mps_device)
    # loss_fn = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # trained_model = ntrain(model, 250, optimizer, loss_fn, dataLoader, valid_dataset)

    # print(trained_model.state_dict())

    """
    train the housing dataset on all parameters. 
    """
    dataset = HousingDataset("/Users/boop/code/python/intro-to-ml-hw/hw5/data/Housing.csv", qty=False) # grabs all data points
    dataLoader = DataLoader(dataset, 128, shuffle=True)
    model = LinearModel(12)
    model.to(mps_device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trained_model = ntrain(model, 250, optimizer, loss_fn, dataLoader, valid_dataset)

    print(trained_model.state_dict())



if __name__ == "__main__":
    main()
