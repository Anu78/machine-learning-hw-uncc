import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import time
from torchinfo import summary 
from helpers import unpackHDF
from helpers import distBetweenCoordinates

# select backend based on hardware
global device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("No GPU hardware found.")

class ImageDataset(Dataset):
    def __init__(self, validation):
        self.images, self.coords = unpackHDF("./data/compressed/NC.h5", validation=validation)

        self.images = torch.Tensor(self.images)
        self.images = self.images.permute(0,3,1,2)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        image = np.array(self.images[idx] / 255, dtype=np.float32)
        location = np.array(self.coords[idx], dtype=np.float16)

        return torch.Tensor(image), torch.Tensor(location)

class ClassificationModel(nn.Module):
    def __init__(self):
        super(ClassificationModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 50)

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

class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        # 640x480 input image, 3 channels

        # convolution layers (reduced number and filters)
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=1)
        
        # max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # modified fully connected layer
        self.fc1 = nn.Linear(16 * 120 * 160, 256) # Adjusted for reduced layer size
        self.fc2 = nn.Linear(256, 2) # Output layer

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 16 * 120 * 160) # flatten output

        x = F.relu(self.fc1(x))
        x = self.fc2(x) # output

        return x

def train(epochs, batchSize, shuffle, lr):
    traindataset = ImageDataset(validation=False)
    trainloader = DataLoader(traindataset, shuffle=shuffle, batch_size=batchSize)

    model = ClassificationModel()
    model.to(device)

    lossFn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start = time.time()
    for i in range(epochs):
        totalLoss = 0  # Reset total loss for each epoch

        for images, coords in trainloader:
            images, coords = images.to(device), coords.to(device)
            predCoords = model(images)
            loss = lossFn(predCoords, coords)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            totalLoss += loss.item()

        avg_loss = totalLoss / len(trainloader)
        print(f"Epoch [{i+1}/{epochs}], Average Loss: {avg_loss:.4f}")
    end = time.time()

    print(f"training time: {end-start:.2f}")

    print(f"Saving model to ./data/models/model.pth")
    torch.save(model.state_dict(), "./data/models/model.pth")

def validate():
    # validation dataloader
    validDataset = ImageDataset(validation=True)
    validLoader = DataLoader(validDataset, batch_size=64)

    # load model from file
    model = ClassificationModel()
    model.load_state_dict(torch.load("./data/models/model.pth"))
    model.to(device)
    model.eval()

    total_distance = 0
    total_samples = 0

    with torch.no_grad():
        for images, actual_coords in validLoader:
            images, actual_coords = images.to(device), actual_coords.to(device)

            predicted_coords = model(images)
            predicted_coords = predicted_coords.cpu().numpy()
            actual_coords = actual_coords.cpu().numpy()

            for pred, actual in zip(predicted_coords, actual_coords):
                distance = distBetweenCoordinates((pred[0], pred[1]), (actual[0], actual[1]))
                total_distance += distance
                total_samples += 1

    avg_distance = total_distance / total_samples
    print(f"Average distance error: {avg_distance} miles")
    return avg_distance

def evaluate(image):
    pass

summary(RegressionModel(), (32, 3, 480, 640))