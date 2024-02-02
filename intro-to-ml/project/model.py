import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageGrab
from numpy import asarray
import torchvision.transforms as transforms

latMin = 24
longMin = -125
latMax = 49
longMax = -68

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("No GPU hardware found.")

class HaversineLoss(nn.Module):
    def __init__(self):
        super(HaversineLoss, self).__init__()

    def forward(self, preds, targets):
        # Radius of the Earth in miles
        R = 3956.0

        # Convert degrees to radians
        pi = torch.tensor(torch.pi)
        lat1 = preds[:, 0] * (pi / 180)
        lon1 = preds[:, 1] * (pi / 180)
        lat2 = targets[:, 0] * (pi / 180)
        lon2 = targets[:, 1] * (pi / 180)

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = torch.sin(dlat / 2.0)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2.0)**2
        c = 2 * torch.arcsin(torch.sqrt(a))

        # Calculate distances
        distances = R * c

        # Return the average distance, scaled down
        return torch.mean(distances) / 3300 # scaling factor

class ImageDataset(Dataset):
    def __init__(self, folderPath, csvPath):
        self.labels = pd.read_csv(csvPath)
        self.labels["filename"] = self.labels["filename"].astype(str) + ".png"
        self.folderPath = folderPath
        self.mean = [0.5164, 0.5426, 0.5141]
        self.std = [0.2047, 0.1988, 0.2530]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def normalize_labels(self, latitude, longitude):

        normalized_lat = (latitude - latMin) / (latMax - latMin)
        normalized_lon = (longitude - longMin) / (longMax - longMin)

        return normalized_lat, normalized_lon

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        filename, latitude, longitude = self.labels.iloc[idx]
        image = Image.open(self.folderPath + f"/{filename}").convert('RGB')
        image = self.transform(image)
        normalized_lat, normalized_lon = self.normalize_labels(latitude, longitude)
        label = torch.tensor([normalized_lat, normalized_lon], dtype=torch.float32)

        return image, label
    
class LocationCNN2(nn.Module):
    def __init__(self):
        super(LocationCNN2, self).__init__()
        # Updated model architecture
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Output: 16 x 240 x 320
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Output: 32 x 120 x 160
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Output: 64 x 60 x 80
        )
        self.fc_layers = nn.Sequential(
             nn.Linear(64 * 60 * 80, 128),  # Adjusted to match the new conv layer output
             nn.ReLU(),
             nn.Dropout(0.5),  # Adding dropout for regularization
             nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # Flatten the tensor
        x = self.fc_layers(x)
        return x


class LocationCNN(nn.Module):
    def __init__(self):
        super(LocationCNN, self).__init__()
        # Simplified model architecture
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
             nn.Linear(32 * 120 * 160, 64),  # Adjusted to match the conv layer output
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # Flatten the tensor
        x = self.fc_layers(x)
        return x

def trainLoop(model, epochs, train_loader, valid_loader, optimizer, loss_fn):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for (inputs, labels) in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {avg_loss:.4f}')

        # Validation phase
        model.eval()
        val_loss = 0
        predictions = []
        ground_truths = []
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

                predictions.extend(outputs.cpu().numpy())
                ground_truths.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(valid_loader)
        print(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_val_loss:.4f}')

def train(epochs, batchSize, lr):
    model = LocationCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = HaversineLoss()

    train_dataset = ImageDataset(folderPath="/content/images", csvPath="/content/train.csv")
    train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)

    valid_dataset = ImageDataset(folderPath="/content/valid_images", csvPath="/content/valid.csv")
    valid_loader = DataLoader(valid_dataset, batch_size=batchSize, shuffle=True)

    trainLoop(model, valid_loader=valid_loader, epochs=epochs, train_loader=train_loader, optimizer=optimizer, loss_fn=loss_fn)

    torch.save(model.state_dict(), '/content/model.pth')

def denormalize_labels(normalized_lat, normalized_lon, min_lat, max_lat, min_lon, max_lon):
    latitude = normalized_lat * (max_lat - min_lat) + min_lat
    longitude = normalized_lon * (max_lon - min_lon) + min_lon
    return latitude, longitude

def center_crop_image(image, output_size=(640, 480)):
    # Calculate the coordinates for the center crop
    width, height = image.size
    left = (width - output_size[0])/2
    top = (height - output_size[1])/2
    right = (width + output_size[0])/2
    bottom = (height + output_size[1])/2

    # Crop the center of the image
    img_cropped = image.crop((left, top, right, bottom))

    return img_cropped


def eval(image):
    # Load the model
    model = LocationCNN2().to(device)
    model.load_state_dict(torch.load('./data/models/model_v2.pth', map_location=device))
    model.eval()

    # Normalization parameters
    mean = [0.5164, 0.5426, 0.5141]
    std = [0.2047, 0.1988, 0.2530]

    # Transformation
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    with torch.no_grad():
        image = center_crop_image(image)  # Apply center cropping
        image = transform(image)
        image = image.unsqueeze(0).to(device)

        # Predict
        output = model(image)
        output = output.cpu().numpy()[0]  # Convert to numpy array and get the first item

        # Denormalize the output
        min_lat, max_lat = latMin, latMax  # Replace with actual min/max values
        min_lon, max_lon = longMin, longMax  # Replace with actual min/max values
        latitude, longitude = denormalize_labels(output[0], output[1], min_lat, max_lat, min_lon, max_lon)

        return latitude, longitude

# Usage example
image = ImageGrab.grabclipboard()
latitude, longitude = eval(image)
print(f"Predicted coordinates: lat/long = {latitude}, {longitude}")