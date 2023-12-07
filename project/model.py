import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms, datasets, models

# select backend based on hardware
global device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("No GPU hardware found.")

class RemoveGoogleLogo(object):
    def __init__(self, num_pixels):
        self.num_pixels = num_pixels

    def __call__(self, img):
        """
        Args:
            img (Tensor): Image tensor of size (C, H, W).

        Returns:
            Tensor: Transformed image with bottom pixels removed.
        """
        # Check if the image has enough pixels to remove
        if img.size(1) > self.num_pixels:
            img = img[:, :-self.num_pixels, :]
        return img

def train(epochs, batchSize, shuffle, lr):
    transform = transforms.Compose([
    transforms.ToTensor()
])

    train_dataset = datasets.ImageFolder(root='./data/compressed/images', transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize, shuffle=shuffle)

    # Similar for validation and test datasets

    # 2. Model Architecture
    model = models.resnet18(pretrained=True)
    
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 50)  # 50 classes for 50 states

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        average_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{epochs} - Average Loss: {average_loss:.4f}")

    print(f"Saving model to ./data/models/model.pth")
    torch.save(model.state_dict(), "./data/models/model.pth")

def validate(model, batch_size=32):
    # Define the model architecture (should match the architecture of the saved model)
    # model = models.resnet18()
    # num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, 50)  # 50 classes for 50 states

    # # Load the model from the .pth file
    # model.load_state_dict(torch.load(model_path, map_location=device))
    # model.to(device)

    # Load the validation dataset
    transform = transforms.Compose([transforms.ToTensor(), RemoveGoogleLogo(22)])
    val_dataset = datasets.ImageFolder(root='/content/valid_images', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Ensure the model is in evaluation mode
    model.eval()

    # Variables to store predictions and labels
    all_preds = []
    all_labels = []

    with torch.no_grad():
            acc = 0
            count = 0
            for inputs, labels in val_loader:
                y_pred = model(inputs)
                acc += (torch.argmax(y_pred, 1) == labels).float().sum()
                count += len(labels)
            acc /= count
            print(f"Accuracy: {acc:.4f}")

def evaluate(image):
    pass