### resnet training for cifar-10 dataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import time

"""
check if device supports mps
"""
mps_device = None
# check for apple silicon support
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")  # temporarily disabled mps
elif torch.cuda.is_available():
    mps_device = torch.device("cuda")
else:
    print("MPS device not found.")

# mps_device = torch.device("cuda")

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection to downsample residual
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Creating layers of blocks
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Average Pooling and Fully Connected Layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# Define the ResNet-10 architecture (2+2+2+2+2 blocks)
def ResNet10():
    return ResNet(BasicBlock, [2, 2, 2, 2, 2])


def part2():
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
    # model = CifarModel2() if model_complexity else CifarModel1()
    model = ResNet10()
    model.to(mps_device)

    # setup optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

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

def main():
    part2()


main()
