import torch
from torch._dynamo.utils import identity
from torch.nn.modules import dropout
from torch.utils.data import DataLoader
from torchvision import transforms 
import torchvision.datasets
import sys
import torch.nn as nn

# determine gpu device 
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda") 

class SimpleAlexNet(nn.Module):
    def __init__(self, num_classes=10, dropout=0.5):
        super(SimpleAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  
        )
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))  
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 2 * 2, 1024),  
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, 1024),  
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x 

class Block(nn.Module): 
    def __init__(self, inC, outC, identDownsample=None, stride=1): 
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(inC, outC, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(outC)
        self.conv2 = nn.Conv2d(outC, outC, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(outC)
        self.relu = nn.ReLU(inplace=True) 
        self.identDownsample = identDownsample
    def forward(self, x): 
        identity = x
        x = self.conv1(x) 
        x = self.bn1(x) 
        x = self.relu(x) 
        x = self.conv2(x) 
        x = self.bn2(x) 

        if self.identDownsample is not None: 
            identity = self.identDownsample(identity) 

class ResNet18(nn.Module): 
    def __init__(self): 
        super(ResNet18).__init__()
        pass 
    def forward(self, x): 
        pass


def train(model, epochs, trainloader, testloader):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs): 
        model.train() 
        runningLoss = 0

        for i, data in enumerate(trainloader, 1):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs) 
            loss = criterion(outputs, labels)
            loss.backward() 
            optimizer.step()

            runningLoss += loss.item() 

        print(f"epoch {epoch + 1}/{epochs} - training loss: {runningLoss / len(trainloader)}")
        
        model.eval() 
        correct = 0
        total = 0
        with torch.no_grad(): 
            for data in testloader: 
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item() 

        print(f'Epoch {epoch + 1}/{epochs} - Validation Accuracy: {100 * correct / total}%')

def part1(epochs):
    """
    simplify the AlexNet model to work on 32x32 images (cifar) 
    """
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

    trainset = torchvision.datasets.CIFAR10(root="../datasets", train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root="../datasets", train=False, download=True, transform=transform)
    
    testloader = DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

    model_dropout = SimpleAlexNet()
    model_no_dropout = SimpleAlexNet(dropout=0)

    # train model and report 
    print("training model with dropout")
    train(model_dropout, epochs, trainloader, testloader)

    print("training model without dropout")
    train(model_no_dropout, epochs, trainloader, testloader)


def part2():
    """
    build a resnet-18 and train on cifar. compare with results from resnet-11 and   
    experiment with dropout for both networks
    """
    transform = transforms.Compose([

    trainset = torchvision.datasets.CIFAR10(root="../datasets", train=True, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root="../datasets", train=False, download=True, transform=transform)
    
    testloader = DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

    # start model training

def main():
    a = sys.argv
    

    if len(sys.argv) < 2:
        print("incorrect number of args")
        return

    if a[1] == "1": 
        part1(int(a[2])) 
    else: 
        part2() 

if __name__ == "__main__":
    main()
