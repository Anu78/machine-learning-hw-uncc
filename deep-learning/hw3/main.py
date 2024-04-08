import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

# device initialization
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")

# Check if the input.txt file exists
if os.path.isfile("input.txt"):
    with open("input.txt", "r") as f:
        text = f.read().lower()
    text = text.replace("\n", " ")
else:
    text = "This is a default text used for demonstration purposes."

# global parameters
epochs = 1000
batch_size = 64
sequence_length = 20  # subject to change
vocabulary = set(text)
vocab_size = len(vocabulary)
decode = {i: c for i, c in enumerate(vocabulary)}
encode = {c: i for i, c in enumerate(vocabulary)}

# define dataset
class TextDataset(Dataset):
    def __init__(self, text, sequence_length):
        self.text = text
        self.sequence_length = sequence_length
        self.x = []
        self.y = []
        for i in range(len(text) - sequence_length):
            self.x.append([encode[char] for char in text[i:i + sequence_length]])
            self.y.append(encode[text[i + sequence_length]])
        self.x = torch.tensor(self.x, dtype=torch.long)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

dataset = TextDataset(text, sequence_length)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

class SimpleRNN(nn.Module):
    def __init__(self, hidden_size, output_size, vocab_size, embed_dim, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True, num_layers=num_layers, dropout=0.1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Get the last time step
        return out

model = SimpleRNN(hidden_size=128, output_size=vocab_size, vocab_size=vocab_size, embed_dim=10, num_layers=3)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    total_val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val, y_val = X_val.to(device), y_val.to(device)
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val)
            total_val_loss += val_loss.item()
            _, predicted = torch.max(val_output, 1)
            total += y_val.size(0)
            correct += (predicted == y_val).sum().item()

    val_acc = correct / total
    if epoch % 100 == 0:
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}, Val Loss: {total_val_loss / len(val_loader)}, Val Acc: {val_acc}')

# Save model
torch.save(model.state_dict(), "./model.pth")

# Inference function
def predict(text, model, device, sequence_length=20):
    model.eval()
    text = text.lower().replace('\n', ' ')
    x = torch.tensor([[encode[char] for char in text]], dtype=torch.long).to(device)
    with torch.no_grad():
        output = model(x)
        _, predicted_idx = torch.max(output, 1)
    return decode[predicted_idx.item()]

# Example usage
test_string = "The meaning of life is"
predicted_char = predict(test_string, model, device, sequence_length)
print(f'Next character: {predicted_char}')
