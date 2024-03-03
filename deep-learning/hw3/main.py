import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# device initialization
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")

# read input into one string. 
with open("input.txt") as f: 
    text = f.read() 
    text = text.lower() 

text.replace("\n", " ")

# global parameters
epochs = 100
batch_size = 1
sequence_length = 10 # subject to change

class CharDataset(Dataset): 
    def __init__(self, text, sequence_length):
        self.data = text 
        self.char_space = sorted(set(text))
        self.encode = {c:i for i,c in enumerate(self.char_space)} 
        self.decode = {i:c for i,c in self.encode.items()} 
        self.vocab_size = len(self.char_space)
        self.seq_length = sequence_length
    
    def char_to_tensor(self, chars): 
        lst = [self.encode[i] for i in chars]
        tensor = torch.tensor(lst, dtype=torch.long) 
        return tensor

    def __len__(self): 
        return len(self.data) - self.seq_length - 1 

    def __getitem__(self, idx): 
        seq = self.data[idx:idx+self.seq_length]
        target_seq = self.data[idx + 1: idx + self.seq_length + 1]

        x = self.char_to_tensor(seq)
        y = self.char_to_tensor(target_seq)

        return x,y

class SimpleRNN(nn.Module):
    def __init__(self, hidden_size, output_size, vocab_size, embed_dim):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size

        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim=embed_dim)
        
        # RNN Cell
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden):
        x = self.embedding(x)

        # Update the hidden state
        out, hidden = self.rnn(x, hidden)
        
        # Compute the output
        out = self.fc(out.contiguous().view(-1, self.hidden_size))
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        # Initialize hidden state with zeros
        return torch.zeros(1, batch_size, self.hidden_size).to(device)

text_dataset = CharDataset(text, sequence_length=10)
data_loader = DataLoader(text_dataset, batch_size=batch_size, shuffle=True)
vocab_size = len(set(text))
model = SimpleRNN(hidden_size=20, output_size=vocab_size, vocab_size=vocab_size, embed_dim=50)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# Training loop
for epoch in range(epochs):
    model.train()  # Set model to training mode
    total_loss = 0

    for batch, (x, y) in enumerate(data_loader):
        x = x.to(device)
        y = y.to(device).view(-1)  # Flatten y to match the output dimensions
        optimizer.zero_grad()  # Clear gradients for this training step

        # Initializing hidden state for each batch; necessary due to variable sequence lengths
        hidden = model.init_hidden(batch_size)
        hidden = hidden.data  # Detach hidden state from its history

        output, hidden = model(x, hidden)  # Forward pass: Compute predicted y by passing x to the model
        
        # Compute loss. Here we're assuming that your RNN outputs a prediction at each time step
        loss = criterion(output, y)  # No need to flatten y anymore as we did it before
        loss.backward()  # Backward pass: compute gradient of the loss with respect to model parameters
        optimizer.step()  # Perform a single optimization step (parameter update)

        total_loss += loss.item()  # Sum up batch loss

    # Print average loss for the epoch
    average_loss = total_loss / len(data_loader)
    print(f'Epoch {epoch+1}/{epochs}, Loss: {average_loss:.4f}')

    # You can add any validation logic here

# save model 
torch.save(model.state_dict(), "./model.pth")