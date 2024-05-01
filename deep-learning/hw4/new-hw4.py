import torch
import random
import pandas as pd
import torch.nn as nn

# importing dataset and tokenization





# define models 
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")

class Encoder(nn.Module): 
  def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout): 
    super(Encoder, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers

    self.dropout = nn.Dropout(dropout)
    self.embedding = nn.Embedding(input_size, embedding_size)
    self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, dropout=dropout)
  def forward(self, x):

    embedded = self.dropout(self.embedding(x))
    outputs, (hidden, cell) = self.rnn(embedded)

    return hidden, cell

class Decoder(nn.Module): 
  def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, dropout):
    super(Decoder, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.dropout = nn.Dropout(dropout)
    self.embedding = nn.Embedding(input_size, embedding_size)
    self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, dropout=dropout)
    self.fc = nn.Linear(hidden_size, output_size)
  
  def forward(self, x, hidden, cell):
    x = x.unsqueeze(0)
    embedded = self.dropout(self.embedding(x))
    outputs, (hidden, cell) = self.rnn(embedded, (hidden, cell))
    predictions = self.fc(outputs)
    predictions = predictions.squeeze(0)
    return predictions, hidden, cell

class Seq2Seq(nn.Module): 
  def __init__(self, encoder, decoder): 
    super(Seq2Seq, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
  
  def forward(self, source, target, teacher_force_ratio=0.5):
    batch_size = source.shape[1]
    target_len = target.shape[0]
    target_vocab_size =  0 # len(english.vocab)

    outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
    hidden, cell = self.encoder(source)
    x = target[0]

    for t in range(1, target_len):
      output, hidden, cell = self.decoder(x, hidden, cell)
      outputs[t] = output
      best_guess = output.argmax(1)
      x = target[t] if random.random() < teacher_force_ratio else best_guess

      return outputs


# train said models.