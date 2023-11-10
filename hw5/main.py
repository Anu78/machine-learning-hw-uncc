#! /opt/homebrew/bin/python3.11 
import torch
import torch.nn as nn

# check for apple silicon support
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
else:
    print ("MPS device not found.")

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1,1)

    def forward(self, x):
        return self.linear(x)


def train(model, epochs, optimizer, loss_function, t_u, t_c):
    for epoch in range(epochs):
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

def main():
    # input data, not sure why the temp conversions are inaccurate
    t_c = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0], dtype=torch.float32,device=mps_device)
    t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4], dtype=torch.float32, device=mps_device)

    # reshape data
    t_c = t_c.view(-1,1)
    t_u = t_u.view(-1,1)
    
    model = LinearModel()
    model.to(mps_device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

    trained_model = train(model, 5000, optimizer, loss_fn, t_u, t_c)

    print(trained_model.state_dict())



if __name__ == "__main__":
    main()
