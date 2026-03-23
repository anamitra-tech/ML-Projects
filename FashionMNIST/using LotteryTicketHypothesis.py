import torch
import torch.nn as nn
import torch.optim as optim

# Simple model
class Net(nn.Module): #this nn.module is required as this tells us that now this class belongs to neural network
  #and thus will help in tracking the parameters it knows that model.parameters() without nn.module
  
    def __init__(self, input_dim=784, hidden=256, output=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, output)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def train(model, dataloader, epochs=3):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()


#loss_fn = nn.CrossEntropyLoss()
#model.train()
#optimizer = optim.Adam(model.parameters(), lr=1e-3)
#this also works fine
  
    for _ in range(epochs):#this _ is important as the loop variable is not of any use
        for x, y in dataloader:
            optimizer.zero_grad()#this is required as we need to clear the gradient before the next epoch starts and we dont 
          #gradients to accumulate as we want new updated gradients not the accumulated ones
          #this is an important bug leading to bad training 
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

#pruning happens globally across the network, not layer by layer
#Because not all layers are equally important.

#Example:

#Sometimes:

#Layer 1 might contain many weak weights
#Layer 2 might contain mostly strong weights

#Layer-wise pruning forces both to lose weights equally.
def prune_by_magnitude(model, prune_percent):
    all_weights = []

    # collect weights
    for p in model.parameters():
        all_weights.append(p.data.abs().flatten())

    all_weights = torch.cat(all_weights)
    threshold = torch.quantile(all_weights, prune_percent)

    mask = []
    for p in model.parameters():
        m = (p.data.abs() > threshold).float()
        mask.append(m)
        p.data *= m  # apply pruning

    return mask


def reset_to_initial(model, initial_weights, mask):
    i = 0
    for p in model.parameters():
        p.data = initial_weights[i] * mask[i]
        i += 1


# ---- LTH Algorithm ----

model = Net()

# Save initial weights
initial_weights = [p.data.clone() for p in model.parameters()]

for iteration in range(5):

    print(f"Iteration {iteration}")

    train(model, train_loader)

    mask = prune_by_magnitude(model, prune_percent=0.2)

    reset_to_initial(model, initial_weights, mask)
