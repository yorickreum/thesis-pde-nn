import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Net(nn.Module):

    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))  # activation function for hidden layer
        x = self.predict(x)  # linear output
        return x


net = Net(n_feature=1, n_hidden=2, n_output=1)

print("Network")
print(net)

print("\nParameters")
params = list(net.parameters())
print(len(params))
print(params[0].size())

netInput = torch.tensor([1], dtype=torch.float)
# netInput = torch.randn(1)

# In-Out-Test
with torch.no_grad():
    print("\nIn-Out-Test")
    print("Input ", netInput)
    out = net(netInput)
    print("Output ", out)
    print("\n")


# criterion = nn.MSELoss()
# optimizer = optim.SGD(net.parameters(), lr=0.05)
# in your training loop:

def target_func(x):
    return 5 * x + 40 * x**2


optimizer = torch.optim.SGD(net.parameters(), lr=.005)
# loss_func = torch.nn.MSELoss()
loss_func = torch.nn.L1Loss()

x = torch.linspace(-5, 5, 100)
x = torch.unsqueeze(x, dim=1)
y = torch.tensor([[target_func(xi)] for xi in x])
x, y = Variable(x), Variable(y)

# with torch.no_grad():
#     plt.plot(x.data.numpy(), y.data.numpy())
#     plt.show()

losses = []
for i in range(int(1e4)):
    prediction = net(x)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients
    with torch.no_grad():
        losses += [loss.data.numpy()]
        print(loss)

# In-Out-Test
# netInput = torch.tensor([1], dtype=torch.float)
with torch.no_grad():
    print("\nIn-Out-Test")
    print("Input ", netInput)
    out = net(netInput)
    print("Output ", out)

with torch.no_grad():
    plt.plot(x, y)
    yPredicted = [net(x) for x in x]
    plt.plot(x, yPredicted)
    plt.show()

plt.plot(
    [i for i in range(len(losses))],
    [i for i in losses]
)
plt.show()
