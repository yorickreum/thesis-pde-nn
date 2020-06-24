import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 1, bias=False)  # 6*6 from image dimension

    def forward(self, x):
        x = self.fc1(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()

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

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)
# in your training loop:
for i in range(500):
    optimizer.zero_grad()  # zero the gradient buffers
    # print('fc1.bias.grad before backward', net.fc1.bias.grad)
    output = net(netInput)
    target = torch.tensor([2], dtype=torch.float)
    loss = criterion(output, target)
    loss.backward()
    # print('fc1.bias.grad after backward', net.fc1.bias.grad)
    optimizer.step()  # Does the update

# In-Out-Test
netInput = torch.tensor([1], dtype=torch.float)
with torch.no_grad():
    print("\nIn-Out-Test")
    print("Input ", netInput)
    out = net(netInput)
    print("Output ", out)
