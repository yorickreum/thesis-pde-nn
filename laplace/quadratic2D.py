import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 8)  # 6*6 from image dimension
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
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

# netInput = torch.tensor([1, 1], dtype=torch.float)
# netInput = torch.randn(1)

# In-Out-Test
with torch.no_grad():
    print("\nIn-Out-Test")
    print("Input ", netInput)
    out = net(netInput)
    print("Output ", out)
#    print("\n")


# criterion = nn.MSELoss()
# optimizer = optim.SGD(net.parameters(), lr=0.05)
# in your training loop:

optimizer = optim.LBFGS(net.parameters(), lr=0.05)

for i in range(50):
    def closure():
        optimizer.zero_grad()  # zero the gradient buffers
        output = net(netInput)
        # loss = 2 + 5 * output + 40 * torch.pow(output, 2) - 8 * torch.pow(output, 3)
        x1 = output[0]
        x2 = output[1]
        loss = (pow((x1 - 5.8), 2) + pow((x2 + 3.2), 2))
        loss.backward()
        return loss
    loss = optimizer.step(closure)  # Does the update

# In-Out-Test
# netInput = torch.tensor([1], dtype=torch.float)
with torch.no_grad():
    print("\nIn-Out-Test")
    print("Input ", netInput)
    out = net(netInput)
    print("Output ", out)
