import torch

from laplace_v3.func_lib import yorick_delta_relu_sq

x = torch.tensor(0.5, requires_grad=True)
y = yorick_delta_relu_sq(x)
y.backward()
print(x.grad)

x = torch.tensor(2., requires_grad=True)
y = yorick_delta_relu_sq(x)
y.backward()
print(x.grad)

x = torch.tensor(0., requires_grad=True)
y = yorick_delta_relu_sq(x)
y.backward()
print(x.grad)

x = torch.tensor(1., requires_grad=True)
y = yorick_delta_relu_sq(x)
y.backward()
print(x.grad)
