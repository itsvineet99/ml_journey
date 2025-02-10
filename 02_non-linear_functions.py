import torch
from matplotlib import pyplot as plt

a = torch.arange(-10, 10, 1, dtype=torch.float)
def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.maximum(torch.tensor(0), x) # returns maximum between 0 and x

# print(a, relu(a))
# plt.plot(relu(a))
# plt.show()

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1/(1 + torch.exp(-x))

plt.plot(sigmoid(a))
plt.show()