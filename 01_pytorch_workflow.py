import torch
from torch import nn
import matplotlib.pyplot as plt

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02
x = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * x + bias

# splitting data into training set and testing set
train_split = int(0.8 * len(x))
x_train, y_train = x[:train_split], y[:train_split]
x_test, y_test = x[train_split:], y[train_split:]

def plot_predictions(
        train_data=x_train,
        train_lables=y_train,
        test_data=x_test,
        test_lable=y_test,
        predictions=None):
    plt.figure(figsize=(10,7))

    plt.scatter(train_data, train_lables, c="b", s=4, label="Training data")

    plt.scatter(test_data, test_lable, c="g", s=4, label="Testing data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="predictions")
    
    plt.legend(prop={"size" : 14});

# plot_predictions();
# plt.show()

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # initiallizing model parameters
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float32, requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float32, requires_grad=True))

        # all subclasses must override this forward method of nn.module, this defines the forward computation of model
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias 



torch.manual_seed(42)
model_0 = LinearRegressionModel()
# print(list(model_0.parameters()))
print(model_0.state_dict())

# with torch.inference_mode():
#     y_pred = model_0(x_test)
#     plot_predictions(predictions=y_pred)
#     plt.show()

# setting up loss function and optimixer
loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(model_0.parameters(), lr=0.01)

epochs = 168

for epoch in range(epochs):
    model_0.train()

    y_pred = model_0(x_train)

    loss = loss_fn(y_pred, y_train)
    # print(f"loss : {loss}")

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    model_0.eval()

    print(model_0.state_dict()) 

with torch.inference_mode():
    y_pred = model_0(x_test)
    plot_predictions(predictions=y_pred)
    plt.show()



