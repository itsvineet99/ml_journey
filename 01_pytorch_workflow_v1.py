import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import os

''' CREATING A FAKE DATASET FOR LEARNING PURPOSE '''

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02
x = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * x + bias

''' splitting data into training set and testing set '''

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

''' BUILDIN A LINEAR REGRESSION MODEL '''

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # initiallizing model parameters
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float32, requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float32, requires_grad=True))

        # all subclasses must override this forward method of nn.module, this defines the forward computation of model
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias 



''' TRAINING A LINEAR REGRESSION MODEL (training loop) '''

torch.manual_seed(42)
model_0 = LinearRegressionModel()

# setting up loss function and optimixer
loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(model_0.parameters(), lr=0.01)

epochs = 168

epoch_count = []
train_loss_count = []
test_loss_count = []

for epoch in range(epochs):
    model_0.train()

    y_pred = model_0(x_train)

    loss = loss_fn(y_pred, y_train)
    # print(f"loss : {loss}")

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    model_0.eval()

'''   
    #  testing the model (testing loop) 
    with torch.inference_mode():
        test_pred = model_0(x_test)

        test_loss = loss_fn(test_pred, y_test)

        if epoch % 10 == 0:

            epoch_count.append(epoch)
            train_loss_count.append(loss)
            test_loss_count.append(test_loss)

            print(f"epoch : {epoch} | loss : {loss} | test_loss : {test_loss}")
            print(model_0.state_dict()) 


with torch.inference_mode():
    plt.plot(epoch_count, train_loss_count, label="Train loss")
    plt.plot(epoch_count, test_loss_count, label="Test loss")
    plt.title("Training and test loss curves")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

        # plot_predictions(predictions=y_pred)
        # plt.show() '''

model_dir = "models"
model_path = os.path.join(model_dir, "model_0.pth")

torch.save(model_0.state_dict(), model_path)

loaded_model_0 = LinearRegressionModel()

loaded_model_0.load_state_dict(torch.load(model_path,  weights_only=True))

# print(loaded_model_0.state_dict())

loaded_model_0.eval()
with torch.inference_mode():
    old_model_preds = model_0(x_test)
    loaded_model_preds = loaded_model_0(x_test)

# print(loaded_model_preds)

print(old_model_preds == loaded_model_preds)
