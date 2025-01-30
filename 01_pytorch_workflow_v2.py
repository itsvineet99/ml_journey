import torch
from torch import nn
import matplotlib.pyplot as plt
import torch.optim.sgd

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")


# CREATING DATA FOR MODEL

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

x = torch.arange(start, end, step).unsqueeze(dim=1)
# print(x, x.shape)
y = weight * x + bias

# for i in range(len(x)):
#     print(f"x: {x[i].item():.4f} | y: {y[i].item():.4f}")


# SPLITTING DATA

train_split = int(0.8 * len(x))
x_train, y_train = x[:train_split], y[:train_split]
x_test, y_test = x[train_split:], y[train_split:]

# print(len(x_train), len(x_test), len(y_train), len(y_test))


# CREATING FUNCTION TO PLOT OUR DATA

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

# plot_predictions(x_train, y_train, x_test, y_test)
# plt.show()


# BUILDING MODEL

class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)
    

# INITIALIZING MODEL

torch.manual_seed(42)
model_1 = LinearRegressionModelV2()
model_1.to(device=device) # putting our tensors on mps device from cpu
# print(model_1, model_1.state_dict())
# print(next(model_1.to(device).parameters()).device)


# TRAINING LOOP -> SETTING UP LOSS FUNCTION AND OPTIMIZER AND TRAINING MODEL

loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.01)

torch.manual_seed(42)

# TRAINING
epochs = 120

# putting all our data tensors into mps device
x_train = x_train.to(device)
x_test = x_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

for epoch in range(epochs):
    model_1.train()

    y_pred = model_1(x_train)

    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    
    # TESTING
    model_1.eval()
    with torch.inference_mode():
        test_pred = model_1(x_test)

        test_loss = loss_fn(test_pred, y_test)

        # if epoch % 10 == 0:
        #     print(f"epoch : {epoch} | loss : {loss} | test loss : {test_loss}")

# print(model_1.state_dict())


# PLOTTING THE GRAPH FOR NEW PREDICTED VALUES

with torch.inference_mode():
    y_preds = model_1(x_test)

plot_predictions(predictions=y_preds.cpu()) 
# plt.show()


# SAVING OUR MODEL

model_path = "models/model_02.pth"

print(f"saving model to: {model_path}")

torch.save(obj=model_1.state_dict(), f=model_path)

# LOADING OUT MODEL

loaded_model_1 = LinearRegressionModelV2()

loaded_model_1.load_state_dict(torch.load(model_path, weights_only=True))

loaded_model_1.to(device)

loaded_model_1.eval()

loaded_model_1_preds = loaded_model_1(x_test)
print(y_preds == loaded_model_1_preds)

