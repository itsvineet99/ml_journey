import sklearn
from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import requests
from pathlib import Path    


# print(sklearn.__version__)

n_samples = 1000
x, y = make_circles(n_samples, noise=0.03, random_state=42)
# print(x[:5], y[:5])
# print(x.shape, y.shape)

circles = pd.DataFrame({ "x1" : x[:,0],
                         "x2": x[:, 1],
                         "lable": y }) 

# print(circles.head(10))

# plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.RdYlBu)
# plt.show()


# TURN DATA INTO TENSORS

x = torch.from_numpy(x).type(torch.float)
# print(x[:5], x.dtype)
y = torch.from_numpy(y).type(torch.float)
# print(y[:5], y.dtype)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# SETTING DEVICE AGNOSTIC CODE

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

# SETTING UP A MODEL

class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(in_features=2, out_features=5)
        self.layer2 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        return self.layer2(self.layer1(x)) # layer1 -> layer2 -> output


# another method for creating model
# model_1 = nn.Sequential(
#     nn.Linear(in_features=2, out_features=5),
#     nn.Linear(in_features=2, out_features=1)
# ).to(device)


# model_1 = CircleModelV0().to(device)
# print(model_1, next(model_1.parameters()).device)

# CREATING LOSS FUNCTION AND OPTIMIZER

# loss_fn = nn.BCEWithLogitsLoss()

# optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.1)

# def accuracy_fn(y_true, y_preds):
#     correct = torch.eq(y_true, y_preds).sum().item()
#     acc = (correct/len(y_preds)) * 100
#     return acc


# TRAINING MODEL

# model_1.eval()
# with torch.inference_mode():
#     y_logits = model_1(x_test.to(device))[:5]

# y_pred_probs = torch.sigmoid(y_logits)

# y_preds = torch.round(y_pred_probs)

# print(y_preds)

# torch.manual_seed(42)

# epochs = 100

# x_train, y_train = x_train.to(device), y_train.to(device)
# x_test, y_test = x_test.to(device), y_test.to(device)

# for epoch in range(epochs):
#     model_1.train()

#     y_logits = model_1(x_train).squeeze()

#     y_preds = torch.round(torch.sigmoid(y_logits)) # converts logits into prediction probabilities using sigmoid functin and then convert it into lables i.e 1's and 0's


#     loss = loss_fn(y_logits, y_train)
#     acc = accuracy_fn(y_train, y_preds)

#     optimizer.zero_grad()

#     loss.backward()

#     optimizer.step()

#     model_1.eval()
#     with torch.inference_mode():
#         test_logits = model_1(x_test).squeeze()

#         test_preds = torch.round(torch.sigmoid(test_logits))

#         test_loss = loss_fn(test_logits, y_test)

#         test_acc = accuracy_fn(y_test, test_preds)

#         if epoch % 10 == 0:
#             print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")



# importing some helper funtions to visualize

if Path("helper_functions.py").is_file():
    print("file already exists")
else:
    print("download helper_function.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/helper_functions.py")

    with open("helper_functions.py", "wb") as f:
        f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary

def model_plot(model, x_train, y_train,x_test,y_test):
    plt.figure(figsize=(12, 6))
    plt.subplot(1,2,1)
    plt.title("Train")
    plot_decision_boundary(model, x_train, y_train)
    plt.subplot(1,2,2)
    plt.title("Test")
    plot_decision_boundary(model, x_test, y_test)
    plt.show()

class CircleModelv1(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)

    def forward(self, x):
        return self.layer_3(self.layer_2(self.layer_1(x)))
        
# model_1 = CircleModelv1().to(device)

class CircleModelv2(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(2,10)
        self.layer_2 = nn.Linear(10,10)
        self.layer_3 = nn.Linear(10,1)

        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))


torch.manual_seed(42)
model_2 = CircleModelv2().to(device)

loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(params=model_2.parameters(), lr=0.1)

def accuracy_fn(y_true, y_preds):
    correct = torch.eq(y_true, y_preds).sum().item()
    acc = (correct/len(y_preds)) * 100
    return acc

epochs = 2000

x_train, y_train = x_train.to(device), y_train.to(device)
x_test, y_test = x_test.to(device), y_test.to(device)

for epoch in range(epochs):
    model_2.train()

    y_logits = model_2(x_train).squeeze()
    y_preds = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_train, y_preds)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    model_2.eval()
    with torch.inference_mode():
        test_logits = model_2(x_test).squeeze()
        test_preds = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_test, test_preds)

        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")
            
# model_plot(model_2,x_train,y_train,x_test,y_test)

model_2.eval()
with torch.inference_mode():
    new_preds = torch.round(torch.sigmoid(model_2(x_test))).squeeze()
print(new_preds[:10], y_test[:10])