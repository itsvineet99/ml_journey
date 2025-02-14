import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from torchmetrics import Accuracy
from helper_functions import plot_predictions, plot_decision_boundary

NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

x_blob, y_blob = make_blobs(n_samples=1000, n_features= NUM_FEATURES,
                            centers=NUM_CLASSES, cluster_std=1.5,
                            random_state=RANDOM_SEED)

x_blob = torch.from_numpy(x_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

x_blob_train, x_blob_test, y_blob_train, y_blob_test = train_test_split(x_blob,
                                                                        y_blob,
                                                                        test_size=0.2,
                                                                        random_state=RANDOM_SEED)

# plt.figure(figsize=(10,7))
# plt.scatter(x_blob[:,0], x_blob[:,1], c=y_blob, cmap=plt.cm.RdYlBu)
# plt.show()

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"using device: {device}")

class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_features=8):
        super().__init__()

        self.linear_layer_stack = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, output_features)
        )

    def forward(self, x):
        return self.linear_layer_stack(x)

# print(x_blob_train.shape, y_blob_train.shape, torch.unique(y_blob_train))

model = BlobModel(input_features=2,
                  output_features=4, 
                  hidden_features=8).to(device)


loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

def accuracy_fn(y_true, y_preds):
    correct = torch.eq(y_true, y_preds).sum().item()
    acc = (correct/len(y_preds)) * 100
    return acc


torch.manual_seed(42)

x_blob_train, x_blob_test = x_blob_train.to(device), x_blob_test.to(device)
y_blob_train, y_blob_test = y_blob_train.to(device), y_blob_test.to(device)

epochs = 100
for epoch in range(epochs):
    model.train()

    y_logits = model(x_blob_train)
    y_preds = torch.softmax(y_logits, dim=1).argmax(dim=1)

    loss = loss_fn(y_logits, y_blob_train)
    acc = accuracy_fn(y_blob_train, y_preds)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_logits = model(x_blob_test)
        test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)

        test_loss = loss_fn(test_logits, y_blob_test)
        test_acc = accuracy_fn(y_blob_test, test_preds)

        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")
            
print(y_blob_test, test_preds)
torch_metrics_acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(device)
print(torch_metrics_acc(test_preds, y_blob_test))

# plt.figure(figsize=(12, 6))
# plt.subplot(1,2,1)
# plt.title("Train")
# plot_decision_boundary(model, x_blob_train, y_blob_train)
# plt.subplot(1,2,2)
# plt.title("Test")
# plot_decision_boundary(model, x_blob_test, y_blob_test)
# plt.show()