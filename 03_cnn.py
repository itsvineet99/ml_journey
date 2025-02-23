import torch
from torch import nn
import torch.optim.optimizer
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from helper_functions import (accuracy_fn, 
                              train_step, 
                              test_step, 
                              print_train_time, 
                              eval_model)
from timeit import default_timer as timer
from tqdm.auto import tqdm


# downloading dataset separatiely for training and testing 
train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

class_names = train_data.classes
class_to_idx = train_data.class_to_idx



BATCH_SIZE = 32

train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

# defining model

device = "mps" if torch.backends.mps.is_available() else "cpu"

class FashionMNISTModelv2(nn.Module):
    def __init__(self, input_shape: int, output_shape: int, hidden_units: int):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=2,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=2,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*8*8,
                      out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        return x
    
model_v2 = FashionMNISTModelv2(input_shape=1,
                               hidden_units=10,
                               output_shape=len(class_names)).to(device)

# print(model_v2)
# print(model_v2.state_dict())

# torch.manual_seed(42)
# images = torch.randn(size=(32, 3, 28, 28)).to(device)
# test_image = images[0]

# print(test_image.shape)

# pred = model_v2(test_image.unsqueeze(0))
# print(pred.shape)
# print(pred)

# conv_layer = nn.Conv2d(in_channels=3,
#                        out_channels=10,
#                        kernel_size=3, 
#                        stride=1,
#                        padding=0)

# conv_output = conv_layer(test_image.unsqueeze(0))

# print(conv_output.shape)

# setting up loss function and optimizer

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_v2.parameters(), lr=0.1)

torch.manual_seed(42)

train_time_start = timer()

epochs = 3

for epoch in range(epochs):
    print(f"Epoch: {epoch}\n -----")
    train_step(model=model_v2,
               data_loader=train_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn,
               device=device)
    test_step(model=model_v2,
              data_loader=test_dataloader,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn,
              device=device)
    
train_time_end = timer()

total_train_time = print_train_time(start=train_time_start, end=train_time_end, device=device)

model_v2_results = eval_model(model=model_v2,
                            data_loader=test_dataloader,
                            loss_fn=loss_fn,
                            Accuracy_fn=accuracy_fn,
                            device=device)


def make_predictions(model:torch.nn.Module,
                     data: list,
                     device :torch.device = device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0).to(device)

            pred_logit = model(sample)

            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)

            pred_probs.append(pred_prob.cpu())

    return torch.stack(pred_probs)

import random
random.seed(42)
test_samples = []
test_lables = []

for sample, lable in random.sample(list(test_data), 9):
    test_samples.append(sample)
    test_lables.append(lable)

print(test_samples[0].shape)

# plt.imshow(test_sample[0].squeeze(), cmap="gray")
# plt.show()

pred_probs = make_predictions(model=model_v2,
                              data=test_samples,
                              device=device)

pred_classes = pred_probs.argmax(dim=1)
print(pred_classes)
print(test_lables)


# # Plot predictions
# plt.figure(figsize=(9, 9))
# nrows = 3
# ncols = 3
# for i, sample in enumerate(test_samples):
#   # Create a subplot
#   plt.subplot(nrows, ncols, i+1)

#   # Plot the target image
#   plt.imshow(sample.squeeze(), cmap="gray")

#   # Find the prediction label (in text form, e.g. "Sandal")
#   pred_label = class_names[pred_classes[i]]

#   # Get the truth label (in text form, e.g. "T-shirt")
#   truth_label = class_names[test_lables[i]] 

#   # Create the title text of the plot
#   title_text = f"Pred: {pred_label} | Truth: {truth_label}"
  
#   # Check for equality and change title colour accordingly
#   if pred_label == truth_label:
#       plt.title(title_text, fontsize=10, c="g") # green text if correct
#   else:
#       plt.title(title_text, fontsize=10, c="r") # red text if wrong
#   plt.axis(False);

# plt.show()

# Import tqdm for progress bar
from tqdm.auto import tqdm

# 1. Make predictions with trained model
y_preds = []
model_v2.eval()
with torch.inference_mode():
  for X, y in tqdm(test_dataloader, desc="Making predictions"):
    # Send data and targets to target device
    X, y = X.to(device), y.to(device)
    # Do the forward pass
    y_logit = model_v2(X)
    # Turn predictions from logits -> prediction probabilities -> predictions labels
    y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1) # note: perform softmax on the "logits" dimension, not "batch" dimension (in this case we have a batch size of 32, so can perform on dim=1)
    # Put predictions on CPU for evaluation
    y_preds.append(y_pred.cpu())
# Concatenate list of predictions into a tensor
y_pred_tensor = torch.cat(y_preds)
print(y_pred_tensor)

import mlxtend
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
confmat_tensor = confmat(preds=y_pred_tensor,
                         target=test_data.targets)

fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(), 
    class_names=class_names, 
    figsize=(10, 7)
);
plt.show()


# saving and loading model


MODEL_PATH = "models/fashion_mnist_model_v2.pth"
print(f"Saving model to: {MODEL_PATH}")
torch.save(model_v2.state_dict(), f=MODEL_PATH)

torch.manual_seed(42)

loaded_model = FashionMNISTModelv2(input_shape=1,   
                                   hidden_units=10,
                                   output_shape=len(class_names))

loaded_model.load_state_dict(torch.load(MODEL_PATH))

loaded_model.to(device)

loaded_model_results = eval_model(model=loaded_model,
                                  data_loader=test_dataloader,
                                  loss_fn=loss_fn,
                                  Accuracy_fn=accuracy_fn)


print(model_v2_results)
print(loaded_model_results)