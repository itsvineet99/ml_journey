import torch
from torch import nn
import torch.optim.optimizer
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from helper_functions import accuracy_fn
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

# print(len(train_data), len(test_data))

# image, lable = train_data[0]
# print(image, lable) 

class_names = train_data.classes
class_to_idx = train_data.class_to_idx
# print(class_names)
# print(class_to_idx)
# print(train_data.targets)
# print(image.shape)

# visualizing data

# fig = plt.figure(figsize=(9, 9))
# rows, cols = 4, 4
# for i in range(1, rows*cols+1):
#     random_idx = torch.randint(0, len(train_data), size=[1]).item()
#     img, lable = train_data[random_idx]
#     fig.add_subplot(rows, cols, i)
#     plt.imshow(img.squeeze(), cmap="gray")
#     plt.title(class_names[lable])
#     plt.axis(False)
# plt.show()

# loading data or using dataloader
BATCH_SIZE = 32

train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)
print(len(train_dataloader))

train_feature_batch, train_lable_batch = next(iter(train_dataloader))

# torch.manual_seed(42)

random_idx = torch.randint(0, len(train_feature_batch), size=[1]).item()

img, lable = train_feature_batch[random_idx], train_lable_batch[random_idx]

# plt.imshow(img.squeeze(), cmap="gray")
# plt.title(class_names[lable])
# plt.axis(False)
# plt.show()
# print(f"image size: {img.shape}")
# print(f"lable: {lable}, lable size: {lable.shape}")

flatten_model = nn.Flatten() # model to reduce dimensions

class FashionMNISTModelV0(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int,
                 ):
        super().__init__()

        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_shape, hidden_units),
            nn.Linear(hidden_units, output_shape)
        )

    def forward(self, x):
        return self.layer_stack(x)

torch.manual_seed(42)
model_v0 = FashionMNISTModelV0(
    input_shape=784, #28*28
    hidden_units=10,
    output_shape=len(class_names)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_v0.parameters(), lr=0.1)

def print_train_time(start:float,
                     end:float,
                     device: torch.device =None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

train_time_start_on_cpu = timer()

epochs = 3

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-----")

    train_loss = 0

    for batch, (x,y) in enumerate(train_dataloader):
        model_v0.train()

        y_preds = model_v0(x)

        loss = loss_fn(y_preds, y)

        train_loss += loss

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if batch % 400 == 0:
            print(f"Looked at {batch * len(x)}/{len(train_dataloader.dataset)} samples.")
    
    train_loss /= len(train_dataloader)

    test_loss, test_acc = 0,0
    model_v0.eval() 
    with torch.inference_mode():
        for x_test, y_test in test_dataloader:
            test_pred = model_v0(x_test)
        
            test_loss += loss_fn(test_pred, y_test)

            test_acc += accuracy_fn(y_test, test_pred.argmax(dim=1))
        
        test_loss /= len(test_dataloader)

        test_acc /= len(test_dataloader)
    
    print(f"train loss: {train_loss:.4f} | test loss: {test_loss:.4f} | test acc: {test_acc:.4f}")

train_time_end_on_cpu = timer()

total_train_time_model_v0 = print_train_time(start=train_time_start_on_cpu,
                                             end=train_time_end_on_cpu,
                                             device=str(next(model_v0.parameters()).device))


device = "mps" if torch.backends.mps.is_available() else "cuda"
print(f"device: {device}")

def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               Accuracy_fn,
               device: torch.device = device):  

                test_loss, test_acc = 0,0
                model.eval() 
                with torch.inference_mode():
                    for x, y in data_loader:
                        x,y = x.to(device), y.to(device)
                        test_pred = model(x)
                    
                        test_loss += loss_fn(test_pred, y)

                        test_acc += Accuracy_fn(y, test_pred.argmax(dim=1))
                    
                    test_loss /= len(data_loader)

                    test_acc /= len(data_loader)
                
                return {"model_name": model.__class__.__name__,
                        "model_loss": test_loss.item(),
                        "model_acc" : test_acc}

model_v0_results = eval_model(model=model_v0,
                              data_loader=test_dataloader,
                              loss_fn=loss_fn,
                              Acccuracy_fn=accuracy_fn)

print(model_v0_results)



class FashionMNISTModelv1(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()

        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_shape, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_shape),
            nn.ReLU()
        )
    def forward(self, x):
        return self.layer_stack(x)

torch.manual_seed(42)
model_v1 = FashionMNISTModelv1(input_shape=784,
                               hidden_units=10,
                               output_shape=len(class_names)).to(device)

optimizer = torch.optim.SGD(params=model_v1.parameters(), lr=0.1)


def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
                    train_acc, train_loss = 0, 0
                    for batch, (x,y) in enumerate(data_loader):
                        x,y = x.to(device), y.to(device)
                        model.train()

                        y_preds = model(x)

                        loss = loss_fn(y_preds, y)
                        train_loss += loss

                        train_acc += accuracy_fn(y, 
                                                y_preds.argmax(dim=1))
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    
                    train_loss /= len(data_loader)
                    train_acc /= len(data_loader)

                    print(f"train loss: {train_loss:.5f}| train acc: {train_acc:.2f}%")

def test_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device: torch.device = device):

                test_loss, test_acc = 0,0
                model.eval() 
                with torch.inference_mode():
                    for x_test, y_test in data_loader:
                        x_test,y_test = x_test.to(device), y_test.to(device)

                        test_pred = model(x_test)
                    
                        test_loss += loss_fn(test_pred, y_test)
                        test_acc += accuracy_fn(y_test, test_pred.argmax(dim=1))
                    
                    test_loss /= len(data_loader)

                    test_acc /= len(data_loader)

                    print(f"test loss: {test_loss:.2f} | test acc: {test_acc:.2f}%")
                
                return {"model_name": model.__class__.__name__,
                        "model_loss": test_loss.item(),
                        "model_acc" : test_acc}


train_time_start_on_cpu = timer()

epochs = 3

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-----")
    
    train_step(model=model_v1,
                data_loader=train_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                accuracy_fn=accuracy_fn,
                device= device)

    test_step(model=model_v1,
                data_loader=test_dataloader,
                loss_fn=loss_fn,
                accuracy_fn=accuracy_fn,
                device= device)
train_time_end_on_cpu = timer()

total_train_time_model_v1 = print_train_time(start=train_time_start_on_cpu,
                                             end=train_time_end_on_cpu,
                                             device=str(next(model_v1.parameters()).device))

model_v1_results = eval_model(model=model_v1,
                                data_loader=test_dataloader,
                                loss_fn=loss_fn,
                                Accuracy_fn=accuracy_fn,
                                device=device)
print(model_v1_results)