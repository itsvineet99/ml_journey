import torch
import numpy as np

'''this file includes me trying out some code examples i learned while following
   learn pytorch in a day series (video). all of them are commented out as i use 
   same variable names in multiple places. '''

# print(torch.__version__)
# print(torch.cuda.is_available())
# print(torch.backends.mps.is_available())
# print(torch.backends.mps.is_built()) 

# scalor = torch.tensor(53)
# print(scalor)
# print(scalor.ndim)
# print(scalor.item())

# vector = torch.tensor([1,43,21,5])
# print(vector)
# print(vector.ndim)
# print(vector.shape)

# matrix = torch.tensor([[6,54,34,23],[2,3,4,5]])
# print(matrix)
# print(matrix.ndim)
# print(matrix.shape)
# print(matrix[1])
# print(matrix[0,2])

# tensor = torch.tensor([[[6,23],[2,3,],[3,6]],
                    #    [[4,54],[33,90],[20,40]]])
# print(tensor)
# print(tensor.ndim)
# print(tensor.shape)
# print(tensor[0])
# print(tensor.is_cuda)
# random_tensor = torch.rand(4,5)
# print(random_tensor)
# print(random_tensor.shape)
# print(random_tensor.ndim)
# print(random_tensor[1,156,190])

# zeros_tensor = torch.zeros(size=(4,5))
# print(zeros_tensor[1])
# random_tensor[0] = zeros_tensor[0]*random_tensor[1]
# print(random_tensor)

# print(random_tensor[1])

# ones_tensor = torch.ones(2,3)
# print(ones_tensor)
# print(ones_tensor.dtype)

# range = torch.arange(1,1000,99)
# print(range)

# ones = torch.ones_like(range)
# print(ones)

# float_32_to_int_tensor = torch.tensor([2.33,4.55,7], dtype=int, device=None, requires_grad=False)
# print(float_32_to_int_tensor)
# print(float_32_to_int_tensor.device)


# tensor_example = torch.tensor([1,3,5])
# new_tensor = tensor_example
# print(new_tensor)
# new_tensor += 29
# print(f"new tensoor : {new_tensor}")
# print(f"original tensor ; {tensor_example}")
# print(tensor_example + new_tensor)
# element wise multiplication also known as dot product
# print(new_tensor * tensor_example)
# matrix multiplication
# print(torch.matmul(new_tensor,tensor_example))

# some_num = torch.rand(size=(4,3))
# some_num2 = torch.rand(size=(5,3))
# print(some_num)
# some_num2 = some_num2.T
# print(torch.matmul(some_num, some_num2))
# print(some_num.T.shape)

# print(some_num.min(), torch.min(some_num2))
# print(some_num.max(), torch.max(some_num2))
# print(some_num.mean(), torch.mean(some_num2))
# print(some_num.sum(), torch.sum(some_num))

# print(some_num.argmin())
# print(some_num.argmax())

# x = torch.arange(1, 10)
# print(x, x.shape)
# x_reshaped = x.reshape(3,3)
# print(x_reshaped, x_reshaped.shape)
# print(x, x.shape)
# reshape will only work when number of elements do not change from original tensor

# x = torch.arange(1, 10)
# print(x, x.shape)
# x_reshaped = x.view(3,3)
# print(x_reshaped, x_reshaped.shape)
# print(x, x.shape)
# x_reshaped[0] = torch.tensor([94,88,77])
# print(x_reshaped, x_reshaped.shape)
# print(x, x.shape)
# view is same as reshape but it changes also changes original tensor as they both share same memory 

# y = torch.rand(size=(1,5,5))
# z = torch.rand(size=(1,5,5))
# ten_stacked = torch.stack([y, z], dim=2)
# print(ten_stacked, ten_stacked.shape, "dimension: ", ten_stacked.dim())

# y = torch.rand(size=(3,224,224))
# print(y.shape)
# print(y.squeeze())
# print(y.squeeze().shape) #removes all single dimensions
# y_unsqueezed = y.unsqueeze(dim=0)
# print(y_unsqueezed)
# print(y_unsqueezed.shape)

# y_rearanged = y.permute(2,0,1)
# print(y[0,0,0])
# print(y_rearanged.shape)
# y_rearanged[0,0,0] = 565656
# print(y_rearanged[0,0,0])
# print(y[0,0,0])
# permutation is basically changing the order of dimensions 
# and they create view of original tensor so change in permutated
# tensor also changes original tensor

# indexing_int = torch.arange(1,10).reshape(1,3,3)
# print(indexing_int)
# print(indexing_int.shape)
# print(indexing_int[:,1,0])
# first index gives first dim and second dim gives second dim and so on 
# if elements in dim are n then we can access only n-1 coz of index starting with 0

# array = np.arange(1 ,10)
# tensor = torch.from_numpy(array).type(torch.float64)
# print(array, array.dtype)
# print(tensor)
# tensor is copy of array so changes in array after creating tensor won't affect tensor
# array += 1
# print(array, array.dtype)
# print(tensor)

# numpy_tensor = tensor.numpy()
# print(numpy_tensor, numpy_tensor.dtype)


# RANDOM_SEED = 124
# torch.manual_seed(RANDOM_SEED)

# random_tensor1 = torch.rand(size=(2,4))
# torch.manual_seed(RANDOM_SEED)
# random_tensor2 = torch.rand(size=(2,4))
# print(random_tensor1 == random_tensor2)


# device = "mps" if torch.backends.mps.is_available() else "cpu"
# print(device)
# print(torch.mps.device_count())

# tensor = torch.tensor([1,2,3])
# print(tensor, tensor.device)

# tensor_on_mps = tensor.to(device)
# print(tensor_on_mps, tensor_on_mps.device)
# back_to_cpu_tensor = tensor_on_mps.cpu()
# print(back_to_cpu_tensor, back_to_cpu_tensor.device)
