from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import time






train_data = datasets.MNIST(
    root='data',
    train=True,
    transform=ToTensor(),
    download=True,
)
# test_data = datasets.MNIST(
#     root='data',
#     train=False,
#     transform=ToTensor()
# )

import matplotlib.pyplot as plt
import torch
plt.imshow(train_data.data[0], cmap='gray')
plt.title('%i' % train_data.targets[0])
plt.show()

def distance_matrix(x, y):
    return torch.cdist(x, y, 2).sum()

def distance_matrix_2(x, y):
    #return torch.sqrt(torch.square(x-y).sum())
    return ((x - y)**2).sum()**0.5


train=train_data.data.float()
#relation = np.array([ [distance_matrix_2(x, y).item() for x in train_data.data ] for y in train_data.data])
t0 = time.time()

relation = np.array([ [distance_matrix_2(x, y).item() for x in train ] for y in train])
#relation = np.array([ [ ((x - y)**2).sum()**0.5  for x in train ] for y in train])


t1 = time.time()
print(t1-t0)


np.savetxt("mnist/relation.csv", relation, delimiter=",")
np.savetxt("mnist/target.csv", train_data.targets, delimiter=",")