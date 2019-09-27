import torch
import torchvision
import torchvision.transforms as transforms

train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)


train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=100
)

import numpy as np
import matplotlib.pyplot as plt

# print(len(train_set))
# print(train_set.targets)

sample = next(iter(train_set))
image, label = sample

print(image.shape)

# plt.imshow(image.squeeze(), cmap='gray')
# # plt.show()
print('label: ', label)


batch = next(iter(train_loader))
images, labels = batch
print(batch)
print(images.shape)
print(labels.shape)

grid = torchvision.utils.make_grid(images, nrow=10)

plt.figure(figsize=(15,15))
plt.imshow(np.transpose(grid, (1, 2, 0)))
plt.show()
print('labels: ', labels)