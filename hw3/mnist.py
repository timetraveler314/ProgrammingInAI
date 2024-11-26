import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms.v2 as v2

from Genshin import Tensor # My impl. of Tensor

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5], std=[0.5])
])

mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_images = np.array([img for img, label in mnist_train])
train_labels = np.array([label for img, label in mnist_train])

# Load all training images into my Tensor class
train_images_t = Tensor.from_numpy(train_images)
print(train_images_t.shape()) # Expected: [60000, 28, 28]

# Example: Load the first image
first_image = Tensor.from_numpy(train_images[0])
print(first_image) # Expected [28, 28] image with [-1.0, 1.0] entries
