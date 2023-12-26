import torch, torchvision
import torch.nn as nn
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from io import BytesIO

filename = 'question.jpg'

img = Image.open(filename)

with open(filename, 'rb') as f:
    binary = f.read()
img = Image.open(BytesIO(binary))

img_array = np.asarray(img)

print(img_array)

# for the MNIST data

transform = transforms.Compose([
    transforms.ToTensor()
    ])

# download MNIST
train = torchvision.datasets.MNIST(
        root='.',
        train=True,
        download=True,
        transform=transform)

img = np.transpose(train[0][0], (1,2,0))
img = img.reshape(img.shape[0], img.shape[1])
print("result:", train[0][1])
plt.imshow(img, cmap='gray')

# convolution

conv = nn.Conv2d(
        in_channels=1,
        out_channels=4,
        kernel_size=3,
        stride=1,
        padding=1)

# the reader to make int data from the image on the jpg or anything

# image reader

# int-izer

# return

