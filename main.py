from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy

# choosing which device to run GPU/CPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loading images

## desired size of the output images
imsize = 512 if torch.cuda.is_available() else 128

loader = transforms.Compose([
    transforms.Resize(imsize), # scale imported images
    transforms.ToTensor() # transform it into torch tensor
])

def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension needed to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

style_img = image_loader("./data/picasso.jpg")
content_img = image_loader("./data/dancing.jpg")

assert style_img.size() == content_img.size(),\
    "we need to import style and content img of the same size"

## Display the input data

unloader = transforms.ToPILImage() # reconvert into PIL image

plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone() # we clone the tensor to not do changes on it
    image = image.squeeze(0)     # removing the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated

plt.figure()
imshow(style_img, title="Style Image")

plt.figure()
imshow(content_img, title="Content Image")
