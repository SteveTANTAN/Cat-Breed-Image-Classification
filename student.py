#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables train_val_split,
batch_size as well as the transform function.
You are encouraged to modify these to improve the performance of your model.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchsummary as ts
"""
   Answer to Question:

Briefly describe how your program works, and explain any design and training
decisions you made along the way.

Ans:
In program works, ResNet-18 is selected as network model to deal with cat breads classification, 
because ResNet has strong model capacity and is more easy to trained with shortcut structure.
The optimizer and loss function are Adam optimizer and CrossEntropyLoss, respectively.
The greatest difficulty of training used ResNet model is overfitting. Therefore, in order to handle 
overfitting problem, the following strategies are used.
    1. Data augementation ---- In addition to regular transform operation ToTensor which converts 
    original images to a tensor with the value [0,1], some others operations are also used. For example, 
    ColorJitter is used to adjust image's brightness, contrast and saturation; RandomRotation, 
    RandomHorizontalFlip and RandomVerticalFlip are used to rotate image; and RandomErasing randomly 
    erases a regoin in original image. 
    2. Regularization ---- Due to Adam optimizer is used as our model optimizer, the weight_decay can 
    achieve the effect of regularization, therefore the weight_decaly of Adam is setted as 0.01 in model 
    training.

"""

############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################
def transform(mode):
    """
    Called when loading the data. Visit this URL for more information:
    https://pytorch.org/vision/stable/transforms.html
    You may specify different transforms for training and testing
    """
    if mode == 'train':
        transSet = [transforms.RandomRotation(degrees=30), # randomly rotating image
               transforms.RandomHorizontalFlip(), 
               transforms.RandomVerticalFlip(),
               transforms.ColorJitter(0.5, 0.5, 0.5), # adjust image's brightness, contrast and saturation 
            #    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
               transforms.RandomErasing(p=0.3) # randomly erasing a regoin
            ]
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((80, 80)),
            transforms.RandomApply(transSet, p=0.5) # randomly appling transSet with probability p
            # transforms.RandomChoice(transSet)
        ])
        return trans
    elif mode == 'test':
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((80, 80)),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return trans


############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################

# Define basic residual block #
# In each block, there are two convolutional layers and a shortcut,
# and the size of all convolotional kernel is 3X3
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Define residual network #
# In residual network, there are one normal convolutional layer and
# four residual block groups which group consists of two same ResBlock.
class Network(nn.Module):
    def __init__(self, block, num_blocks, num_classes=8):
        super(Network, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        self.linear = nn.Linear(1024, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


net = Network(ResBlock, [2, 2, 2, 2])

############################################################################
######      Specify the optimizer and loss function                   ######
############################################################################
lr = 1e-3
optimizer = optim.Adam(net.parameters(), lr=0.001)

loss_func = nn.CrossEntropyLoss()


############################################################################
######  Custom weight initialization and lr scheduling are optional   ######
############################################################################

# Normally, the default weight initialization and fixed learing rate
# should work fine. But, we have made it possible for you to define
# your own custom weight initialization and lr scheduler, if you wish.
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)

# scheduler = None

############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
train_val_split = 0.8
batch_size = 100
epochs = 300

