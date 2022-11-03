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

For cat breed classification task, a modified residual network with 18 layers (ResNet-18) 
is trained without any pre-trained weights. Except ResNet, some other common models are 
also considered, such as LeNet, and AlexNet, however they can not fit the dataset or be 
well-trained. ResNet has great performance and is easier to train due to its residual 
structure. ResNet-18 is the simplest common residual model, it is enough to deal with 
cat breed classification. The number of convolutional kernels in the last residual block 
is reduced from 512 to 256, which provides lighter model and is more adaptable to given 
dataset.

For training modified ResNet-18, train_val_split is set as 0.8 which means 6,400 images 
are used to train model and 1,600 images for validation. Moreover, the batch_size and 
epochs are 100 and 200, respectively. The kaiming_normal is used as weight initialization 
method, and ladder learning ratio scheduler is designed for better training. The optimizer 
and loss function are Adam and cross entropy loss, it is very common yet efficient. 
Furthermore, some different data transform strategies are used to reduce overfitting, such as 
random rotation, random horizontal flip, random vertical flip, random erasing and color jitter.

Explanations of Metaparameters and training options:
Train_val_split: 
Setting train_val_split as 0.8 is a suitable choice. Because the ResNet-18 is used as 
classification model, training this model needs enough data. Given dataset consists of 
8,000 images in total and each class has 1,000 images, the amount of data is not enough 
for training ResNet-18. Therefore, it is necessary to split more data to train set. 
Simultaneously, enough validation data helps to judge the performance of trained model. 
Therefore, 0.8 is a suitable choice in my opinion. 

Batch_size:
The batch size is 100 in the stage of model training, which is adjusted according to 
performance of validation data when model training. 

Epochs:
The epochs is setting as 200. In process of model training, training 200 epochs could 
provide a nice trained model.

Weight initialization:
The kaiming_uniform is used for initializing weights of my model, because it is friendlier 
to activation function ReLU. 

Learning ratio:
The initial learning ratio is 1e-3, and a ladder learning ratio scheduler is designed as
lr=1/(epoch//5+1)
The decay of the learning rate with the epoch allows the model to be better trained.

Optimizer:
The optimizer of ResNet-18 is Adam which is a common and effective optimizer. Moreover, 
the weight decay of Adam helps to achieve regularization of the model, which helps 
overcome model overfitting. 

Loss function:
The loss function is one of the most commonly used multi-class model loss function, 
cross entropy loss function. It is useful to train modified ResNet-18.

Data transform:
Data transform is necessary especially for deep neural network, it helps to deal with 
model overfitting issue. For train data, in addition to regular transform operation 
ToTensor which converts original images to a tensor with the value [0,1], some others 
operations are also used. For example, ColorJitter is used to adjust image's brightness, 
contrast and saturation; RandomRotation, RandomHorizontalFlip and RandomVerticalFlip 
are used to randomly rotate image; and RandomErasing randomly erases a regoin in original 
image. For test data, just ToTensor and Resize operations are applied. More data transform 
operations are used for train data improves model generalization ability.


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
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        return out


net = Network(ResBlock, [2, 2, 2, 2])

############################################################################
######      Specify the optimizer and loss function                   ######
############################################################################
lr = 1e-3
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.01)
loss_func = nn.CrossEntropyLoss()


############################################################################
######  Custom weight initialization and lr scheduling are optional   ######
############################################################################

# Normally, the default weight initialization and fixed learing rate
# should work fine. But, we have made it possible for you to define
# your own custom weight initialization and lr scheduler, if you wish.
def weights_init(m):
    if isinstance(m, nn.Linear): # linear layer weight initialization
        nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d): # convolutional layer weigt initialization
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d): # batch norm weight initialization
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


# learning ratio scheduler
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch//5+1))


############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
train_val_split = 0.8
batch_size = 100
epochs = 200

