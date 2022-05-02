#!/usr/bin/env python
# coding: utf-8

#%% Implementing ResNet in PyTorch


import torch
import torch.nn as nn

from functools import partial
from collections import OrderedDict

#%% Basic Block
# Auto padded convolutional layer


class Conv1dAuto(nn.Conv1d): # dynamic add padding based on the kernel_size
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  [(self.kernel_size[0]//2)]

conv3x3 = partial(Conv1dAuto, kernel_size=3, bias=False)
        
# class Conv2dAuto(nn.Conv2d):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size
        
# conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)      

conv = conv3x3(in_channels=32, out_channels=64)
# print(conv)
del conv


#%% Residual Block
# The residual block takes an input with `in_channels`, applies some blocks of
# convolutional layers to reduce it to `out_channels` and sum it up to the 
# original input. If their sizes mismatch, then the input goes into an `identity`. 


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels =  in_channels, out_channels
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()   
    
    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: 
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels
    
#%% Random test
ResidualBlock(32, 64)
# Let's test it with a dummy vector with one one, we should get a vector with two
dummy = torch.ones((1, 1, 1))
block = ResidualBlock(1, 64)
block(dummy)

#%% ResNetResidualBlock
# In ResNet each block has a expansion parameter in order to increase the `out_channels`.
# Also, the identity is defined as a Convolution followed by an Activation layer, 
# this is referred as `shortcut`. 
# Then, we can just extend `ResidualBlock` and defined the `shortcut` function.

class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(OrderedDict(
        {
            'conv' : nn.Conv1d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            'bn' : nn.BatchNorm1d(self.expanded_channels)
            
        })) if self.should_apply_shortcut else None
        
        
    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels

ResNetResidualBlock(32, 64)


#%% Basic Block
# A basic ResNet block is composed by two layers of `3x3` convs/batchnorm/relu.
 
# function to stack one conv and batchnorm layer. 
# Using `OrderedDict` to properly name each sublayer.

def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs), 
                          'bn': nn.BatchNorm1d(out_channels) }))

conv_bn(3, 3, nn.Conv1d, kernel_size=3)

class ResNetBasicBlock(ResNetResidualBlock):
    expansion = 1
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation(),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )
#%% Test with  dummy data
dummy = torch.ones((1, 32, 224))

block = ResNetBasicBlock(32, 64)
block(dummy).shape
# print(block)


#%% BottleNeck
# To increase the network deepths but to decrese the number of parameters, 
# the Authors defined a BottleNeck block that 
# "The three layers are 1x1, 3x3, and 1x1 convolutions, where the 1×1 layers
# are responsible for reducing and then increasing (restoring) dimensions,
# leaving the 3×3 layer a bottleneck with smaller input/output dimensions.
# " We can extend the `ResNetResidualBlock` and create these blocks.

class ResNetBottleNeckBlock(ResNetResidualBlock):
    expansion = 4
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
           conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
             activation(),
             conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
             activation(),
             conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
        )
    
#%% test with dummy data
dummy = torch.ones((1, 32, 10))

block = ResNetBottleNeckBlock(32, 64)
block(dummy).shape
# print(block)


#%% Layer
# A ResNet's layer is composed by blocks stacked one after the other. 
# We can easily defined it by just stuck `n` blocks one after the other,
# just remember that the first convolution block has a stride of two since
# "We perform downsampling directly by convolutional layers that have a stride of 2".

class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1
        
        self.blocks = nn.Sequential(
            block(in_channels , out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion, 
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x

#%% test with dummy data
dummy = torch.ones((1, 32, 48))

layer = ResNetLayer(64, 128, block=ResNetBasicBlock, n=3)
# layer(dummy).shape
layer


#%% Encoder
# Similarly, the encoder is composed by multiple layer at increasing features size.

class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by increasing different layers with increasing features.
    """
    def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], deepths=[2,2,2,2], 
                 activation=nn.ReLU, block=ResNetBasicBlock, *args,**kwargs):
        super().__init__()
        
        self.blocks_sizes = blocks_sizes
        
        self.gate = nn.Sequential(
            nn.Conv1d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(self.blocks_sizes[0]),
            activation(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([ 
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation, 
                        block=block,  *args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion, 
                          out_channels, n=n, activation=activation, 
                          block=block, *args, **kwargs) 
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]       
        ])
        
        
    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x


#%% Decoder
# The decoder is the last piece we need to create the full network. 
# It is a fully connected layer that maps the features learned by the network
# to their respective classes. Easily, we can defined it as:

class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling 
    and maps the output to the correct class by using a fully connected layer.
    """
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool1d((1))
        self.decoder = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x


#%% ResNet
# Final, we can put all the pieces together and create the final model.

class ResNet(nn.Module):
    
    def __init__(self, in_channels, n_classes, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# We can now defined the five models proposed by the Authors, `resnet18,34,50,101,152`

def resnet18(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBasicBlock, deepths=[2, 2, 2, 2])

def resnet34(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBasicBlock, deepths=[3, 4, 6, 3])

def resnet50(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBottleNeckBlock, deepths=[3, 4, 6, 3])

def resnet101(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBottleNeckBlock, deepths=[3, 4, 23, 3])

def resnet152(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBottleNeckBlock, deepths=[3, 8, 36, 3])


#%% Summary to compare with original


# from torchinfo import summary

# model = resnet101(1, 1000)
# summary(model, input_size=(64,1, 224))

