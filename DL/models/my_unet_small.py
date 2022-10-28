from torch import nn
import sys
sys.path.append('..')
import properties
from my_layers import *

class UNet(nn.Module):
    def __init__(self, is_deconv=False, is_batchnorm=True, filters=properties.filters_child):
        super(UNet, self).__init__()
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.filters = filters

        ## -------------max pool--------------
        self.pool = nn.MaxPool2d(kernel_size=2)

        ## -------------Encoder--------------
        self.conv1 = conv_block(self.filters[0], self.filters[1], self.is_batchnorm)
        self.conv2 = conv_block(self.filters[1], self.filters[2], self.is_batchnorm)
        self.conv3 = conv_block(self.filters[2], self.filters[3], self.is_batchnorm)

        ## -------------Decoder--------------
        self.concat2 = ConvCat(self.filters[3], self.filters[2], self.is_batchnorm, self.is_deconv)
        self.concat1 = ConvCat(self.filters[2], self.filters[1], self.is_batchnorm, self.is_deconv)

        ## -------------Predictor--------------
        self.density_pred = nn.Conv2d(in_channels=self.filters[1], out_channels=1, kernel_size=1, bias=False)


    def forward(self, inputs):
        ## -------------Encoder--------------
        conv1 = self.conv1(inputs)
        pool1 = self.pool(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool(conv2)
        conv3 = self.conv3(pool2)
        ## -------------Decoder--------------
        concat2 = self.concat2(conv3, conv2)
        concat1 = self.concat1(concat2, conv1)
        ## -------------Predictor--------------
        output = self.density_pred(concat1)

        return output