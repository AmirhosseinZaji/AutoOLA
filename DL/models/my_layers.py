import torch
from torch import nn
import sys
sys.path.append('..')
#########################################################################################
def conv_block(in_channels, out_channels, is_batchnorm):
    if is_batchnorm:
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    else:
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
    return conv
#########################################################################################
class ConvCat(nn.Module):
    def __init__(self, in_channels, out_channels, is_batchnorm, is_deconv, n_concat=2):
        super(ConvCat, self).__init__()
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.UpsamplingBilinear2d(scale_factor=2))

        self.conv = conv_block(in_channels+(n_concat-2)*out_channels, out_channels, is_batchnorm)

    def forward(self, to_conv, *to_cat):
        output = self.up(to_conv)
        for i in range(len(to_cat)):
            output = torch.cat([output, to_cat[i]], dim=1)
        output = self.conv(output)
        return output
#########################################################################################
def ppp_block(in_channels, out_channels, kernel_size, padding):

    conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

    return conv