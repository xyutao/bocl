from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn

#Helpers
def _conv3x3(channels, stride, in_channels):
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                     use_bias=False, in_channels=in_channels)