# ResNet for CIFAR
from __future__ import division
import os
from mxnet import autograd as ag
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from gluoncv.model_zoo.cifarresnet import CIFARBasicBlockV1
from mxnet import cpu
from .common import _conv3x3

__all__ = ['get_cifar_resnet', 'cifar_resnet20_v1']

class CIFARBasicBlockV1(HybridBlock):
    def __init__(self, channels, stride, downsample=False, in_channels=0, bn_global=True, **kwargs):
        super(CIFARBasicBlockV1, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        self.body.add(_conv3x3(channels, stride, in_channels))
        self.body.add(nn.BatchNorm(use_global_stats=bn_global))
        self.body.add(nn.Activation('relu'))
        self.body.add(_conv3x3(channels, 1, channels))
        self.body.add(nn.BatchNorm(use_global_stats=bn_global))
        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
                                          use_bias=False, in_channels=in_channels))
            self.downsample.add(nn.BatchNorm(use_global_stats=bn_global))
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        """Hybrid forward"""
        residual = x
        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)

        x = F.Activation(residual+x, act_type='relu')

        return x

class CIFARResNetV1(HybridBlock):
    def __init__(self, block, layers, channels, classes=10, bn_global=True, fix_layers=False, **kwargs):
        super(CIFARResNetV1, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        self.fix_layers = fix_layers
        with self.name_scope():
            self.fix_feats = nn.HybridSequential(prefix='')
            self.fix_feats.add(nn.Conv2D(channels[0], 3, 1, 1, use_bias=False))
            self.fix_feats.add(nn.BatchNorm(use_global_stats=bn_global))

            self.feats = nn.HybridSequential(prefix='')

            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                layer = self._make_layer(block, num_layer, channels[i+1],
                                         stride, i+1, in_channels=channels[i])
                if i == 0:
                    self.fix_feats.add(layer)
                else:
                    self.feats.add(layer)
                self.fix_feats.add(layer)
            self.feats.add(nn.GlobalAvgPool2D())
            self.feats.add(nn.Dense(64, in_units=channels[-1]))

            self.output = nn.Dense(classes, in_units=64)

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, channels != in_channels, in_channels=in_channels,
                            prefix=''))
            for _ in range(layers-1):
                layer.add(block(channels, 1, False, in_channels=channels, prefix=''))
        return layer

    def hybrid_forward(self, F, x):
        if self.fix_layers:
            with ag.pause():
                x = self.fix_feats(x)
        else:
            x = self.fix_feats(x)
        x = self.feats(x)
        out1 = F.L2Normalization(x)
        out2 = self.output(out1)

        return out1, out2


def _get_resnet_spec(num_layers):
    assert (num_layers - 2) % 6 == 0

    n = (num_layers - 2) // 6
    channels = [16, 16, 32, 64]
    layers = [n] * (len(channels) - 1)
    return layers, channels


def get_cifar_resnet(session, num_layers, pretrained=False, ctx=cpu(),
                     root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    layers, channels = _get_resnet_spec(num_layers)

    bn_global = False if session == 0 else True
    fix_layers = False if session == 0 else True
    net = CIFARResNetV1(CIFARBasicBlockV1, layers, channels,
                        bn_global=bn_global, fix_layers=fix_layers, **kwargs)
    if pretrained:
        model_file = os.path.join(root, 'cifar_resnet%d_%d.params'%(num_layers, session))
        net.load_parameters(model_file, ctx=ctx)
    return net

def cifar_resnet20_v1(session, **kwargs):
    return get_cifar_resnet(session, 20, **kwargs)
