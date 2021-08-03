##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""ResNeSt models"""

import torch
from resnet_new import ResNet, Bottleneck
#from .resnet_new import ResNet, Bottleneck
import torch.nn as nn
from Config import *
from Weight import Weight
import mmd

__all__ = ['resnest50', 'resnest101', 'resnest200', 'resnest269']
from build import RESNEST_MODELS_REGISTRY

_url_format = 'https://github.com/zhanghang1989/ResNeSt/releases/download/weights_step1/{}-{}.pth'

_model_sha256 = {name: checksum for checksum, name in [
    ('528c19ca', 'resnest50'),
    ('22405ba7', 'resnest101'),
    ('75117900', 'resnest200'),
    ('0cc87c48', 'resnest269'),
    ]}

def short_hash(name):
    if name not in _model_sha256:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha256[name][:8]

resnest_model_urls = {name: _url_format.format(name, short_hash(name)) for
    name in _model_sha256.keys()
}

class DSAN(nn.Module):
    # 实例化DSAN时，执行此操作。
    def __init__(self, num_classes=10):
        super(DSAN, self).__init__()
        # 输入参数为true表示加载利用ImageNet预训练好的resnet50模型
		# 之前此处调用的参数一直为true，即是用训练好的域训练模型，后续可以尝试使用没有预训练的模型进行试验。
        self.feature_layers = resnest50(False)
        self.num_classes = num_classes
        if bottle_neck:
            self.bottle = nn.Linear(2048, 256)
            self.cls_fc = nn.Linear(256, num_classes)
        else:
            self.cls_fc = nn.Linear(2048, num_classes)

    #当把DSAN当作一个方法来调用的时候，执行此操作。
    def forward(self, source, target, s_label):
        source = self.feature_layers(source)
        if bottle_neck:
            source = self.bottle(source)
        s_pred = self.cls_fc(source)
        if self.training ==True:
            target = self.feature_layers(target)
            if bottle_neck:
                target = self.bottle(target)
            t_label = self.cls_fc(target)
            #原始的lmmd
            loss = mmd.lmmd(source, target, s_label, torch.nn.functional.softmax(t_label, dim=1),num_classes = self.num_classes)
            #混合核lmmd
            #loss = mklmmd.mix_poly_rbf(source, target, s_label, torch.nn.functional.softmax(t_label, dim=1))
        else:
            loss = 0
        return s_pred, loss

@RESNEST_MODELS_REGISTRY.register()
def resnest50(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest50'], progress=True, check_hash=True))
    return model

@RESNEST_MODELS_REGISTRY.register()
def resnest101(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest101'], progress=True, check_hash=True))
    return model

@RESNEST_MODELS_REGISTRY.register()
def resnest200(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 24, 36, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest200'], progress=True, check_hash=True))
    return model

@RESNEST_MODELS_REGISTRY.register()
def resnest269(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 30, 48, 8],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest269'], progress=True, check_hash=True))
    return model
