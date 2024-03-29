from torchvision import models
from .attention import Attention
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import torch.nn as nn
import torch
from .DCNv2.dcn_v2 import DCN

__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, transposed=False):
        padding = (kernel_size - 1) // 2
        if transposed and stride > 1:
            super(ConvBNReLU, self).__init__(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_planes, out_planes, kernel_size, stride=1, padding=padding,
                          groups=groups, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.ReLU6(inplace=True)
            )
        else:
            super(ConvBNReLU, self).__init__(
                nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.ReLU6(inplace=True)
            )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, transposed=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))

        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, transposed=transposed),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, width_mult=1.0, inverted_residual_setting=None, round_nearest=8, up=False):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 320 if up else 32
        last_channel = 1280

        if inverted_residual_setting is None:
            if not up:
                inverted_residual_setting = [
                    # t, c, n, s
                    [1, 16, 1, 1],
                    [6, 24, 2, 2],
                    [6, 32, 3, 2],
                    [6, 64, 4, 2],
                    [6, 96, 3, 1],
                    [6, 160, 3, 2],
                    [6, 320, 1, 1],
                ]
            else:
                inverted_residual_setting = list(reversed([
                    # t, c, n, s
                    [1, 24, 1, 2],
                    [6, 32, 2, 2],
                    [6, 64, 3, 1],
                    [6, 96, 4, 2],
                    [6, 160, 3, 1],
                    [1, 320, 1, 2]
                ]))
        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [block(320, inverted_residual_setting[0][1],
                          stride=2, expand_ratio=1,
                          transposed=False)] if up else [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                ratio = min(t, 1) if stride > 1 and up else t
                features.append(block(input_channel, output_channel, stride, expand_ratio=ratio, transposed=up))
                input_channel = output_channel
        # building last several layers
        if not up:
            features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, 1000),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        return x


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def mobilenet_v2(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


class MobileNetDetect(torch.nn.Module):
    @staticmethod
    def _get_deconv_cfg(deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        self.inplanes = 320
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])
            planes = num_filters[i]
            fc = DCN(self.inplanes, planes, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1)
            up = nn.ConvTranspose2d(
                    in_channels=planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False)

            layers.append(fc)
            layers.append(nn.BatchNorm2d(planes, momentum=0.1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(up)
            layers.append(nn.BatchNorm2d(planes, momentum=0.1))
            if self.attention:
                layers.append(Attention(planes, 16))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def __init__(self, heads, head_conv, attention=False):
        super(MobileNetDetect, self).__init__()
        self.backbone = mobilenet_v2(True)
        self.backbone.classifier = None
        self.heads = heads
        self.head_conv = head_conv
        self.attention = attention

        self.deconv_layers = self._make_deconv_layer(
            3,
            [256, 128, 64],
            [4, 4, 4],
        )

        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(64, head_conv,
                              kernel_size=1, padding=0, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, head_conv,
                              kernel_size=3, padding=1, groups=head_conv, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes,
                              kernel_size=1, stride=1,
                              padding=0, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(24, classes, kernel_size=1, stride=1, padding=0, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, input):
        backbone_layers = list(list(self.backbone.children())[0].children())
        backbone_scales = [backbone_layers[0](input)]
        y = backbone_scales[0]
        for layer in backbone_layers[1:-1]:
            ny = layer(y)
            if ny.shape != y.shape:
                backbone_scales.append(y)
            y = ny
        y = self.deconv_layers(y)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(y)
        return [ret]

    def get_head_params(self):
        params = []
        for head in self.heads:
            params += list(self.__getattr__(head).parameters())
        return params

    def get_deconv_params(self):
        params = []
        for layer in list(self.deconv_layers.children())[0:]:
            params += list(layer.parameters())
        for head in self.heads:
            params += list(self.__getattr__(head).parameters())
        return params

    def set_group_param(self, group):
        params_map = {0: self.get_head_params, 1: self.get_deconv_params, 2: self.parameters}
        return params_map[group]()


def get_mobile_net(heads, head_conv, attention=False):
    net = MobileNetDetect(heads, head_conv, attention)
    return net
