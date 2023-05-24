import torch
import torch.nn as nn
import torch.nn.functional as F 
from AttentionBlock import ShallowAttention, DeepAttention_token
#from .utils import load_state_dict_from_url


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet_MS(nn.Module):

    def __init__(self, block, layers, num_classes=1000, norm_layer=None,
                    use_cls_token=True, num_subs=1):
        super(ResNet_MS, self).__init__()

        if norm_layer is None:
            #norm_layer = nn.BatchNorm2d
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer
        zero_init_residual = True

        self.inplanes = 64
        #self.base_width = width_per_group
        self.embed_dim = 512
        self.conv1 = nn.Conv3d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn2 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.embed_dim, layers[2], stride=2) 
        self.layer4 = self._make_layer(block, self.embed_dim, layers[3], stride=1)

        self.num_subs = num_subs
        self.hlv_token = nn.Parameter(torch.zeros(self.num_subs, 1, self.embed_dim))
        self.llv_token = nn.Parameter(torch.zeros(self.num_subs, 1, self.embed_dim))

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.use_cls_token = use_cls_token
        if self.use_cls_token:
            self.fc1 = nn.Linear(2 * self.embed_dim * block.expansion, num_classes)
        else:
            self.fc1 = nn.Linear(self.embed_dim * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

        torch.nn.init.normal_(self.hlv_token, std=.02)
        torch.nn.init.normal_(self.llv_token, std=.02)

        #self.attention1 = mlp_mixer_pytorch.MLPLayer(dim1=64, dim2=14*16*14)
        self.attention2 = ShallowAttention(dim1=128, dim2=14*16*14)
        self.attention3 = DeepAttention_token(dim=self.embed_dim, heads=4, dim_head=64, mlp_dim=256)
        self.attention4 = DeepAttention_token(dim=self.embed_dim, heads=4, dim_head=64, mlp_dim=512)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, sub_id):
        hlv_token = self.hlv_token[sub_id]
        llv_token = self.llv_token[sub_id]

        #print(x.size())
        # if x.size()[1] == 1:  # using the model pretrained on ImageNet
        #     x = x.repeat(1,3,1,1,1)
        #print(x.size())
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        #x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.attention2(x)

        x = self.layer3(x)
        x_attn, hlv_token, llv_token = self.attention3(x, hlv_token, llv_token)
        x = x + x_attn

        x = self.layer4(x)
        x_attn, hlv_token, llv_token = self.attention4(x, hlv_token, llv_token)
        x = x + x_attn
        fea = x

        hlv_token = torch.squeeze(hlv_token, 1)
        llv_token = torch.squeeze(llv_token, 1)
        if self.use_cls_token:
            mid_output = torch.cat((hlv_token, llv_token), 1)
        else:
            mid_output = self.avgpool(x)
            mid_output = mid_output.view(mid_output.size(0), -1)
        # print(mid_output.size())
        pred_logit = self.fc1(mid_output)

        return pred_logit, fea, hlv_token, llv_token

arch_settings = {
    10: (BasicBlock, (1, 1, 1, 1)),
    18: (BasicBlock, (2, 2, 2, 2)),
    34: (BasicBlock, (3, 4, 6, 3)),
    50: (Bottleneck, (3, 4, 6, 3)),
    101: (Bottleneck, (3, 4, 23, 3)),
    152: (Bottleneck, (3, 8, 36, 3))
}


def BrainFormer_MS(depth, num_classes=1000,
                    use_cls_token=True, num_subs=1):

    block, layers = arch_settings[depth]
    model = ResNet_MS(block, layers, num_classes=num_classes,
                    use_cls_token=use_cls_token, num_subs=num_subs)


    param = []
    params_dict = dict(model.named_parameters())
    for key, v in params_dict.items():  
        if 'attention' in key:
            #print('0.1 lr key', key)
            param += [{ 'params':v,  'lr_mult':0.1}]
        else:
            #print('lr key', key)
            param += [{ 'params':v,  'lr_mult':1}]


    return model, param
