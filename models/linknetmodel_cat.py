import torch.nn as nn
from torchvision import models
import torch


def center_crop(layer, max_height, max_width):
    _, _, h, w = layer.size()
    xy1 = (w - max_width) // 2
    xy2 = (h - max_height) // 2
    return layer[:, :, xy2:(xy2 + max_height), xy1:(xy1 + max_width)]

    
class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True, dilation=1):
        super(conv2DBatchNormRelu, self).__init__()

        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, 
                                 padding=padding, stride=stride, bias=bias, dilation=dilation)

        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, 
                                 padding=padding, stride=stride, bias=bias, dilation=1)

        self.cbr_unit = nn.Sequential(conv_mod,
                                      nn.BatchNorm2d(int(n_filters)),
                                      nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class deconv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, output_padding, bias=True):
        super(deconv2DBatchNormRelu, self).__init__()

        self.dcbr_unit = nn.Sequential(nn.ConvTranspose2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                padding=padding, stride=stride, output_padding=output_padding, bias=bias),
                                 nn.BatchNorm2d(int(n_filters)),
                                 nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.dcbr_unit(inputs)
        return outputs


class linknetUp(nn.Module):
    def __init__(self, in_channels, n_filters, stride=2, output_padding=1):
        super(linknetUp, self).__init__()
        self.convbnrelu1 = conv2DBatchNormRelu(in_channels, in_channels/4, k_size=1, stride=1, padding=0, bias=False)
        self.deconvbnrelu2 = deconv2DBatchNormRelu(in_channels/4, in_channels/4, k_size=3, stride=stride, padding=1, output_padding=output_padding, bias=False)
        self.convbnrelu3 = conv2DBatchNormRelu(in_channels/4, n_filters, k_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.convbnrelu1(x)
        x = self.deconvbnrelu2(x)
        x = self.convbnrelu3(x)
        return x

    
class linknet(nn.Module):
    def __init__(self, num_classes=21):
        super(linknet, self).__init__()
        self.num_classes = num_classes
        resnet = models.resnet18(pretrained=True)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, nn.MaxPool2d(2,2))
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        self.decoder4 = linknetUp(512, 256)
        self.decoder3 = linknetUp(512, 128)
        self.decoder2 = linknetUp(256, 64)
        self.decoder1 = linknetUp(128, 64, stride=1, output_padding=0)
        
        self.finaldeconvbnrelu1 = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),)
        self.finalconvbnrelu2 = conv2DBatchNormRelu(in_channels=32, n_filters=32, k_size=3, padding=1, stride=1, bias=False)
        self.finalconv3 = nn.ConvTranspose2d(32, num_classes, kernel_size=2, stride=2)
        
        
        b = []
        b.append(self.decoder4)
        b.append(self.decoder3)
        b.append(self.decoder2)
        b.append(self.decoder1)
        b.append(self.finaldeconvbnrelu1)
        b.append(self.finalconvbnrelu2 )
        b.append(self.finalconv3)
        
        for j in range(len(b)):
            for m in b[j].modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal(m.weight, mode='fan_out')
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.kaiming_normal(m.weight, mode='fan_out')
        
        
    def forward(self, x):
        x = self.layer0(x) 
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        d4 = self.decoder4(e4)
        d4 = center_crop(d4, e3.size(2), e3.size(3))
        d4 = torch.cat([d4, e3], 1)
        
        d3 = self.decoder3(d4)
        d3 = center_crop(d3, e2.size(2), e2.size(3))
        d3 = torch.cat([d3, e2], 1)
        
        d2 = self.decoder2(d3)
        d2 = center_crop(d2, e1.size(2), e1.size(3))
        d2 = torch.cat([d2, e1], 1)
        
        d1 = self.decoder1(d2)
        f = self.finaldeconvbnrelu1(d1)
        f = self.finalconvbnrelu2(f)
        f = self.finalconv3(f)
        
        return f

