import torch.nn as nn
import torch.nn.functional as F

    
class conv2DRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True, dilation=1):
        super(conv2DRelu, self).__init__()
       
        conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, 
                                 padding=padding, stride=stride, bias=bias, dilation=dilation)

        self.cbr_unit = nn.Sequential(conv_mod,
                                      nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs

class Down2(nn.Module):
    def __init__(self, in_size, out_size):
        super(Down2, self).__init__()
        self.conv1 = conv2DRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DRelu(out_size, out_size, 3, 1, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2,padding=1)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs= self.maxpool(outputs)
        return outputs


class Down3(nn.Module):
    def __init__(self, in_size, out_size,stride=2,padding=1,dilation=1):
        super(Down3, self).__init__()
        self.conv1 = conv2DRelu(in_size, out_size, 3, 1, padding=padding,dilation=dilation)
        self.conv2 = conv2DRelu(out_size, out_size, 3, 1, padding=padding,dilation=dilation)
        self.conv3 = conv2DRelu(out_size, out_size, 3, 1, padding=padding,dilation=dilation)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        outputs = self.maxpool(outputs)
        return outputs

    
class deeplab(nn.Module):

    def __init__(self, label_nbr=21, in_channels=3,learned_billinear=False):
        super(deeplab, self).__init__()
        
        self.in_channels = in_channels
        self.label_nbr=label_nbr
        
        self.down1 = Down2(self.in_channels, 64)
        self.down2 = Down2(64, 128)
        self.down3 = Down3(128, 256)
        self.down4 = Down3(256, 512,stride=1)
        self.down5 = Down3(512, 512,stride=1,padding=2,dilation=2)
        self.avgpool = nn.AvgPool2d(kernel_size=3,stride=1,padding=1)
            
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 1024, 3,dilation=12,padding=12),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(1024, 1024, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(1024, label_nbr, 1))
        
    def forward(self, x):
        
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        x = self.avgpool(x)
        x=self.classifier(x)
        

        return x
    
    def init_vgg16_params(self, vgg16):
        blocks = [self.down1,
                  self.down2,
                  self.down3,
                  self.down4,
                  self.down5]

        features = list(vgg16.features.children())
        
        vgg_layers = []
        for _layer in features:
            if isinstance(_layer, nn.Conv2d):
                vgg_layers.append(_layer)

        merged_layers = []
        for idx, conv_block in enumerate(blocks):
            if idx < 2:
                units = [conv_block.conv1.cbr_unit,
                         conv_block.conv2.cbr_unit]
            else:
                units = [conv_block.conv1.cbr_unit,
                         conv_block.conv2.cbr_unit,
                         conv_block.conv3.cbr_unit]
            for _unit in units:
                for _layer in _unit:
                    if isinstance(_layer, nn.Conv2d):
                        merged_layers.append(_layer)

        assert len(vgg_layers) == len(merged_layers)

        for l1, l2 in zip(vgg_layers, merged_layers):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
            else:
                print('no')
        
