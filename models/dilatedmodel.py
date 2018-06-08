import torch.nn as nn

    
class conv2DRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding=0, bias=True, dilation=1):
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
        self.conv1 = conv2DRelu(in_size, out_size,  k_size=3, stride=1)
        self.conv2 = conv2DRelu(out_size, out_size, k_size=3, stride=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2,padding=0)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs= self.maxpool(outputs)
        return outputs


class Down3(nn.Module):
    def __init__(self, in_size, out_size,dilation=1):
        super(Down3, self).__init__()
        self.conv1 = conv2DRelu(in_size, out_size, k_size=3, stride=1,dilation=dilation)
        self.conv2 = conv2DRelu(out_size, out_size, k_size=3, stride=1,dilation=dilation)
        self.conv3 = conv2DRelu(out_size, out_size, k_size=3, stride=1,dilation=dilation)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        
        outputs = self.maxpool(outputs)
        return outputs

class Down3_2(nn.Module):
    def __init__(self, in_size, out_size,dilation=1):
        super(Down3_2, self).__init__()
        self.conv1 = conv2DRelu(in_size, out_size, k_size=3, stride=1,dilation=dilation)
        self.conv2 = conv2DRelu(out_size, out_size, k_size=3, stride=1,dilation=dilation)
        self.conv3 = conv2DRelu(out_size, out_size, k_size=3, stride=1,dilation=dilation)
        
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs
    
class dilated(nn.Module):

    def __init__(self, label_nbr=21, in_channels=3,learned_billinear=False):
        super(dilated, self).__init__()
        
        self.in_channels = in_channels
        self.label_nbr=label_nbr
        
        self.down1 = Down2(self.in_channels, 64)
        self.down2 = Down2(64, 128)
        self.down3 = Down3(128, 256)
        self.down4 = Down3_2(256, 512)
        self.down5 = Down3_2(512, 512,dilation=2)
        
            
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7, dilation=4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, label_nbr, 1))
        
        self.ctx = nn.Sequential(
            nn.Conv2d(label_nbr, label_nbr, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(label_nbr, label_nbr, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(label_nbr, label_nbr, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(label_nbr, label_nbr, 3, padding=4, dilation=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(label_nbr, label_nbr, 3, padding=8, dilation=8),
            nn.ReLU(inplace=True),
            nn.Conv2d(label_nbr, label_nbr, 3, padding=16, dilation=16),
            nn.ReLU(inplace=True),
            nn.Conv2d(label_nbr, label_nbr, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(label_nbr, label_nbr, 3, padding=1))
        
    def forward(self, x):
        
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        x = self.ctx(x)

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
        
        l1 = vgg16.classifier[0]
        l2 = self.classifier[0]
        l2.weight.data = l1.weight.data.view(l2.weight.size())
        l2.bias.data = l1.bias.data.view(l2.bias.size())
        l1 = vgg16.classifier[3]
        l2 = self.classifier[3]
        l2.weight.data = l1.weight.data.view(l2.weight.size())
        l2.bias.data = l1.bias.data.view(l2.bias.size())

