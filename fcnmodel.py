import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()

# FCN32s
class fcn32s(nn.Module):

    def __init__(self, label_nbr=21, learned_billinear=False):
        super(fcn32s, self).__init__()
        self.learned_billinear = learned_billinear
        self.label_nbr = label_nbr

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, self.label_nbr, 1),)
        
        # zero initialization as paper
        self.classifier[6].weight.data.zero_()
        self.classifier[6].bias.data.zero_()
               
        self.upscore32 = nn.ConvTranspose2d(self.label_nbr, self.label_nbr, 64, stride=32, bias=False)
        
        if learned_billinear is False:
            for param in self.upscore32.parameters():
                param.requires_grad = False
    
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)

        score = self.classifier(conv5)
        
        out=self.upscore32(score)
        out = out[:, :, 19:19 + x.size()[2], 19:19 + x.size()[3]].contiguous()
        
            
        return out

    def init_vgg16_params(self, vgg16, copy_fc8=False):
        blocks = [self.conv_block1,
                  self.conv_block2,
                  self.conv_block3,
                  self.conv_block4,
                  self.conv_block5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0]:ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    # print(idx, l1, l2)
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
        for i1, i2 in zip([0, 3], [0, 3]):
            l1 = vgg16.classifier[i1]
            l2 = self.classifier[i2]
            # print(type(l1), dir(l1))
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())
        n_class = self.classifier[6].weight.size()[0]
        if copy_fc8:
            l1 = vgg16.classifier[6]
            l2 = self.classifier[6]
            l2.weight.data = l1.weight.data[:n_class, :].view(l2.weight.size())
            l2.bias.data = l1.bias.data[:n_class]

class fcn16s(nn.Module):

    def __init__(self,label_nbr, learned_billinear=False):
        super(fcn16s, self).__init__()
        self.learned_billinear = learned_billinear
        self.label_nbr = label_nbr
        
        fcn32=fcn32s(label_nbr=self.label_nbr)
        
        self.conv_block1 = nn.Sequential(*list(fcn32.conv_block1.children()))

        self.conv_block2 = nn.Sequential(*list(fcn32.conv_block2.children()))

        self.conv_block3 = nn.Sequential(*list(fcn32.conv_block3.children()))

        self.conv_block4 = nn.Sequential(*list(fcn32.conv_block4.children()))

        self.conv_block5 = nn.Sequential(*list(fcn32.conv_block5.children()))

        self.classifier = nn.Sequential(*list(fcn32.classifier.children()))

        self.score_pool4 = nn.Conv2d(512, self.label_nbr, 1)
        
        self.score_pool4.weight.data.zero_()
        self.score_pool4.bias.data.zero_()
            
        self.upscore2 = nn.ConvTranspose2d(self.label_nbr, self.label_nbr, kernel_size=4, stride=2, bias=False)
        self.upscore16 = nn.ConvTranspose2d(self.label_nbr, self.label_nbr, kernel_size=32, stride=16, bias=False)

        if learned_billinear is False:
            for param in self.upscore16.parameters():
                param.requires_grad = False
        
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)
                
    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)

        score = self.classifier(conv5)
        score4 = self.score_pool4(conv4)
        
        score = self.upscore2(score)
        score4 = score4[:,:,5:5+score.size()[2],5:5+score.size()[3]]

        score += score4
        
        out=self.upscore16(score)
        out = out[:, :, 27:27 + x.size()[2], 27:27 + x.size()[3]].contiguous()
       
        return out

    def init_fcn32s_params(self, fcn32):
        
        self.conv_block1 = nn.Sequential(*list(fcn32.conv_block1.children()))
        self.conv_block2 = nn.Sequential(*list(fcn32.conv_block2.children()))
        self.conv_block3 = nn.Sequential(*list(fcn32.conv_block3.children()))
        self.conv_block4 = nn.Sequential(*list(fcn32.conv_block4.children()))
        self.conv_block5 = nn.Sequential(*list(fcn32.conv_block5.children()))
        self.classifier = nn.Sequential(*list(fcn32.classifier.children()))

# FCN 8s
class fcn8s(nn.Module):

    def __init__(self,label_nbr, learned_billinear=False):
        super(fcn8s, self).__init__()
        self.learned_billinear = learned_billinear
        self.label_nbr = label_nbr
        
        fcn32=fcn32s(label_nbr=self.label_nbr)
        
        self.conv_block1 = nn.Sequential(*list(fcn32.conv_block1.children()))

        self.conv_block2 = nn.Sequential(*list(fcn32.conv_block2.children()))

        self.conv_block3 = nn.Sequential(*list(fcn32.conv_block3.children()))

        self.conv_block4 = nn.Sequential(*list(fcn32.conv_block4.children()))

        self.conv_block5 = nn.Sequential(*list(fcn32.conv_block5.children()))

        self.classifier = nn.Sequential(*list(fcn32.classifier.children()))

        self.score_pool4 = nn.Conv2d(512, self.label_nbr, 1)
        self.score_pool3 = nn.Conv2d(256, self.label_nbr, 1)
        
        self.score_pool3.weight.data.zero_()
        self.score_pool3.bias.data.zero_()
        self.score_pool4.weight.data.zero_()
        self.score_pool4.bias.data.zero_()
        
        self.upscore2 = nn.ConvTranspose2d(self.label_nbr, self.label_nbr, kernel_size=4, stride=2, bias=False)
        self.pool4_upscore2 = nn.ConvTranspose2d(self.label_nbr, self.label_nbr, kernel_size=4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(self.label_nbr, self.label_nbr, kernel_size=16, stride=8, bias=False)
        
        if learned_billinear is False:
            for param in self.upscore8.parameters():
                param.requires_grad = False
                
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)
      
    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)

        score = self.classifier(conv5)
        score4 = self.score_pool4(conv4)
        score3 = self.score_pool3(conv3)
        
        score = self.upscore2(score)
        
        score4 = score4[:,:,5:5+score.size()[2],5:5+score.size()[3]]
        score += score4
        score = self.pool4_upscore2(score)
        
        score3 = score3[:,:,9:9+score.size()[2],9:9+score.size()[3]]
        score += score3
        
        out= self.upscore8(score)
        out = out[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()
    
        return out

    def init_fcn16s_params(self, fcn16):
        
        self.conv_block1 = nn.Sequential(*list(fcn16.conv_block1.children()))
        self.conv_block2 = nn.Sequential(*list(fcn16.conv_block2.children()))
        self.conv_block3 = nn.Sequential(*list(fcn16.conv_block3.children()))
        self.conv_block4 = nn.Sequential(*list(fcn16.conv_block4.children()))
        self.conv_block5 = nn.Sequential(*list(fcn16.conv_block5.children()))
        self.classifier = nn.Sequential(*list(fcn16.classifier.children()))
        self.score_pool4 = fcn16.score_pool4
        self.upscore2 = fcn16.upscore2
