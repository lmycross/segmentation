import torch.nn as nn
import torch
import numpy as np
from torchvision import models
affine_par = True


def outS(i):
    i = int(i)
    i = int(np.floor((i+1)/2))
    i = int(np.floor((i+1)/2))
    i = int(np.floor((i+1)/2))
    return i



class Classifier_Module(nn.Module):

    def __init__(self,dilation_series,padding_series,NoLabels):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation,padding in zip(dilation_series,padding_series):
            self.conv2d_list.append(nn.Conv2d(2048,NoLabels,kernel_size=3,stride=1, padding = padding, dilation = dilation,bias = True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out



class ResNet(nn.Module):
    def __init__(self, NoLabels):
        super(ResNet, self).__init__()
        resnet = models.resnet101(pretrained=True)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        
        for idx in range(len(self.layer3)):
            self.layer3[idx].conv2.dilation = 2
            self.layer3[idx].conv2.padding = 2
        
        for idx in range(len(self.layer4)):
            self.layer4[idx].conv2.dilation = 4
            self.layer4[idx].conv2.padding = 4
    
        self.classifier = Classifier_Module([6,12,18,24], [6,12,18,24], NoLabels)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.classifier(x)

        return x

class Deeplab_Resnet(nn.Module):
    def __init__(self,NoLabels):
        super(Deeplab_Resnet,self).__init__()
        self.Scale = ResNet(NoLabels)   

    def forward(self,x):
        H = x.size()[2]
        W = x.size()[3]
        self.interp1 = nn.Upsample(size=(int(H*0.5), int(W*0.5)), mode='bilinear')
        self.interp2 = nn.Upsample(size=(int(H*0.75), int(W*0.75)), mode='bilinear')
        self.interp3 = nn.Upsample(size=(outS(H), outS(W)), mode='bilinear')
        out = []
        x2 = self.interp1(x)
        x3 = self.interp2(x)
        out.append(self.Scale(x))	# for original scale
        out.append(self.interp3(self.Scale(x2)))	# for 0.75x scale, resize to original scale
        out.append(self.Scale(x3))	# for 0.5x scale


        x2Out_interp = out[1]
        x3Out_interp = self.interp3(out[2])  # resize 0.5x size back to original scale
        temp1 = torch.max(out[0],x2Out_interp)
        out.append(torch.max(temp1,x3Out_interp))  # pixel-wise max over outputs of three scale
        return out





