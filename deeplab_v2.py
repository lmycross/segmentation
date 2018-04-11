import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
affine_par = True


def outS(i):
    i = int(i)
    i = int(np.floor((i+1)/2))
    i = int(np.floor((i+1)/2))
    i = int(np.floor((i+1)/2))
    return i

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine = affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine = affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,  dilation_ = 1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes,affine = affine_par)
#        for i in self.bn1.parameters():
#             i.requires_grad = False
        
        padding = 1
        if dilation_ == 2:
	         padding = 2
        elif dilation_ == 4:
	         padding = 4
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation = dilation_)
        self.bn2 = nn.BatchNorm2d(planes,affine = affine_par)
#        for i in self.bn2.parameters():
#            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par)
#        for i in self.bn3.parameters():
#            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride



    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Classifier_Module(nn.Module):

    def __init__(self,dilation_series,padding_series,NoLabels):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation,padding in zip(dilation_series,padding_series):
            self.conv2d_list.append(nn.Conv2d(2048,NoLabels,kernel_size=3,stride=1, padding =padding, dilation = dilation,bias = True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out



class ResNet(nn.Module):
    def __init__(self, block, layers,NoLabels):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64,affine = affine_par)
#        for i in self.bn1.parameters():
#            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation__ = 2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation__ = 4)
        self.layer5 = self._make_pred_layer(Classifier_Module, [6,12,18,24],[6,12,18,24],NoLabels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #        for i in m.parameters():
        #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1,dilation__ = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation__ == 2 or dilation__ == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par),
            )
#        for i in downsample._modules['1'].parameters():
#            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation_=dilation__, downsample = downsample ))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,dilation_=dilation__))

        return nn.Sequential(*layers)
    
    def _make_pred_layer(self,block, dilation_series, padding_series,NoLabels):
        return block(dilation_series,padding_series,NoLabels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x

class Deeplab_Resnet(nn.Module):
    def __init__(self,NoLabels):
        super(Deeplab_Resnet,self).__init__()
        self.Scale = ResNet(Bottleneck,[3, 4, 23, 3],NoLabels)   #changed to fix #4 

    def forward(self,x):
        H = x.size()[2]
        W = x.size()[3]
        self.interp1 = nn.Upsample(scale_factor=0.75, mode='bilinear', align_corners=True)
        self.interp2 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.interp3 = nn.Upsample(size=(outS(H), outS(W)), mode='bilinear', align_corners=True)
        out = []
        x2 = self.interp1(x)
        x3 = self.interp2(x)
        out.append(self.Scale(x))	# for original scale
        out.append(self.interp3(self.Scale(x2)))	# for 0.75x scale
        out.append(self.Scale(x3))	# for 0.5x scale


        x2Out_interp = out[1]
        x3Out_interp = self.interp3(out[2])
        temp1 = torch.max(out[0],x2Out_interp)
        out.append(torch.max(temp1,x3Out_interp))
        return out

    def init_resnet_params(self, resnet):
        
        self.Scale.conv1.weight.data = resnet.conv1.weight.data
        self.Scale.bn1.weight.data = resnet.bn1.weight.data
        self.Scale.bn1.bias.data = resnet.bn1.bias.data
        
        blocks = [self.Scale.layer1,
                  self.Scale.layer2,
                  self.Scale.layer3,
                  self.Scale.layer4]
                  

        blocks_2 = [resnet.layer1,
                    resnet.layer2,
                    resnet.layer3,
                    resnet.layer4]
        
        resnet_layers = []
        for block in blocks_2:
            for _layer in block.modules():
                
                if isinstance(_layer, nn.Conv2d):
                    resnet_layers.append(_layer)
                elif isinstance(_layer, nn.BatchNorm2d):
                    resnet_layers.append(_layer)
        
        merged_layers = []
        for block in blocks:
             for _layer in block.modules():
                
                if isinstance(_layer, nn.Conv2d):
                    merged_layers.append(_layer)
                elif isinstance(_layer, nn.BatchNorm2d):
                     merged_layers.append(_layer)


        assert len(resnet_layers) == len(merged_layers)

        for l1, l2 in zip(resnet_layers, merged_layers):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                
                l2.weight.data = l1.weight.data
                
                
            elif isinstance(l1, nn.BatchNorm2d) and isinstance(l2, nn.BatchNorm2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
            else:
                print('no')



