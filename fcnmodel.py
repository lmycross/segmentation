import torch.nn as nn
import torch.nn.functional as F

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

        # TODO: Add support for learned upsampling
        if self.learned_billinear:
            raise NotImplementedError
            # upscore = nn.ConvTranspose2d(self.label_nbr, self.label_nbr, 64, stride=32, bias=False)
            # upscore.scale_factor = None


    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)

        score = self.classifier(conv5)

        out = F.upsample(score, x.size()[2:],mode='bilinear')

        return out


    def init_vgg16_params(self, vgg16, copy_fc8=True):
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

    def __init__(self, fcn32,label_nbr, learned_billinear=False):
        super(fcn16s, self).__init__()
        self.learned_billinear = learned_billinear
        self.label_nbr = label_nbr
        
        self.conv_block1 = nn.Sequential(*list(fcn32.conv_block1.children()))

        self.conv_block2 = nn.Sequential(*list(fcn32.conv_block2.children()))

        self.conv_block3 = nn.Sequential(*list(fcn32.conv_block3.children()))

        self.conv_block4 = nn.Sequential(*list(fcn32.conv_block4.children()))

        self.conv_block5 = nn.Sequential(*list(fcn32.conv_block5.children()))

        self.classifier = nn.Sequential(*list(fcn32.classifier.children()))

        self.score_pool4 = nn.Conv2d(512, self.label_nbr, 1)
        
        self.score_pool4.weight.data.zero_()
        self.score_pool4.weight.data.zero_()
        # TODO: Add support for learned upsampling
        if self.learned_billinear:
            raise NotImplementedError
            # upscore = nn.ConvTranspose2d(self.label_nbr, self.label_nbr, 64, stride=32, bias=False)
            # upscore.scale_factor = None


    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)

        score = self.classifier(conv5)
        score_pool4 = self.score_pool4(conv4)

        score = F.upsample(score,score_pool4.size()[2:],mode='bilinear')
        score += score_pool4
        out = F.upsample(score, x.size()[2:],mode='bilinear')

        return out




# FCN 8s
class fcn8s(nn.Module):

    def __init__(self, fcn16,label_nbr, learned_billinear=False):
        super(fcn8s, self).__init__()
        self.learned_billinear = learned_billinear
        self.label_nbr = label_nbr

        self.conv_block1 = nn.Sequential(*list(fcn16.conv_block1.children()))

        self.conv_block2 = nn.Sequential(*list(fcn16.conv_block2.children()))

        self.conv_block3 = nn.Sequential(*list(fcn16.conv_block3.children()))

        self.conv_block4 = nn.Sequential(*list(fcn16.conv_block4.children()))

        self.conv_block5 = nn.Sequential(*list(fcn16.conv_block5.children()))

        self.classifier = nn.Sequential(*list(fcn16.classifier.children()))

        self.score_pool4 = nn.Conv2d(512, self.label_nbr, 1)
        self.score_pool3 = nn.Conv2d(256, self.label_nbr, 1)

        # TODO: Add support for learned upsampling
        if self.learned_billinear:
            raise NotImplementedError
            # upscore = nn.ConvTranspose2d(self.label_nbr, self.label_nbr, 64, stride=32, bias=False)
            # upscore.scale_factor = None

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)

        score = self.classifier(conv5)
        score_pool4 = self.score_pool4(conv4)
        score_pool3 = self.score_pool3(conv3)

        score = F.upsample(score, score_pool4.size()[2:],mode='bilinear')
        score += score_pool4
        score = F.upsample(score, score_pool3.size()[2:],mode='bilinear')
        score += score_pool3
        out = F.upsample(score, x.size()[2:],mode='bilinear')
        return out


    def init_vgg16_params(self, vgg16, copy_fc8=True):
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
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
        for i1, i2 in zip([0, 3], [0, 3]):
            l1 = vgg16.classifier[i1]
            l2 = self.classifier[i2]
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())
        n_class = self.classifier[6].weight.size()[0]
        if copy_fc8:
            l1 = vgg16.classifier[6]
            l2 = self.classifier[6]
            l2.weight.data = l1.weight.data[:n_class, :].view(l2.weight.size())
            l2.bias.data = l1.bias.data[:n_class]
