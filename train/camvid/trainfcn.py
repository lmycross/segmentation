import sys
sys.path.append('../../')
from datasets.camvid_loader import Loaddata, class_weight, mean, std, MaskToTensor
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import models.fcnmodel as fcnmodel
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import utils.jointtransform as jointtransform


trainset="/mnt/iusers01/eee01/mchiwml4/CamVid/train"
validset="/mnt/iusers01/eee01/mchiwml4/CamVid/val"
testset="/mnt/iusers01/eee01/mchiwml4/CamVid/test"
transform = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize(mean, std)
                ])
train_joint_transform = jointtransform.Compose([
    jointtransform.RandomHorizontallyFlip()
])

train_dataset = DataLoader(
        Loaddata(trainset, transform=transform, target_transform=MaskToTensor(), joint_transform=train_joint_transform),
        batch_size=4, shuffle=True, num_workers=8)
valid_dataset = DataLoader(
        Loaddata(validset, transform=transform, target_transform=MaskToTensor()),
        batch_size=1, shuffle=True, num_workers=8)



train_size = len(Loaddata(trainset))
valid_size = len(Loaddata(validset))
test_size = len(Loaddata(testset))
    
def get_1x_lr_params(model):
    b = []

    b.append(model.conv_block1.parameters())
    b.append(model.conv_block2.parameters())
    b.append(model.conv_block3.parameters())
    b.append(model.conv_block4.parameters())
    b.append(model.conv_block5.parameters())
    b.append(model.classifier.parameters())
    b.append(model.upscore2.parameters())
    b.append(model.score_pool4.parameters())

    
    for j in range(len(b)):
        for i in b[j]:
            yield i

def get_10x_lr_params(model):
    b = []
    b.append(model.score_pool3.parameters())
    b.append(model.pool4_upscore2.parameters())
    b.append(model.upscore8.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i
            
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.weight = weight
        self.nll_loss = nn.NLLLoss2d(weight, size_average,ignore_index=11)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs,dim=1), targets)
        
def train_model(model, criterion, optimizer, num_epochs=30):
    
    best_acc = 0.0
    
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        
        model.train()
        running_loss = 0.0
        running_corrects = 0
        num_pixel = 0
        for j, data in enumerate(train_dataset):
            

            inputs, labels = data
            
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())

            optimizer.zero_grad()
            outputs = model(inputs)
            

            loss = criterion(inputs=outputs, targets=labels)
            
            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels.data.cpu().numpy()
            mask=gt<11
            num=np.sum(mask)
            num_pixel+=num  # number of ground truth labeled pixels
            pred=pred[mask]
            gt=gt[mask]
            
            loss.backward()
            optimizer.step()
            
            
            running_loss += loss.data[0] * inputs.size(0)
            running_corrects += np.sum(pred == gt)
            
            if (j+1) % 20 == 0:
                print("Iteartion %d Loss: %.4f" % (int(j)+1, loss.data[0] ))
        epoch+=20    
        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects / num_pixel       
        print("Epoch %d train Loss: %.4f Acc: %.4f" % (epoch+1, epoch_loss,epoch_acc))
    
    
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        num_pixel = 0
        for j, data in enumerate(valid_dataset):
           

            inputs, labels = data
            
            inputs = Variable(inputs.cuda(),volatile=True)
            labels = Variable(labels.cuda(),volatile=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            

            loss = criterion(inputs=outputs, targets=labels)
            
            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels.data.cpu().numpy()
            mask=gt<11
            num=np.sum(mask)
            num_pixel+=num
            pred=pred[mask]
            gt=gt[mask]
            
            
            
            running_loss += loss.data[0] * inputs.size(0)
            running_corrects += np.sum(pred == gt)
            
        epoch_loss = running_loss / valid_size
        epoch_acc = running_corrects / num_pixel      
        print("Epoch %d valid Loss: %.4f Acc: %.4f" % (epoch+1, epoch_loss,epoch_acc))
        
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            
        torch.save(model.state_dict(), 'fcn8s-5/net_params'+str(epoch+1)+'.pkl')   
        print()
    
    print('Best val Acc: {:4f}'.format(best_acc))


label_num=11
model = fcnmodel.fcn32s(label_num)
model.load_state_dict(torch.load('fcn32s-4/net_params20.pkl')) 
model=model.cuda()
#vgg16=models.vgg16(pretrained=True)
#model.init_vgg16_params(vgg16)

model_ft = fcnmodel.fcn8s(label_num,learned_billinear=True)
model_ft.init_fcn16s_params(model)
model_ft=model_ft.cuda()
model_ft.load_state_dict(torch.load('fcn8s-5/net_params40.pkl'))

weight = torch.Tensor(class_weight).cuda()
criterion = CrossEntropyLoss2d(weight=weight)
optimizer = optim.SGD([
                        {'params': get_1x_lr_params(model_ft)},
                        {'params': get_10x_lr_params(model_ft), 'lr': 1e-5}
                       ], lr=1e-6,momentum=0.9,weight_decay=5e-4)

    
max_epochs = 20

train_model(model_ft, criterion, optimizer, num_epochs=max_epochs)


