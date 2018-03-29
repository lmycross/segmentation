from load2 import Loaddata
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import segnetmodel
import fcnmodel
import copy
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models
import numpy as np

trainset="/mnt/iusers01/eee01/mchiwml4/CamVid/train"
validset="/mnt/iusers01/eee01/mchiwml4/CamVid/val"
testset="/mnt/iusers01/eee01/mchiwml4/CamVid/test"
train_dataset = DataLoader(
        Loaddata(trainset, is_transform=True),
        batch_size=2, shuffle=True, num_workers=4)
valid_dataset = DataLoader(
        Loaddata(validset, is_transform=True),
        batch_size=1, shuffle=True, num_workers=4)
test_dataset = DataLoader(
        Loaddata(testset, is_transform=True),
        batch_size=1, shuffle=True, num_workers=4)



train_size = len(Loaddata(trainset))
valid_size = len(Loaddata(validset))
test_size = len(Loaddata(testset))
    

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
            num_pixel+=num
            pred=pred[mask]
            gt=gt[mask]
            
            loss.backward()
            optimizer.step()
            
            
            running_loss += loss.data[0] * inputs.size(0)
            running_corrects += np.sum(pred == gt)
            
            if (j+1) % 20 == 0:
                print("Iteartion %d Loss: %.4f" % (int(j)+1, loss.data[0] ))
        epoch+=40    
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
            
        torch.save(model.state_dict(), 'segnet-8/net_params'+str(epoch+1)+'.pkl')   
        print()
    
    print('Best val Acc: {:4f}'.format(best_acc))
    
    


def test_model(model, criterion):
    model.eval()
    running_loss = 0.0
    tp=np.zeros((label_num,))
    total=np.zeros((label_num,))
    fp=np.zeros((label_num,))
    for j, data in enumerate(test_dataset):
           

        inputs, labels = data
            
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())

        
        outputs = model(inputs)
            

        loss = criterion(inputs=outputs, targets=labels)
            
        pred = outputs.data.max(1)[1].cpu().numpy()
        gt = labels.data.cpu().numpy()
      
        
        
        for i in range(label_num):
            mask=gt==i  # ground truth mask of class i
            mask2=pred==i  # prediction mask of class i
            mioupred=pred[mask]  # predicted pixels of class i
            miougt=gt[mask]  # pixels of class i
            
            tp_temp = np.sum(mioupred == miougt)  # true positive 
            tp[i] += tp_temp
            total[i]+=np.sum(mask)  # total number of pixels of class i (tp+fn)
            fp[i]+=(np.sum(mask2)-tp_temp)  # false positive
            
              
        running_loss += loss.data[0] * inputs.size(0)
        
    print(tp/total)    
    print(total) 
    miou=np.sum(tp/(total+fp))/label_num          
    epoch_loss = running_loss / test_size
    avg_acc = np.sum(tp/total)/label_num  
    global_acc = np.sum(tp)/np.sum(total)     
    print("test Loss: %.4f Avg Acc: %.4f Global Acc: %.4f mIOU: %.4f" % (epoch_loss,avg_acc,global_acc,miou))


label_num=11
model = segnetmodel.segnet(label_nbr=label_num)

vgg16=models.vgg16_bn(pretrained=True)
model.init_vgg16_params(vgg16)

model=model.cuda()

weight = torch.Tensor([0.2595,0.1826,4.5640,0.1417,0.9051,0.3826,9.6446,1.8418,0.6823,6.2478,7.3614]).cuda()
criterion = CrossEntropyLoss2d(weight=weight)
optimizer = optim.SGD(model.parameters(), lr=1e-3*0.5*0.5,momentum=0.9,weight_decay=5e-4)

model.load_state_dict(torch.load('segnet-7/net_params40.pkl'))     


max_epochs = 20
train_model(model, criterion, optimizer, num_epochs=max_epochs)

test_model(model, criterion)








