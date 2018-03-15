from load import SegNetData
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import segnetmodel
import fcnmodel
import copy
import torch
import torch.nn.functional as F
from torchvision import models
trainset="/mnt/iusers01/eee01/mchiwml4/CamVid/train"
validset="/mnt/iusers01/eee01/mchiwml4/CamVid/val"
testset="/mnt/iusers01/eee01/mchiwml4/CamVid/test"
train_dataset = DataLoader(
        SegNetData(trainset, is_transform=True),
        batch_size=2, shuffle=True, num_workers=4)
valid_dataset = DataLoader(
        SegNetData(validset, is_transform=True),
        batch_size=1, shuffle=True, num_workers=4)
test_dataset = DataLoader(
        SegNetData(testset, is_transform=True),
        batch_size=1, shuffle=True, num_workers=4)



train_size = len(SegNetData(trainset))
valid_size = len(SegNetData(validset))
test_size = len(SegNetData(testset))

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average,ignore_index=11)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs,dim=1), targets)
        
def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)
    
    mask = target >= 0
    target = target[mask]
#    weight = torch.Tensor([0.2595,0.1826,4.5640,0.1417,0.9051,0.3826,9.6446,1.8418,0.6823,6.2478,7.3614]).cuda()    
    
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False,ignore_index=11)
    if size_average:
        loss /= mask.data.sum()
    return loss

def train_model(model, criterion, optimizer, num_epochs=30):
    
    best_acc = 0.0
    
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for j, data in enumerate(train_dataset):
            

            inputs, labels = data
            
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())

            optimizer.zero_grad()
            outputs = model(inputs)
            

            loss = criterion(input=outputs, target=labels)
            
            pred = outputs.data.max(1)[1]
            gt = labels.data
            
            loss.backward()
            optimizer.step()
            
            
            running_loss += loss.data[0] * inputs.size(0)
            running_corrects += torch.sum(pred == gt)
            
            if (j+1) % 20 == 0:
                print("Iteartion %d Loss: %.4f" % (int(j)+1, loss.data[0] ))
        epoch+=30   
        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects / (train_size*480*360)       
        print("Epoch %d train Loss: %.4f Acc: %.4f" % (epoch+1, epoch_loss,epoch_acc))
    
    
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        for j, data in enumerate(valid_dataset):
           

            inputs, labels = data
            
            inputs = Variable(inputs.cuda(),volatile=True)
            labels = Variable(labels.cuda(),volatile=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            

            loss = criterion(input=outputs, target=labels)
            
            pred = outputs.data.max(1)[1]
            gt = labels.data
            
            
            
            running_loss += loss.data[0] * inputs.size(0)
            running_corrects += torch.sum(pred == gt)
            
        epoch_loss = running_loss / valid_size
        epoch_acc = running_corrects / (valid_size*480*360)       
        print("Epoch %d valid Loss: %.4f Acc: %.4f" % (epoch+1, epoch_loss,epoch_acc))
        
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            
        torch.save(model.state_dict(), 'fcn32s-2/net_params'+str(epoch+1)+'.pkl')   
        print()
    
    print('Best val Acc: {:4f}'.format(best_acc))
    
    


def test_model(model, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    for j, data in enumerate(test_dataset):
           

        inputs, labels = data
            
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())

        
        outputs = model(inputs)
            

        loss = criterion(input=outputs, target=labels)
            
        pred = outputs.data.max(1)[1]
        gt = labels.data
            
            
            
        running_loss += loss.data[0] * inputs.size(0)
        running_corrects += torch.sum(pred == gt)
            
    epoch_loss = running_loss / test_size
    epoch_acc = running_corrects / (test_size*480*360)       
    print("test Loss: %.4f Acc: %.4f" % (epoch_loss,epoch_acc))


label_num=11
model = fcnmodel.fcn32s(label_num)

vgg16=models.vgg16(pretrained=True)
model.init_vgg16_params(vgg16)

model=model.cuda()
criterion = cross_entropy2d
optimizer = optim.SGD(model.parameters(), lr=1e-4,momentum=0.9,weight_decay=5e-4)

model.load_state_dict(torch.load('fcn32s-2/net_params30.pkl'))     
max_epochs = 10

train_model(model, criterion, optimizer, num_epochs=max_epochs)

#test_model(model, criterion)





