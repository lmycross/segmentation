import sys
sys.path.append('../../')
import datetime
from datasets.cityscapes_loader import Loaddata, class_weight, mean, std
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import models.deeplab_v2 as deeplab_v2
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torchvision import transforms
import argparse
import utils.jointtransform as jointtransform
from tensorboardX import SummaryWriter
import os
from utils import check_mkdir

def get_1x_lr_params(model):
    b = []

    b.append(model.layer0.parameters())
    b.append(model.layer1.parameters())
    b.append(model.layer2.parameters())
    b.append(model.layer3.parameters())
    b.append(model.layer4.parameters())

    
    for j in range(len(b)):
        for i in b[j]:
            yield i

def get_10x_lr_params(model):
    b = []
    b.append(model.classifier.parameters())


    for j in range(len(b)):
        for i in b[j]:
            yield i


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average,ignore_index=255)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs,dim=1), targets)

    
def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]
    return lr        


def lr_poly(base_lr, iter,max_iter,power):
    return base_lr*((1-float(iter)/max_iter)**(power))


def train_model(model, criterion, optimizer,base_lr, num_epochs,train_dataset,valid_dataset):

    best_acc = 0.0
    
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        
        model.train()
        running_loss = 0.0
        running_corrects = 0
        num_pixel = 0
        train_size = 0
        valid_size = 0
        max_iter = num_epochs*len(train_dataset)
        for j, data in enumerate(train_dataset):
            
            inputs, labels = data      
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda().long())
            
            lr_ = lr_poly(base_lr,epoch*len(train_dataset)+j,max_iter,0.9)
            optimizer.param_groups[0]['lr'] = lr_
            optimizer.param_groups[1]['lr'] = lr_
            optimizer.zero_grad()
            
            outputs = model(inputs) 
            outputs = F.upsample(outputs, scale_factor=8, mode='bilinear')
            loss = criterion(inputs=outputs, targets=labels)  
            loss.backward()
            optimizer.step()
            
            running_loss += loss.data[0]* inputs.size(0) 
            train_size += inputs.size(0)
            
            if (j+1) % 10 == 0:
                print("Iteartion %d Loss: %.4f" % (int(j)+1, loss.data[0] ))
        #epoch+=10    
        epoch_loss = running_loss / train_size     
        print("Epoch %d train Loss: %.4f" % (epoch+1, epoch_loss))
        writer.add_scalar('data/train_loss', epoch_loss, epoch)
    
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        num_pixel = 0
        for j, data in enumerate(valid_dataset):
           
            inputs, labels = data
           
            inputs = Variable(inputs.cuda(),volatile=True)
            labels = Variable(labels.cuda().long(),volatile=True)
            
            outputs = model(inputs)
            outputs = F.upsample(outputs, scale_factor=8, mode='bilinear')
            loss = criterion(inputs=outputs, targets=labels)
                
            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels.data.cpu().numpy()
            mask=gt<19
            num=np.sum(mask)
            num_pixel+=num
            pred=pred[mask]
            gt=gt[mask]
            
            
            running_loss += loss.data[0]* inputs.size(0) 
            valid_size += inputs.size(0) 
            running_corrects += np.sum(pred == gt)
            
        valid_loss = running_loss / valid_size
        valid_acc = running_corrects / num_pixel      
        print("Epoch %d valid Loss: %.4f Acc: %.4f" % (epoch+1, valid_loss,valid_acc))
        writer.add_scalar('data/val_loss', valid_loss, epoch)
        writer.add_scalar('data/val_acc', valid_acc, epoch)
        
        if valid_acc > best_acc:
            best_acc = valid_acc
        

        torch.save(model.state_dict(), os.path.join(ckpt_path, exp_name, 'net_params'+str(epoch+1)+'.pth'))   
        torch.save(optimizer.state_dict(), os.path.join(ckpt_path, exp_name, 'optimizer'+str(epoch+1)+'.pth'))
        print()
    
    writer.export_scalars_to_json(os.path.join(ckpt_path, 'exp', exp_name, "./all_scalars.json"))
    writer.close()  
    print('Best val Acc: {:4f}'.format(best_acc))
       

def main(train_args):
    
    path="/mnt/iusers01/eee01/mchiwml4/dataset/cityscapes/"
    
    transform = transforms.Compose([
                transforms.Normalize(mean, std)
                ])
    
    train_joint_transform = jointtransform.Compose([
        jointtransform.Resize((1024, 512)),
        jointtransform.RandomHorizontallyFlip(),
        jointtransform.ToTensor()
    ])
    
    val_joint_transform = jointtransform.Compose([
        jointtransform.ToTensor()
    ])
    
    train_dataset = DataLoader(
            Loaddata(path, split="train", transform=transform, joint_transform=train_joint_transform),
            batch_size=1, shuffle=True, num_workers=4)
    valid_dataset = DataLoader(
            Loaddata(path, split="val", transform=transform, joint_transform=val_joint_transform),
            batch_size=1, shuffle=False, num_workers=4)
    
    label_num=19
    model = deeplab_v2.Deeplab_Resnet(label_num)
    model=model.cuda()
    
    weight = torch.Tensor(class_weight).cuda()
    
    criterion = CrossEntropyLoss2d(weight=weight).cuda()
    
    lr_ = train_args.lr
    optimizer = optim.SGD([ {'params': get_1x_lr_params(model),'lr':lr_},
                            {'params': get_10x_lr_params(model),'lr':lr_}
                            ],momentum=train_args.momentum,weight_decay=train_args.weight_decay)
        
    
    if train_args.load_param is not None:
        model.load_state_dict(torch.load(train_args.load_param))  
    if train_args.load_optim is not None:   
        optimizer.load_state_dict(torch.load(train_args.load_optim))  
    
    max_epochs = train_args.epoch_num
            
    train_model(model, criterion, optimizer,lr_, max_epochs,train_dataset,valid_dataset)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--epoch_num', nargs='?', type=int, default=50, 
                        help='Max training epoch')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1, 
                        help='Batch size')
    parser.add_argument('--lr', nargs='?', type=float, default=1e-3, 
                        help='Learning Rate')
    parser.add_argument('--weight_decay', nargs='?', type=float, default=5e-4, 
                        help='Weight decay')
    parser.add_argument('--momentum', nargs='?', type=float, default=0.9, 
                        help='momentum')
    parser.add_argument('--load_param', nargs='?', type=str, default=None, 
                        help='Path to previous saved parameters to restart from')
    parser.add_argument('--load_optim', nargs='?', type=str, default=None, 
                        help='Path to previous saved optimizer to restart from')
    args = parser.parse_args()
    
    ckpt_path = '../../net_data/cityscapes'
    exp_name = 'cityscapes-deeplab_v2'
    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt'), 'w').write(str(args) + '\n\n')
    writer = SummaryWriter(os.path.join(ckpt_path, 'exp', exp_name))
    
    main(args)





