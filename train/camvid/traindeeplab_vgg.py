import sys
sys.path.append('../../')
from datasets.camvid_loader import Loaddata, class_weight, mean, std, MaskToTensor
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import models.deeplab_vggmodel as deeplab_vggmodel
import torch
import torch.nn.functional as F
from torchvision import models
import numpy as np
import torch.nn as nn
from torchvision import transforms
import argparse
import utils.jointtransform as jointtransform
from utils import check_mkdir
import os
import datetime
from tensorboardX import SummaryWriter


def get_1x_lr_params(model):
    """
    This generator returns all the parameters of the net except for 
    the last classification layer. Note that for each batchnorm layer, 
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return 
    any batchnorm parameter
    """
    b = []

    b.append(model.down1.parameters())
    b.append(model.down2.parameters())
    b.append(model.down3.parameters())
    b.append(model.down4.parameters())
    b.append(model.down5.parameters())

    
    for j in range(len(b)):
        for i in b[j]:
            yield i

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """

    b = []
    b.append(model.classifier.parameters())


    for j in range(len(b)):
        for i in b[j]:
            yield i

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average,ignore_index=11)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs,dim=1), targets)
def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]
    return lr        

def train_model(model, criterion, optimizer,scheduler, num_epochs,train_dataset,valid_dataset,savefolder):
    
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
        for j, data in enumerate(train_dataset):
            

            inputs, labels = data
            
            
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())

            optimizer.zero_grad()
            outputs = model(inputs)
            
            labels.unsqueeze_(dim=1)
            H = outputs.size()[2]
            W = outputs.size()[3]
            shrink_8x = nn.Upsample(size=(int(H), int(W)), mode='bilinear')    
            labels_8x = shrink_8x(labels.clone())  # upsample transforms tensor to variable
            labels_8x.squeeze_(dim=1)    
            labels_8x = labels_8x.cuda().long()

            loss = criterion(inputs=outputs, targets=labels_8x)
            #loss=loss/N
            
            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels_8x.data.cpu().numpy()
            mask=gt<11
            num=np.sum(mask)
            num_pixel+=num
            pred=pred[mask]
            gt=gt[mask]
            
            
            loss.backward()
            optimizer.step()
            
            
            running_loss += loss.data[0]* inputs.size(0) 
            train_size += inputs.size(0)
            running_corrects += np.sum(pred == gt)
            
            if (j+1) % 10 == 0:
                print("Iteartion %d loss: %.4f" % (int(j)+1, loss.data[0] ))
        #epoch+=10    
        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects / num_pixel       
        print("Epoch %d train loss: %.4f Acc: %.4f" % (epoch+1, epoch_loss,epoch_acc))
        writer.add_scalar('data/train_loss', epoch_loss, epoch)
    
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
            
            labels.unsqueeze_(dim=1)
            H = outputs.size()[2]
            W = outputs.size()[3]
            shrink_8x = nn.Upsample(size=(int(H), int(W)), mode='bilinear')    
            labels_8x = shrink_8x(labels.clone())  # upsample transforms tensor to variable
            labels_8x.squeeze_(dim=1)    
            labels_8x = labels_8x.cuda().long()

            loss = criterion(inputs=outputs, targets=labels_8x)
            #loss.data/=N          
            
            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels_8x.data.cpu().numpy()
            mask=gt<11
            num=np.sum(mask)
            num_pixel+=num
            pred=pred[mask]
            gt=gt[mask]
            
            running_loss += loss.data[0]* inputs.size(0) 
            valid_size += inputs.size(0) 
            running_corrects += np.sum(pred == gt)
            
        valid_loss = running_loss / valid_size
        valid_acc = running_corrects / num_pixel      
        print("Epoch %d valid loss: %.4f acc: %.4f" % (epoch+1, valid_loss,valid_acc))
        writer.add_scalar('data/val_loss', valid_loss, epoch)
        writer.add_scalar('data/val_acc', valid_acc, epoch)
        scheduler.step(valid_loss)
        
        if valid_acc > best_acc:
            best_acc = valid_acc
            
        torch.save(model.state_dict(), os.path.join(ckpt_path, exp_name, 'net_params'+str(epoch+1)+'.pth'))   
        torch.save(optimizer.state_dict(), os.path.join(ckpt_path, exp_name, 'optimizer'+str(epoch+1)+'.pth'))
        print()
    
    writer.export_scalars_to_json(os.path.join(ckpt_path, 'exp', exp_name, "./all_scalars.json"))
    writer.close()    
    print('Best val acc: {:4f}'.format(best_acc))
       

def main(train_args):
    
    trainset="/mnt/iusers01/eee01/mchiwml4/CamVid/train"
    validset="/mnt/iusers01/eee01/mchiwml4/CamVid/val"
    
    transform = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize(mean, std)
                ])
    
    train_joint_transform = jointtransform.Compose([
        jointtransform.RandomHorizontallyFlip()
    ])
    
    train_dataset = DataLoader(
            Loaddata(trainset, transform=transform, target_transform=MaskToTensor(), joint_transform=train_joint_transform),
            batch_size=train_args.batch_size, shuffle=True, num_workers=8)
    valid_dataset = DataLoader(
            Loaddata(validset, transform=transform, target_transform=MaskToTensor()),
            batch_size=1, shuffle=True, num_workers=8)
    
    label_num=11
    model = deeplab_vggmodel.deeplab(label_num)
    
    vgg16=models.vgg16(pretrained=True)
    model.init_vgg16_params(vgg16)
    
    model=model.cuda()
    
    weight = torch.Tensor(class_weight).cuda()
    
    criterion = CrossEntropyLoss2d(weight=weight).cuda()
    
    lr_ = train_args.lr
    optimizer = optim.SGD([ {'params': get_1x_lr_params(model),'lr':lr_},
                            {'params': get_10x_lr_params(model),'lr':lr_*10}
                            ], lr=lr_,momentum=train_args.momentum,weight_decay=train_args.weight_decay)
        
#    optimizer.param_groups[0]['lr'] = lr_
#    optimizer.param_groups[1]['lr'] = lr_*10
    
    scheduler = ReduceLROnPlateau(optimizer, patience=10, min_lr=1e-10)
    
    if train_args.load_param is not None:
        model.load_state_dict(torch.load(train_args.load_param))  
    if train_args.load_optim is not None:   
        optimizer.load_state_dict(torch.load(train_args.load_optim))  
    
    max_epochs = train_args.epoch_num
    savefolder = train_args.save_folder
    train_model(model, criterion, optimizer, scheduler, max_epochs,train_dataset,valid_dataset,savefolder)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--epoch_num', nargs='?', type=int, default=50, 
                        help='Max training epoch')
    parser.add_argument('--batch_size', nargs='?', type=int, default=2, 
                        help='Batch size')
    parser.add_argument('--lr', nargs='?', type=float, default=1e-3, 
                        help='Learning Rate')
    parser.add_argument('--weight_decay', nargs='?', type=float, default=5e-4, 
                        help='Weight decay')
    parser.add_argument('--momentum', nargs='?', type=float, default=0.9, 
                        help='momentum')
    parser.add_argument('--save_folder', nargs='?', type=str, default=None, 
                        help='Folder to save model')
    parser.add_argument('--load_param', nargs='?', type=str, default=None, 
                        help='Path to previous saved parameters to restart from')
    parser.add_argument('--load_optim', nargs='?', type=str, default=None, 
                        help='Path to previous saved optimizer to restart from')
    args = parser.parse_args()
    
    ckpt_path = '../../net_data/camvid'
    exp_name = 'camvid_deeplab_vgg'
    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt'), 'w').write(str(args) + '\n\n')
    writer = SummaryWriter(os.path.join(ckpt_path, 'exp', exp_name))
    
    main(args)

