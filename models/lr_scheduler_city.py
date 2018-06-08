import sys
sys.path.append('../')
import matplotlib.pyplot as plt
from datasets.cityscapes_loader import Loaddata, class_weight, mean, std
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import  linknetmodel_cat as linknetmodel
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torchvision import transforms
import utils.jointtransform as jointtransform
import os
    
    
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average,ignore_index=255)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs,dim=1), targets)
    

path = "/mnt/iusers01/eee01/mchiwml4/dataset/cityscapes/"
transform = transforms.Compose([
                transforms.Normalize(mean, std)
                ])
    
train_joint_transform = jointtransform.Compose([
        jointtransform.Resize((1024, 512)),
        jointtransform.RandomHorizontallyFlip(),
        jointtransform.ToTensor()
    ]) 
    
train_dataset = DataLoader(
            Loaddata(path, split="train", transform=transform, joint_transform=train_joint_transform),
            batch_size=2, shuffle=True, num_workers=4)
        
label_num=19
model = linknetmodel.linknet(label_num)
model=model.cuda()
weight = torch.Tensor(class_weight).cuda()
criterion = CrossEntropyLoss2d(weight=weight).cuda()

num_epochs = 1
max_iter = num_epochs*len(train_dataset)
min_lr, max_lr = 1e-6, 1
optimizer = optim.RMSprop(model.parameters(), lr=min_lr, momentum=0.9, weight_decay=2e-4)

learning_rate=[]
iter_loss=[]

for epoch in range(num_epochs):
    for j, data in enumerate(train_dataset):
        inputs, labels = data
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda().long())
#        x = (epoch*len(train_dataset)+j) / max_iter
#        lr_ = min_lr + (max_lr-min_lr) * x
        lr_ = min_lr * 1.009331 **(epoch*len(train_dataset)+j)
        optimizer.param_groups[0]['lr'] = lr_
        optimizer.zero_grad()
        outputs = model(inputs)  
        loss = criterion(inputs=outputs, targets=labels)
        loss.backward()
        optimizer.step()
        learning_rate.append(lr_)
        iter_loss.append(loss.data[0])

print(learning_rate)
print(iter_loss)
plt.plot(learning_rate, iter_loss)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Learning rate')
plt.ylabel('Loss')
        
        
        
        


