import math
from cityspace_loader import Loaddata
import jointtransform

total = 0
local_path = 'C:/Users/mchiwml4/dataset/segmentation/cityscapes'
joint_transform = jointtransform.Compose([
                jointtransform.ToTensor()
                ])
dst = Loaddata(local_path,split="train", joint_transform=joint_transform)

poss = [0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0]
num = [0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0]
weight = [0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0]

j=0
for a, target in dst:
    target=target.view(-1)
    
    for i in range(19):
        num[i] += (target==i).sum().item() 
    
#    for i in target:
#        if i.item() != 255:
#            num[i.item()]+=1
    j+=1
    print(j)
total = sum(num)

for i in range(len(poss)):
    poss[i]=num[i]/5523279889.0
    weight[i]= 1/math.log(1.02+poss[i])
   
    
print(poss)