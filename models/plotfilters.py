
import torch
from torch import nn
from matplotlib import pyplot as plt

import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import ducmodel


model = ducmodel.ResNetDUCHDC(11)
model.load_state_dict(torch.load('C:\\Users\\mchiwml4\\pycode\\net-data\\segmentation\\camvid\\duc-3\\net_params45.pth'))
#num_ftrs = model.classifier[6].in_features
## convert all the layers to list and remove the last one
#features = list(model.classifier.children())[:-1]
### Add the last layer based on the num of classes in our dataset
#features.extend([nn.Linear(num_ftrs, 6)])
### convert it into container and add it to our model class.
#model.classifier = nn.Sequential(*features)

#model.classifier[6].out_features=6
#model.load_state_dict(torch.load('alexnetself5/net_params13.pkl')) 
filter = model.layer0[0].weight.data.numpy()


#filter[filter > 1] = 1
#filter[filter < 0] = 0
filter = (1/(2*abs(np.amin(filter))))*filter + 0.5 #Normalizing the values to [0,1]
filter=np.transpose(filter,(0,2,3,1))
#num_cols= choose the grid size you want
def plot_kernels(tensor, num_cols=8):
    if not tensor.ndim==4:
        raise Exception("assumes a 4D tensor")
    if not tensor.shape[-1]==3:
        raise Exception("last dim needs to be 3 to plot")
    num_kernels = tensor.shape[0]
    num_rows = 1+ num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    for i in range(tensor.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        ax1.imshow(tensor[i])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
    
    
plot_kernels(filter)



