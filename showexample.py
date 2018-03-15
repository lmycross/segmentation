from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torch.autograd import Variable
import segnetmodel
import imageio
import fcnmodel


def decode_segmap(temp):
    Sky = [128, 128, 128]
    Building = [128, 0, 0]
    Pole = [192, 192, 128]
    Road = [128, 64, 128]
    Pavement = [60, 40, 222]
    Tree = [128, 128, 0]
    SignSymbol = [192, 128, 128]
    Fence = [64, 64, 128]
    Car = [64, 0, 128]
    Pedestrian = [64, 64, 0]
    Bicyclist = [0, 128, 192]
    Unlabelled = [0, 0, 0]


    label_colours = np.array([Sky, Building, Pole, Road, 
                                  Pavement, Tree, SignSymbol, Fence, Car, 
                                  Pedestrian, Bicyclist, Unlabelled])
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(12):
        r[temp == l] = label_colours[l, 0]
        g[temp == l] = label_colours[l, 1]
        b[temp == l] = label_colours[l, 2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r 
    rgb[:, :, 1] = g 
    rgb[:, :, 2] = b 

    return rgb

truthimagedir='C:/Users/mchiwml4/github/SegNet-Tutorial/CamVid/testannot/Seq05VD_f05010.png'
truth=np.asarray(Image.open(truthimagedir))
truth = decode_segmap(truth)
truth=np.uint8(truth)





imagedir='C:/Users/mchiwml4/github/SegNet-Tutorial/CamVid/test/Seq05VD_f05010.png'
inputs = imageio.imread(imagedir)
transform1 = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize((0.41189488770418226, 0.4251328066237724, 0.432670702070482),(0.09429413169716905, 0.09710146421210114, 0.09438316332790314))
                ])
img=transform1(inputs)
img=img.unsqueeze(0) 
img= Variable(img.cuda())

label_num=11
model = segnetmodel.segnet(label_num)
model.load_state_dict(torch.load('segnet-4/net_params64.pkl'))  

model=model.cuda()
model.eval()
outputs=model(img)

pred = outputs.data.max(1)[1].cpu().numpy()
pred=pred.squeeze(0) 
pre = decode_segmap(pred)
pre=np.uint8(pre)


plt.figure()
plt.imshow(inputs)
plt.axis('off')
plt.figure()
plt.imshow(pre)
plt.axis('off')
plt.figure()
plt.imshow(truth)
plt.axis('off')
plt.show() 
