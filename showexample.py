from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torch.autograd import Variable
import tiramisu_nobias
import imageio


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

label_num=11
model = tiramisu_nobias.FCDenseNet103(label_num)
model.load_state_dict(torch.load('C:/Users/mchiwml4/pycode/net-data/segmentation/camvid/tiramisu2/net_params65.pth'))  

model=model
model.eval()


truthimagedir='C:/Users/mchiwml4/github/SegNet-Tutorial/CamVid/testannot/Seq05VD_f00660.png'

truth=np.asarray(Image.open(truthimagedir))
truth = decode_segmap(truth)

truth=np.uint8(truth)





imagedir='C:/Users/mchiwml4/github/SegNet-Tutorial/CamVid/test/Seq05VD_f00660.png'
inputs = imageio.imread(imagedir)
transform1 = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize((0.41189488770418226, 0.4251328066237724, 0.432670702070482),(0.3070734955953852, 0.3116110784489235, 0.3072184293428751))
                ])
img=transform1(inputs)
img=img.unsqueeze(0) 
img= Variable(img,requires_grad=False)

outputs=model(img)

pred = outputs.data.max(1)[1].numpy()
pred=pred.squeeze(0) 
pre = decode_segmap(pred)
pre=np.uint8(pre)


plt.figure()
plt.imshow(inputs)
plt.axis('off')
plt.figure()
plt.imshow(truth)
plt.axis('off')
plt.figure()
plt.imshow(pre)
plt.axis('off')
plt.show() 
