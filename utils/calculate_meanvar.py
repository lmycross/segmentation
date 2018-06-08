from PIL import Image
import os
import math
from torchvision import transforms

r = 0 # r mean
g = 0 # g mean
b = 0 # b mean

r_2 = 0 # r^2 
g_2 = 0 # g^2
b_2 = 0 # b^2

total = 0
path='/mnt/iusers01/eee01/mchiwml4/dataset/cityspaces/leftImg8bit_trainvaltest/leftImg8bit/train/'

        
for folder in os.listdir(path):
    folder_path = path +'/'+ folder
    for img_file in os.listdir(folder_path):
        img_path = folder_path +'/'+ img_file
        img = Image.open(img_path) # ndarray,  height x width x 3
        transform1 = transforms.Compose([
                    transforms.ToTensor(), 
                    ])
        img=transform1(img)
    #    img = img.astype('float32') / 255.
        total += img.shape[2] * img.shape[1]
        
        r += img[0, :, :].sum()
        g += img[1, :, :].sum()
        b += img[2, :, :].sum()
        
        r_2 += (img[0, :, :]**2).sum()
        g_2 += (img[1, :, :]**2).sum()
        b_2 += (img[2, :, :]**2).sum()

r_mean = r / total
g_mean = g / total
b_mean = b / total

r_var = r_2 / total - r_mean ** 2
g_var = g_2 / total - g_mean ** 2
b_var = b_2 / total - b_mean ** 2

r_std = math.sqrt(r_var)
g_std = math.sqrt(g_var)
b_std = math.sqrt(b_var)

print(r_mean,g_mean,b_mean)
print(r_std,g_std,b_std)