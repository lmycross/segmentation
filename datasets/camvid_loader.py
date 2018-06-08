from torch.utils.data import DataLoader
import torch.utils.data as data
import torch
import os
import os.path
from torchvision import transforms
import numpy as np
from PIL import Image, ImageFile
import matplotlib.pyplot as plt


ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

class_weight = [5.603479582009111, 4.29526217910912, 33.5735969320392, 3.335498275510207, 15.485574406695266, 8.738715632316753, 
              31.541973276563166, 32.01203384130015, 12.828815882297924, 38.01410671061199, 43.907386358778936]
mean = (0.41189488770418226, 0.4251328066237724, 0.432670702070482)
std = (0.3070734955953852, 0.3116110784489235, 0.3072184293428751)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(d):
    images = []
    annot = []


    for root, _, fnames in sorted(os.walk(d)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                item = path
                images.append(item)

    for root, _, fnames in sorted(os.walk("%sannot" % d)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                item = path
                annot.append(item)

    return images, annot


# transform label images to floattensor
class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).float()

class Loaddata(data.Dataset):

    def __init__(self, root, transform=None, target_transform=None, joint_transform=None):

        imgs, annot = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.annot = annot
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self.n_classes = 12

    def __getitem__(self, index):
        path_to_img = self.imgs[index]
        path_to_target = self.annot[index]
        img = Image.open(path_to_img)
#        img = np.array(img, dtype=np.uint8)
        
        target = Image.open(path_to_target)

        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target) 
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
               
                
        return img, target

    def __len__(self):
        return len(self.imgs)


    def decode_segmap(self, temp):
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
        for l in range(0, self.n_classes):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb
    
if __name__ == '__main__':
    local_path = 'C:/Users/mchiwml4/dataset/segmentation/camvid/train'
    joint_transform = jointtransform.Compose([
                jointtransform.RandomCrop(224),
                jointtransform.RandomHorizontallyFlip() 
                ])
    transform = transforms.Compose([
                transforms.ToTensor() 
                ])
    dst = Loaddata(local_path, transform=transform, target_transform=MaskToTensor(), joint_transform= joint_transform)
    bs = 4
    trainloader = DataLoader(dst, batch_size=bs)
    for i, image in enumerate(trainloader):
        imgs, labels = image
        imgs = imgs.numpy()
        imgs = np.transpose(imgs, [0,2,3,1])
        labels.squeeze_()
        f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
        plt.show()
        if i == 0:
            break
        
