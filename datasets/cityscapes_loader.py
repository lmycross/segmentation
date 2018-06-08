import os
import numpy as np
from torch.utils import data
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

class_weight = [ 3.045383480249677,
                 12.862127312658735,
                 4.509888876996228,
                 38.15694593009221,
                 35.25278401818165,
                 31.48260832348194,
                 45.79224481584843,
                 39.69406346608758,
                 6.0639281852733715,
                 32.16484408952653,
                 17.10923371690307,
                 31.5633201415795,
                 47.33397232867321,
                 11.610673599796504,
                 44.60042610251128,
                 45.23705196392834,
                 45.28288297518183,
                 48.14776939659858,
                 41.924631833506794]
mean = (0.286895532993708, 0.32513301755529556, 0.2838917665551808)
std = (0.18696374970616497, 0.19017339249151427, 0.187201942987801)

def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames if filename.endswith(suffix)]
    
class Loaddata(data.Dataset):
    """cityscapesLoader

    https://www.cityscapes-dataset.com

    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/

    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """
    colors = [#[  0,   0,   0],
              [128,  64, 128],
              [244,  35, 232],
              [ 70,  70,  70],
              [102, 102, 156],
              [190, 153, 153],
              [153, 153, 153],
              [250, 170,  30],
              [220, 220,   0],
              [107, 142,  35],
              [152, 251, 152],
              [ 70, 130, 180],
              [220,  20,  60],
              [255,   0,   0],
              [  0,   0, 142],
              [  0,   0,  70],
              [  0,  60, 100],
              [  0,  80, 100],
              [  0,   0, 230],
              [119,  11,  32]]

    label_colours = dict(zip(range(19), colors))

    def __init__(self, root, split="train", transform=None, target_transform=None, joint_transform=None):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations 
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self.n_classes = 19
        self.files = {}

        self.images_base = os.path.join(self.root, 'leftImg8bit_trainvaltest','leftImg8bit', self.split)
        self.annotations_base = os.path.join(self.root, 'gtFine_trainvaltest', 'gtFine', self.split)

        self.files[split] = recursive_glob(rootdir=self.images_base, suffix='.png')
    
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence',\
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain',\
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(19))) 

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))


    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        target_path = os.path.join(self.annotations_base,
                                img_path.split(os.sep)[-2], 
                                os.path.basename(img_path)[:-15] + 'gtFine_labelIds.png')

        img = Image.open(img_path)
        target = Image.open(target_path)

        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target) 
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        target = self.encode_segmap(target)
        
        return img, target


    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]
        r[temp == 255] = 0
        g[temp == 255] = 0
        b[temp == 255] = 0   
        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        #Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask==_voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask==_validc] = self.class_map[_validc]
        return mask


class Loadtestdata(data.Dataset):

    def __init__(self, root, split="train", transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.files = {}
        self.images_base = os.path.join(self.root, 'leftImg8bit_trainvaltest','leftImg8bit', self.split)
        self.files[split] = recursive_glob(rootdir=self.images_base, suffix='.png')
        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))


    def __len__(self):
        """__len__"""
        return len(self.files[self.split])


    def __getitem__(self, index):
        img_path = self.files[self.split][index].rstrip()
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)       
            
        path = img_path.split(os.sep)[-1]
        return path, img
    
    
if __name__ == '__main__':
#    joint_transform = jointtransform.Compose([
#                jointtransform.ToTensor()
#                ])
#
#    local_path = 'C:/Users/mchiwml4/dataset/segmentation/cityscapes'
#    dst = Loaddata(local_path,split="train", joint_transform=joint_transform)
#    bs = 4
#    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0, shuffle=False)
#    for i, image in enumerate(trainloader):
#        imgs, labels = image
#        imgs = imgs.numpy()
#        imgs = np.transpose(imgs, [0,2,3,1])
#        f, axarr = plt.subplots(bs,2)
#        for j in range(bs):      
#            axarr[j][0].imshow(imgs[j])
#            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
#        plt.show()
#        #a = raw_input()
#        if i == 0:
#            break
    transform = transforms.Compose([
                transforms.ToTensor()
                ])

    local_path = 'C:/Users/mchiwml4/dataset/segmentation/cityscapes'
    dst = Loadtestdata(local_path,split="val", transform=transform)
    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0, shuffle=False)
    for i, image in enumerate(trainloader):
        path, imgs = image
        imgs = imgs.numpy()
        imgs = np.transpose(imgs, [0,2,3,1])
        f, axarr = plt.subplots(bs,1)
        for j in range(bs):      
            axarr[j].imshow(imgs[j])
            print(path[j])
        plt.show()
        #a = raw_input()
        if i == 0:
            break


