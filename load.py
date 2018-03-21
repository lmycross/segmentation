import torch.utils.data as data
import torch
import os
import os.path
import imageio
from torchvision import transforms
import numpy as np
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


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



class SegNetData(data.Dataset):

    def __init__(self, root, is_transform=False):

        imgs, annot = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.annot = annot
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.is_transform = is_transform


    def __getitem__(self, index):
        path_to_img = self.imgs[index]
        path_to_target = self.annot[index]
        img = imageio.imread(path_to_img)
#        img = np.array(img, dtype=np.uint8)
        
        target = imageio.imread(path_to_target)

        if self.is_transform:
            img,target = self.transform(img,target)
        
           
        return img, target

    def __len__(self):
        return len(self.imgs)

    def transform(self, img, lbl):
#        img = img[:, :, ::-1]
#        img = img.astype(np.float64)
#        img -= self.mean
#        img = img.astype(float) / 255.0
#        # NHWC -> NCHW
#        img = img.transpose(2, 0, 1)
#
#        img = torch.from_numpy(img).float()
        
        transform1 = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize((0.41189488770418226, 0.4251328066237724, 0.432670702070482),(0.3070734955953852, 0.3116110784489235, 0.3072184293428751))
                ])
        img=transform1(img)
        
        lbl = torch.from_numpy(lbl).long()
        return img, lbl


