import sys
sys.path.append('../../')
from datasets.cityscapes_loader import Loadtestdata, mean, std
from torch.autograd import Variable
from torch.utils.data import DataLoader
import models.linknetmodel as linknetmodel
from torchvision import transforms
import argparse
import torch
from PIL import Image
import os
from utils import check_mkdir


def main(test_args):
    
    path="/mnt/iusers01/eee01/mchiwml4/dataset/cityscapes/"
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                ])
    valid_dataset = DataLoader(
            Loadtestdata(path, split="val", transform=transform),
            batch_size=1, shuffle=False, num_workers=4)
    
    label_num=19
    model = linknetmodel.linknet(label_num)
    model=model.cuda()
    model.load_state_dict(torch.load(test_args.load_param))
    model.eval()
    
    
    valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
    class_map = dict(zip(range(19), valid_classes)) 
    
    for j, data in enumerate(valid_dataset):
        filepath, inputs = data      
        inputs = Variable(inputs.cuda(),volatile=True)
        outputs = model(inputs)
        pred = outputs.data.max(1)[1]
        
        for predid in range(18, -1, -1):
            pred[pred==predid] = class_map[predid]
            
        prediction = pred.cpu().squeeze_().float().numpy()
        prediction = Image.fromarray(prediction).convert("L")
        
        
        prediction.save(os.path.join(test_args.save_path, filepath[0]))
              


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--load_param', nargs='?', type=str, default=None, 
                        help='Path to pretrained parameters to test on')
    parser.add_argument('--save_path', nargs='?', type=str, default=None, 
                        help='Path to save predicted images')
    args = parser.parse_args()
    check_mkdir(args.save_path)
    main(args)
    




