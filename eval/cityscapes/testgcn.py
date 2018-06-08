import sys
sys.path.append('../../')
from datasets.cityscapes_loader import Loadtestdata, mean, std
from torch.autograd import Variable
from torch.utils.data import DataLoader
import models.gcnmodel as gcnmodel
from torchvision import transforms
import argparse
import torch
from PIL import Image
import os


def main(test_args):
    
    path="~/dataset/cityscapes"
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                ])
    valid_dataset = DataLoader(
            Loadtestdata(path, split="val", transform=transform),
            batch_size=1, shuffle=False, num_workers=4)
    
    label_num=19
    model = gcnmodel.GCN(label_num)
    model=model.cuda()
    model.load_state_dict(torch.load(test_args.load_param))
    model.eval()
    
    valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
    class_map = dict(zip(range(19), valid_classes)) 
    
    for j, data in enumerate(valid_dataset):
        filepath, inputs = data      
        inputs = Variable(inputs.cuda(),volatile=True)
        
       # image_patch=[]
       # image_patch.append(inputs[:, :, 0:512, 0:1024])
       # image_patch.append(inputs[:, :, 0:512, 1024:2048])
       # image_patch.append(inputs[:, :, 512:1024, 0:1024])
       # image_patch.append(inputs[:, :, 512:1024, 1024:2048])
 
        
       # output_patch=[]
       # for i in range(4):
       #     outputs = model(image_patch[i])
       #     output_patch.append(outputs)
       #     
       # outputs = torch.cat((torch.cat((output_patch[0].data, output_patch[1].data), 3), 
       #                      torch.cat((output_patch[2].data, output_patch[3].data), 3)), 2)
        
        outputs = model(inputs)
        pred = outputs.data.max(1)[1]
        
        for predid in range(18, -1, -1):
            pred[pred==predid] = class_map[predid]
            
        prediction = pred.squeeze_().float().cpu().numpy()
        prediction = Image.fromarray(prediction).convert("L")
        
        prediction.save(os.path.join(test_args.save_path, filepath[0]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--load_param', nargs='?', type=str, default=None, 
                        help='Path to pretrained parameters to test on')
    parser.add_argument('--save_path', nargs='?', type=str, default=None, 
                        help='Path to save predicted images')
    args = parser.parse_args()
    
    main(args)
    



