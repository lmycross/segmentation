import sys
sys.path.append('../../')
from datasets.camvid_loader import Loaddata, mean, std, MaskToTensor
from torch.autograd import Variable
from torch.utils.data import DataLoader
import models.tiramisu_nobias as tiramisu_nobias
import numpy as np
from utils.metrics import runningScore
from torchvision import transforms
import argparse
import torch
import time

def main(test_args):
    
    testset="/mnt/iusers01/eee01/mchiwml4/CamVid/test"
    transform = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize(mean, std)
                ])
    test_dataset = DataLoader(
            Loaddata(testset, transform=transform, target_transform=MaskToTensor()),
            batch_size=1, shuffle=False, num_workers=8)
    
    label_num=11
    model = tiramisu_nobias.FCDenseNet103(label_num)
    model=model.cuda()
    model.load_state_dict(torch.load(test_args.load_param))
    model.eval()
    
    total=np.zeros((label_num,))
    
    running_metrics = runningScore(label_num)  
    
    time_elapsed = 0
    for j, data in enumerate(test_dataset):

        since = time.time()
        inputs, labels = data      
        inputs = Variable(inputs.cuda(), volatile=True)
        outputs = model(inputs)
        time_elapsed += time.time() - since 

        pred = outputs.data.max(1)[1].cpu().numpy()
        gt = labels.numpy()
        running_metrics.update(gt, pred)
        
        for i in range(label_num):
            mask=gt==i  # ground truth mask of class i
            total[i]+=np.sum(mask)  # total number of pixels of class i (tp+fn)

    print('Inference speed: {:.0f}ms, {:.0f}fps '.format(
          time_elapsed / len(test_dataset)*1000, 1 / (time_elapsed / len(test_dataset))))
    
    score, class_iou, class_acc = running_metrics.get_scores()
    
    for k, v in score.items():
        print(k, v)
    print('class iou: ')
    for i in range(label_num):
        print(i, class_iou[i])
    print('class acc: ')
    for i in range(label_num):
        print(i, class_acc[i])
          
    print('number of pixels:') 
    print(total)     

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--load_param', nargs='?', type=str, default=None, 
                        help='Path to pretrained parameters to test on')
    args = parser.parse_args()
    
    main(args)
    





