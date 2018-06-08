import sys
sys.path.append('../../')
from datasets.camvid_loader import Loaddata, mean, std, MaskToTensor
from torch.autograd import Variable
from torch.utils.data import DataLoader
import models.deeplab_vggmodel as deeplab_vggmodel
import numpy as np
import torch.nn as nn
from metrics import runningScore
from torchvision import transforms
import argparse
import torch
import torch.nn.functional as F
from crf import dense_crf

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
    model = deeplab_vggmodel.deeplab(label_num)
    model=model.cuda()
    model.load_state_dict(torch.load(test_args.load_param))
    model.eval()

    total=np.zeros((label_num,))
    
    running_metrics = runningScore(label_num)  
    
    for j, data in enumerate(test_dataset):
        inputs, labels = data      
        inputs = Variable(inputs.cuda())
        
        outputs = model(inputs)

        H = inputs.size()[2]
        W = inputs.size()[3]
        expand_8x = nn.Upsample(size=(int(H), int(W)), mode='bilinear')
        outputs_8x = expand_8x(outputs.clone())
        
        outputs_8x = F.softmax(outputs_8x, dim=1)
        outputs_8x = outputs_8x.data.cpu().numpy()
        
        
        if test_args.crf:
            crf_output = np.zeros(outputs_8x.shape)
            images = inputs.data.cpu().numpy().astype(np.uint8)
            for i, (image, prob_map) in enumerate(zip(images, outputs_8x)):
                image = image.transpose(1, 2, 0)
                crf_output[i] = dense_crf(image, prob_map)
            outputs_8x = crf_output
        
        pred = np.argmax(outputs_8x, axis=1)
        gt = labels.numpy()
              
        running_metrics.update(gt, pred)
        
        for i in range(label_num):
            mask=gt==i  # ground truth mask of class i
            total[i]+=np.sum(mask)  # total number of pixels of class i (tp+fn)
            
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
    parser.add_argument('--crf', nargs='?', type=bool, default=True, 
                        help='Whether use dense crf')
    args = parser.parse_args()
    
    main(args)
    