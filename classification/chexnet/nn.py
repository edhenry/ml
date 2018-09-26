import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from pathlib import Path
from pathlib import PureWindowsPath

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from data import ChestXRayData
from sklearn.metrics import roc_auc_score


# checkpoint directory
chkpt_path = 'model.pth.tar'
num_classes = 14
class_names = [
    "Atelectasis", "Cardiomegaly", "Effusion",
    "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation",
    "Edema", "Emphysema", "Fibrosis", 
    "Pleural_Thickening", "Hernia"
]
data_dir = '/mnt/md0/datasets/chexnet-data/images'
test_image_list = '/mnt/md0/datasets/chexnet-data/test_list.txt'
batch_size = 64

def main():

    cudnn.benchmark = True

    #reuse densenet model available in torchvision (ease of implementation)
    model = torchvision.models.densenet121(num_classes).cuda()
    model  = torch.nn.DataParallel(model).cuda()

    if os.path.isfile(chkpt_path):
        print("Loading existing model checkpoint...")
        checkpoint = torch.load(chkpt_path)
        model.load_state_dict(checkpoint['state_dict'])
        print("Model checkpoint loaded successfully!")
    else:
        print("No existing model checkpoint found.")

    # Normalization according to original paper section 3.1
    # "normalize based on the mean and stddev of images in the ImageNet dataset"
    # reference imagenet.lua module from torch found here :
    # https://github.com/facebook/fb.resnet.torch/blob/8549d8b98ade293a07ac94b1494e344a22e6e27d/datasets/imagenet.lua#L67-L70
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    test_dataset = ChestXRayData(data_directory=data_dir, image_list_file=test_image_list,
                                 transform=transforms.Compose([
                                     transforms.Resize(256),
                                     transforms.TenCrop(224),
                                     transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                     transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                 ]))

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=1, pin_memory=True)

    ground_truth = torch.FloatTensor()
    ground_truth = ground_truth.cuda()
    prediction = torch.FloatTensor()
    prediction = prediction.cuda()

    model.eval()

    print(test_loader.dataset.image_names)

    for i, (inp, target) in enumerate(test_loader):
        target = target.cuda()
        ground_truth = torch.cat((ground_truth, target), 0)
        bs, n_crops, c, h, w = inp.size()
        input_var = torch.autograd.Variable(inp.view(-1, c, h, w).cuda(), violate=True)
        output = model(input_var)
        output_mean = output.view(bs, n_crops, -1).mean(1)
        prediction = torch.cat((prediction, output_mean.data), 0)
    

    def compute_auc(ground_truth, prediction):
        """Compute area under the reciver operator curve from prediction scores
        
        Arguments:
            ground_truth {Pytorch Tensor} -- Pytorch tensor on GPU in the shape of [n_samples, n_classes] - true binary labels
            prediction {Pytorch Tensor} -- Pytorch tensor on GPU in the shape of [n_samples, n_classes] - can be probability estimates of positive classes or binary decisions
        """

        aurocs = []
        ground_truth_np = ground_truth.cpu().numpy()
        prediction_np = prediction.cpu().numpy()
        for i in range(num_classes):
            aurocs.append(roc_auc_score(ground_truth_np[:, i], prediction_np[:, i]))
        
        return aurocs
    
class DenseNet121(nn.Module):
    """Wrapper for the densenet121 module available in pytorch
    
    """
    def __init__(self, out_size):

        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_features = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_features, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x
    
if __name__ == '__main__':
    main()



