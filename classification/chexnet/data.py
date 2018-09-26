import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class ChestXRayData(Dataset):
    """
    Load chest xray dataset and corresponding labels
    """

    def __init__(self, data_directory: os.path, image_list_file: str, transform=None):
        """[summary]
        
        Arguments:
            data_directory {os.path} -- directory where the chest xray images are located
            image_list_file {str} -- file containing list of chest xray images
        
        Keyword Arguments:
            transform -- type of transform to perform over input Dataset (default: {None})
        """
        image_names = []
        labels = []

        with open(image_list_file, 'r') as f:
            for line in f:
                items = line.split()
                image_name = items[0]
                label = items[1:]
                label = [int(i) for i in label]
                image_name = os.path.join(data_directory, image_name)
                image_names.append(image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """[summary]
        
        Arguments:
            index {[type]} -- index of the item of interest
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        else:
            pass
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)



