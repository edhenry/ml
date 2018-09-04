from data import ChestXRayData
from pathlib import Path
from pathlib import PureWindowsPath
from torchvision.transforms import transforms
import torchvision
from torch.utils.data import DataLoader

def main():
    """Tests for the ChestXRayData Object to ensure images are loaded appropriately

    """
    data_dir = Path('C:/Users/Ed Henry/Documents/code/ml/classification/chexnet/chexnet-data/images/')
    data_dir = PureWindowsPath(data_dir)
    test_image_list = Path('C:/Users/Ed Henry/Documents/code/ml/classification/chexnet/test_list.txt')
    test_image_list = PureWindowsPath(test_image_list)
    batch_size = 64

    test_dataset = ChestXRayData(data_directory=data_dir, image_list_file=test_image_list,
                                 transform=transforms.Compose([
                                     transforms.Resize(256),
                                     transforms.TenCrop(224),
                                     transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                     transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                 ]))

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=1, pin_memory=True)

if __name__ == "__main__":
    main()
