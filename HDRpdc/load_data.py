import os
import glob
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import imageio
from PIL import Image

# 数据集加载和预处理
class HDRDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.sequences = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        self.transform = transform

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_dir = os.path.join(self.data_dir, self.sequences[idx])
        
        # Find LDR images
        ldr_image_paths = sorted(glob.glob(os.path.join(seq_dir, '*.tif')))
        ldr_images = []
        for ldr_image_path in ldr_image_paths[:3]:  # Take only the first 3 TIFF images
            ldr_image = Image.open(ldr_image_path).convert('RGB')
            if self.transform:
                ldr_image = self.transform(ldr_image)
            ldr_images.append(ldr_image)
        
        # Find HDR image
        hdr_image_paths = glob.glob(os.path.join(seq_dir, '*.hdr'))
        if len(hdr_image_paths) > 0:
            hdr_image_path = hdr_image_paths[0]  # Take the first HDR image found
            hdr_image = imageio.imread(hdr_image_path)
            hdr_image = Image.fromarray((hdr_image * 255).astype('uint8')).convert('RGB')
            if self.transform:
                hdr_image = self.transform(hdr_image)
        else:
            hdr_image = None  # Handle case where no HDR image is found
        
        return ldr_images, hdr_image

transform = transforms.Compose([
    transforms.RandomCrop((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])

