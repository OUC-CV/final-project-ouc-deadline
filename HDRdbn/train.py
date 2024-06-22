import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import imageio


def read_hdr_image(hdr_path):
    return imageio.imread(hdr_path, format='HDR-FI').astype(np.float32)


def read_ldr_image(tif_path):
    return np.array(Image.open(tif_path)).astype(np.float32) / 255.0


def read_exposure_times(txt_path):
    with open(txt_path, 'r') as file:
        exposures = file.readlines()
    return [float(exposure.strip()) for exposure in exposures]


# 定义数据集类
class KalantariDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform

        self.scenes = sorted(os.listdir(dataset_dir))

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene_path = os.path.join(self.dataset_dir, self.scenes[idx])

        ldr_images = []
        ldr_files = sorted([f for f in os.listdir(scene_path) if f.endswith('.tif')])
        for ldr_file in ldr_files:
            ldr_image = read_ldr_image(os.path.join(scene_path, ldr_file))
            ldr_image = Image.fromarray((ldr_image * 255).astype(np.uint8))  # 转换为 PIL 图像
            if self.transform:
                ldr_image = self.transform(ldr_image)
            ldr_images.append(ldr_image)
        ldr_images = torch.stack(ldr_images, dim=0)  # Shape: [num_exposures, C, H, W]

        hdr_image = read_hdr_image(os.path.join(scene_path, 'HDRImg.hdr'))
        hdr_image = torch.tensor(hdr_image).permute(2, 0, 1)  # Shape: [C, H, W]

        exposure_times = read_exposure_times(os.path.join(scene_path, 'exposure.txt'))

        return ldr_images, hdr_image, exposure_times


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

train_dir = 'Dataset/Training'
test_dir = 'Dataset/Test'
train_dataset = KalantariDataset(train_dir, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

