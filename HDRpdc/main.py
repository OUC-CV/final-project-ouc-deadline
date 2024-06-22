import torch
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
import net
import load_data
import test
import train
from torch.utils.data import Dataset, DataLoader

def sampleIntensities(images):
    z_min, z_max = 0, 255
    num_intensities = z_max - z_min + 1
    num_images = len(images)
    intensity_values = np.zeros((num_intensities, num_images), dtype=np.uint8)

    mid_img = images[num_images // 2]

    for i in range(z_min, z_max + 1):
        rows, cols = np.where(mid_img == i)
        if len(rows) != 0:
            idx = random.randrange(len(rows))
            for j in range(num_images):
                intensity_values[i, j] = images[j][rows[idx], cols[idx]]
    return intensity_values


def computeResponseCurve(intensity_samples, log_exposures, smoothing_lambda, weighting_function):
    z_min, z_max = 0, 255
    intensity_range = 255
    num_samples = intensity_samples.shape[0]
    num_images = len(log_exposures)

    mat_A = np.zeros((num_images * num_samples + intensity_range, num_samples + intensity_range + 1), dtype=np.float64)
    mat_b = np.zeros((mat_A.shape[0], 1), dtype=np.float64)

    k = 0
    for i in range(num_samples):
        for j in range(num_images):
            z_ij = intensity_samples[i, j]
            w_ij = weighting_function(z_ij)
            mat_A[k, z_ij] = w_ij
            mat_A[k, (intensity_range + 1) + i] = -w_ij
            mat_b[k, 0] = w_ij * log_exposures[j]
            k += 1

    for z_k in range(z_min + 1, z_max):
        w_k = weighting_function(z_k)
        mat_A[k, z_k - 1] = w_k * smoothing_lambda
        mat_A[k, z_k] = -2 * w_k * smoothing_lambda
        mat_A[k, z_k + 1] = w_k * smoothing_lambda
        k += 1

    mat_A[k, (z_max - z_min) // 2] = 1

    inv_A = np.linalg.pinv(mat_A)
    x = np.dot(inv_A, mat_b)

    g = x[0: intensity_range + 1]
    return g[:, 0]


if __name__ == "__main__":
    train_dataset = train('Dataset/Training', transform=train.transform)
    print(len(train_dataset))
    test_dataset = train('Dataset/Test/EXTRA', transform=train.transform)
    print(len(test_dataset))
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hdr_net = net()
    optimizer = torch.optim.Adam(hdr_net.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-8)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30000, gamma=0.95)

    # Training
    num_epochs = 25
    load_data(hdr_net, train_loader, optimizer, scheduler, num_epochs)

    # Testing
    test(hdr_net, test_loader,device)