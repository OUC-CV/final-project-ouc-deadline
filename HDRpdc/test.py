import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
import net
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from hdrnet.metrics import hdr_vdp
import numpy as np


def calculate_psnr(img1, img2, max_val=1.0):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))


def calculate_ssim(img1, img2, data_range=1.0):
    return ssim(img1, img2, data_range=data_range, multichannel=True)


def calculate_hdr_vdp(img1, img2):
    return hdr_vdp(img1, img2)


def mu_law_mapping(img, mu=5000):
    return np.log(1 + mu * img) / np.log(1 + mu)


def evaluate_model(model, dataloader, device):
    model.eval()
    psnr_l, psnr_mu, ssim_l, ssim_mu, hdr_vdp_scores = [], [], [], [], []

    with torch.no_grad():
        for ldr_images, true_hdr in dataloader:
            ldr_images = [img.to(device) for img in ldr_images]
            true_hdr = true_hdr.to(device)

            pred_hdr, _, _, _ = model(ldr_images)

            pred_hdr_np = pred_hdr.cpu().numpy().transpose((1, 2, 0))
            true_hdr_np = true_hdr.cpu().numpy().transpose((1, 2, 0))
            psnr_l.append(calculate_psnr(pred_hdr_np, true_hdr_np))
            ssim_l.append(calculate_ssim(pred_hdr_np, true_hdr_np))
            pred_hdr_mu = mu_law_mapping(pred_hdr_np)

            true_hdr_mu = mu_law_mapping(true_hdr_np)
            psnr_mu.append(calculate_psnr(pred_hdr_mu, true_hdr_mu))
            ssim_mu.append(calculate_ssim(pred_hdr_mu, true_hdr_mu))

            hdr_vdp_scores.append(calculate_hdr_vdp(pred_hdr_np, true_hdr_np))

    avg_psnr_l = np.mean(psnr_l)
    avg_psnr_mu = np.mean(psnr_mu)
    avg_ssim_l = np.mean(ssim_l)
    avg_ssim_mu = np.mean(ssim_mu)
    avg_hdr_vdp = np.mean(hdr_vdp_scores)

    return avg_psnr_l, avg_psnr_mu, avg_ssim_l, avg_ssim_mu, avg_hdr_vdp


def test(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        for ldr_images, true_hdr in test_loader:
            ldr_images = [img.to(device) for img in ldr_images]
            true_hdr = true_hdr.to(device)
            pred_hdr, f2_2_aligned_1, f2_2_aligned_3, f2_2 = model(ldr_images)
            loss = net.compute_loss(pred_hdr, true_hdr, f2_2_aligned_1, f2_2_aligned_3, f2_2)
            total_loss += loss.item()
        print(f'Test Loss: {total_loss / len(test_loader):.4f}')
        avg_psnr_l, avg_psnr_mu, avg_ssim_l, avg_ssim_mu, avg_hdr_vdp = evaluate_model(model, test_loader, device)
        print(f'PSNR-L: {avg_psnr_l:.4f}')
        print(f'PSNR-μ: {avg_psnr_mu:.4f}')
        print(f'SSIM-L: {avg_ssim_l:.4f}')
        print(f'SSIM-μ: {avg_ssim_mu:.4f}')
        print(f'HDR-VDP: {avg_hdr_vdp:.4f}')
