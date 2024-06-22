import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


class UNetEncoder(nn.Module):
    def __init__(self):
        super(UNetEncoder, self).__init__()
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        f0 = self.enc1(x)  # H x W x 64
        f1 = self.pool(f0)  # H/2 x W/2 x 64
        f1 = self.enc2(f1)  # H/2 x W/2 x 128
        f2 = self.pool(f1)  # H/4 x W/4 x 128
        f2 = self.enc3(f2)  # H/4 x W/4 x 256
        return [f0, f1, f2]


class PDCM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PDCM, self).__init__()
        self.offset_conv1 = nn.Conv2d(in_channels * 2, 18, kernel_size=3, stride=1, padding=1)
        self.deform_conv1 = DeformConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.offset_conv2 = nn.Conv2d(out_channels * 2, 18, kernel_size=3, stride=1, padding=1)
        self.deform_conv2 = DeformConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.offset_conv3 = nn.Conv2d(out_channels * 2, 18, kernel_size=3, stride=1, padding=1)
        self.deform_conv3 = DeformConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, ref, nbr):
        offset1 = self.offset_conv1(torch.cat([ref, nbr], dim=1))
        deform1 = self.deform_conv1(nbr, offset1)

        offset2 = self.offset_conv2(torch.cat([ref, deform1], dim=1))
        deform2 = self.deform_conv2(deform1, offset2)

        offset3 = self.offset_conv3(torch.cat([ref, deform2], dim=1))
        deform3 = self.deform_conv3(deform2, offset3)

        return deform3


class CAM(nn.Module):
    def __init__(self, in_channels):
        super(CAM, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16),
            nn.ReLU(),
            nn.Linear(in_channels // 16, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.gap(x).view(b, c)
        y = self.mlp(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class LRB(nn.Module):
    def __init__(self, in_channels):
        super(LRB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.bn(out)
        out = self.conv2(out)
        return x + out


class LRG(nn.Module):
    def __init__(self, in_channels, num_blocks=9):
        super(LRG, self).__init__()
        self.blocks = nn.Sequential(*[LRB(in_channels) for _ in range(num_blocks)])

    def forward(self, x):
        return self.blocks(x) + x


class SAM(nn.Module):
    def __init__(self, in_channels):
        super(SAM, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        y = self.conv(avg_out + max_out)
        y = self.sigmoid(y)
        return x * y


class DASM(nn.Module):
    def __init__(self, in_channels):
        super(DASM, self).__init__()
        self.cam = CAM(in_channels)
        self.lrg = LRG(in_channels)
        self.sam = SAM(in_channels)

    def forward(self, F_hat1, F_hat3, F2):
        F = F_hat1 + F_hat3 + F2
        F = self.cam(F)
        F = self.lrg(F)
        F = self.sam(F)
        return F


class UNetDecoder(nn.Module):
    def __init__(self):
        super(UNetDecoder, self).__init__()
        self.upconv1 = self.upconv_block(256, 128)
        self.upconv2 = self.upconv_block(128, 64)
        self.upconv3 = nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2)

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.upconv1(x)  # H/4 x W/4 x 128
        x = self.upconv2(x)  # H/2 x W/2 x 64
        x = torch.sigmoid(self.upconv3(x))  # H x W x 3
        return x


class HDRNet(nn.Module):
    def __init__(self):
        super(HDRNet, self).__init__()
        self.encoder = UNetEncoder()
        self.pdcm1 = PDCM(256, 256)
        self.pdcm2 = PDCM(256, 256)
        self.dasm = DASM(256)
        self.decoder = UNetDecoder()

    def forward(self, L):
        f0_1, f1_1, f2_1 = self.encoder(L[0])
        f0_2, f1_2, f2_2 = self.encoder(L[1])
        f0_3, f1_3, f2_3 = self.encoder(L[2])

        f2_2_aligned_1 = self.pdcm1(f2_2, f2_1)
        f2_2_aligned_3 = self.pdcm2(f2_2, f2_3)

        fused_feature = self.dasm(f2_2_aligned_1, f2_2_aligned_3, f2_2)

        hdr = self.decoder(fused_feature)
        return hdr, f2_2_aligned_1, f2_2_aligned_3, f2_2


def compute_loss(pred_hdr, true_hdr, f2_2_aligned_1, f2_2_aligned_3, f2_2):
    # L1 loss for HDR image reconstruction
    L_net = F.l1_loss(pred_hdr, true_hdr)

    # L2 loss for alignment
    L1_to_2 = F.mse_loss(f2_2_aligned_1, f2_2)
    L3_to_2 = F.mse_loss(f2_2_aligned_3, f2_2)

    # Total loss
    loss = L_net + L1_to_2 + L3_to_2
    return loss
