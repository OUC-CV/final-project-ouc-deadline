import torch
import torch.nn as nn
import torch.nn.functional as F
import load_data
import train
import align

# 定义可变形卷积模块
class DeformableConvModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeformableConvModule, self).__init__()
        self.offset_conv = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=3, padding=1)
        self.deform_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        offset = self.offset_conv(x)
        out = F.grid_sample(x, offset)
        out = self.deform_conv(out)
        return out


# 定义膨胀残差密集块 (DRDB)
class DRDB(nn.Module):
    def __init__(self, in_channels, growth_rate=32, num_layers=load_data.textnum*6):
        super(DRDB, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1, dilation=2))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = F.relu(layer(torch.cat(features, 1)))
            features.append(out)
        return torch.cat(features, align.ImageAlignment.MINSIZE)


# 定义残差局部特征块 (RLFB)
class RLFB(nn.Module):
    def __init__(self, in_channels, num_blocks=3):
        super(RLFB, self).__init__()
        layers = []
        for _ in range(num_blocks):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x) + x


# 定义双分支网络
class DualBranchHDRNet(nn.Module):
    def __init__(self, num_channels=64):
        super(DualBranchHDRNet, self).__init__()
        self.deformable_conv = DeformableConvModule(num_channels, num_channels)
        self.drdb = DRDB(num_channels)
        self.rlfb = RLFB(num_channels)
        self.fusion_conv = nn.Conv2d(num_channels * 2, num_channels, kernel_size=3, padding=1)
        self.reconstruct_conv = nn.Conv2d(num_channels, 3, kernel_size=3, padding=1)

    def forward(self, ldr_images, ref_image):
        aligned_features = self.deformable_conv(ldr_images)
        fused_features = self.drdb(aligned_features)
        ref_features = self.rlfb(ref_image)
        combined_features = torch.cat([fused_features, ref_features], dim=1)
        fusion_features = self.fusion_conv(combined_features)
        hdr_image = self.reconstruct_conv(fusion_features)

        return hdr_image

model = DualBranchHDRNet()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for ldr_images, ref_image, hdr_target in train.train_dataloader:
        optimizer.zero_grad()
        output = model(ldr_images, ref_image)
        loss = criterion(output, hdr_target)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')