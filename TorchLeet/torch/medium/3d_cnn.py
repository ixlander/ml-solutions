import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F

torch.manual_seed(42)

batch = 10 
num_slices = 10
channels = 3
width = 256
height = 256

print("Generating synthetic data...")
ct_images = torch.randn(size=(batch, num_slices, channels, width, height))
segmentation_masks = (torch.randn(size=(batch, num_slices, 1, width, height)) > 0).float()

print(f"CT images shape: {ct_images.shape}")
print(f"Segmentation masks shape: {segmentation_masks.shape}")

class MedCNN(nn.Module):
    def __init__(self, backbone, out_channels=1):
        super(MedCNN, self).__init__()
        self.backbone = backbone
        
        self.down_conv = nn.Sequential(
            nn.Conv3d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU()
        )
        
        self.upsample1 = nn.ConvTranspose3d(256, 128, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.upsample2 = nn.ConvTranspose3d(128, 64,  kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.upsample3 = nn.ConvTranspose3d(64, 32,   kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.upsample4 = nn.ConvTranspose3d(32, 16,   kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.upsample5 = nn.ConvTranspose3d(16, 16,   kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.segmentation_head = nn.Conv3d(16, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, d, c, w, h = x.size()
        
        x_reshaped = x.view(b * d, c, w, h)
        features = self.backbone(x_reshaped) 
        
        x_3d = features.view(b, d, 512, features.size(2), features.size(3))
        x_3d = x_3d.permute(0, 2, 1, 3, 4)
        
        x_3d = self.down_conv(x_3d)
        
        u1 = F.relu(self.upsample1(x_3d))
        u2 = F.relu(self.upsample2(u1))
        u3 = F.relu(self.upsample3(u2))
        u4 = F.relu(self.upsample4(u3))
        u5 = F.relu(self.upsample5(u4))

        out = self.segmentation_head(u5)
        out = out.permute(0, 2, 1, 3, 4)
        
        return self.sigmoid(out)

