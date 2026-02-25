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

def compute_dice_loss(pred, labels, eps=1e-8):
    pred_flat = pred.contiguous().view(-1)
    labels_flat = labels.contiguous().view(-1)
    
    intersection = (pred_flat * labels_flat).sum()
    union = pred_flat.sum() + labels_flat.sum()
    
    dice_score = (2. * intersection + eps) / (union + eps)
    return 1 - dice_score

resnet_model = torchvision.models.resnet18(pretrained=True)
resnet_backbone = nn.Sequential(*list(resnet_model.children())[:-2])

for param in resnet_backbone.parameters():
    param.requires_grad = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MedCNN(backbone=resnet_backbone)
model.to(device)
print(f"Model loaded on: {device}")

optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5
batch_size = 2 

print("\nStarting Training...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for i in range(0, len(ct_images), batch_size):
        inputs = ct_images[i : i + batch_size].to(device)
        labels = segmentation_masks[i : i + batch_size].to(device)
        
        optimizer.zero_grad()
        
        pred = model(inputs)
        loss = compute_dice_loss(pred, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    avg_loss = running_loss / (len(ct_images) / batch_size)
    print(f"Epoch {epoch+1}/{epochs} | Dice Loss: {avg_loss:.4f}")

print("Training Complete.")