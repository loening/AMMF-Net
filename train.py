import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_msssim
from torchvision.models import vgg16, VGG16_Weights
import torch.nn.functional as F

from fusion_dataset import FusionDataset
from fusion_model import FusionNet
from utils import compute_mse, compute_psnr, compute_ssim, compute_mi

"""
AMMF-Net Training Script
This script trains the AMMF-Net model for multimodal medical image fusion (CT/MRI).
Usage:
    1. Set ct_dir and mri_dir to your image folders.
    2. Run: python train.py
"""

# Configuration
ct_dir = '/path/to/CT/images'   # Path to CT images folder
mri_dir = '/path/to/MRI/images' # Path to MRI images folder
save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./model/ammf-net.pth"))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Data augmentation and preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# Split dataset into training and validation sets
dataset = FusionDataset(ct_dir, mri_dir, transform=transform)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

# Model, optimizer, loss function, scheduler
model = FusionNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# Perceptual loss network (VGG16)
vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval().to(device)
for param in vgg.parameters():
    param.requires_grad = False


def ssim_loss(pred, target):
    """SSIM loss for image similarity."""
    return 1 - pytorch_msssim.ssim(pred, target, data_range=1.0, size_average=True)

def perceptual_loss(pred, target):
    """Perceptual loss using VGG16 features."""
    if pred.shape[1] == 1:
        pred = pred.repeat(1, 3, 1, 1)
        target = target.repeat(1, 3, 1, 1)
    pred_vgg = vgg(pred)
    target_vgg = vgg(target)
    return F.l1_loss(pred_vgg, target_vgg)

# Loss weights
lambda_ct = 1.0
lambda_mri = 1.0
lambda_ssim = 1.0
lambda_perc = 0.01


def train():
    """Train and evaluate AMMF-Net model."""
    num_epochs = 40
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        for ct, mri in train_loader:
            ct, mri = ct.to(device), mri.to(device)
            out = model(ct, mri)
            out_norm = torch.clamp(out, 0, 1)
            ct_norm = torch.clamp(ct, 0, 1)
            mri_norm = torch.clamp(mri, 0, 1)

            # Multi-term loss
            loss_ct = criterion(out_norm, ct_norm)
            loss_mri = criterion(out_norm, mri_norm)
            loss_ssim = ssim_loss(out_norm, ct_norm) + ssim_loss(out_norm, mri_norm)
            loss_perc = perceptual_loss(out_norm, ct_norm) + perceptual_loss(out_norm, mri_norm)
            loss = lambda_ct * loss_ct + lambda_mri * loss_mri + lambda_ssim * loss_ssim + lambda_perc * loss_perc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_total = 0
        with torch.no_grad():
            for ct, mri in val_loader:
                ct, mri = ct.to(device), mri.to(device)
                out = torch.clamp(model(ct, mri), 0, 1)
                ct_norm = torch.clamp(ct, 0, 1)
                mri_norm = torch.clamp(mri, 0, 1)

                loss_ct = criterion(out, ct_norm)
                loss_mri = criterion(out, mri_norm)
                loss_ssim = ssim_loss(out, ct_norm) + ssim_loss(out, mri_norm)
                loss_perc = perceptual_loss(out, ct_norm) + perceptual_loss(out, mri_norm)
                loss = lambda_ct * loss_ct + lambda_mri * loss_mri + lambda_ssim * loss_ssim + lambda_perc * loss_perc

                val_total += loss.item()

        val_loss = val_total / len(val_loader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")

    # Plot and save loss curves
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curve.png')
    print("Train/Val loss curve saved to loss_curve.png")

    # Evaluation on validation set
    model.eval()
    with torch.no_grad():
        mses, psnrs, ssims, mis = [], [], [], []
        for ct, mri in val_loader:
            ct, mri = ct.to(device), mri.to(device)
            fused = model(ct, mri)
            ref = (ct + mri) / 2
            mses.append(compute_mse(fused, ref))
            psnrs.append(compute_psnr(fused, ref))
            ssims.append(compute_ssim(fused, ref))
            mis.append(compute_mi(fused, ref))
        print(f"Val MSE: {np.mean(mses):.4f}, PSNR: {np.mean(psnrs):.2f}, SSIM: {np.mean(ssims):.3f}, MI: {np.mean(mis):.3f}")

if __name__ == '__main__':
    train()