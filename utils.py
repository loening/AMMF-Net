import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compute_mse(img1, img2):
    """Compute Mean Squared Error between two images."""
    return ((img1 - img2) ** 2).mean()

def compute_psnr(img1, img2, max_val=1.0):
    """Compute Peak Signal-to-Noise Ratio between two images."""
    mse = compute_mse(img1, img2)
    if mse < 1e-8:
        return float('inf')
    return 10 * np.log10((max_val ** 2) / mse)

def compute_ssim(img1, img2):
    """Compute Structural Similarity Index between two images."""
    img1 = img1.squeeze().cpu().numpy()
    img2 = img2.squeeze().cpu().numpy()
    return ssim(img1, img2, data_range=img2.max() - img2.min())

def compute_mi(img1, img2, bins=64):
    """Compute Mutual Information between two images."""
    img1 = (img1.squeeze().cpu().numpy() * 255).astype(np.uint8)
    img2 = (img2.squeeze().cpu().numpy() * 255).astype(np.uint8)
    hist_2d, _, _ = np.histogram2d(img1.ravel(), img2.ravel(), bins=bins)
    pxy = hist_2d / float(hist_2d.sum())
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    nzs = pxy > 0
    x_idx, y_idx = np.where(nzs)
    mi = np.sum(pxy[x_idx, y_idx] * np.log(pxy[x_idx, y_idx] / (px[x_idx] * py[y_idx] + 1e-12)))
    return mi