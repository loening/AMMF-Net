import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class FusionDataset(Dataset):
    def __init__(self, ct_dir, mri_dir, transform=None):
        # Load and sort CT and MRI image paths
        self.ct_imgs = sorted([os.path.join(ct_dir, f) for f in os.listdir(ct_dir) if f.lower().endswith('.png')])
        self.mri_imgs = sorted([os.path.join(mri_dir, f) for f in os.listdir(mri_dir) if f.lower().endswith('.png')])
        assert len(self.ct_imgs) == len(self.mri_imgs), "The number of CT and MRI images must be equal."
        self.transform = transform

    def __len__(self):
        # Return the number of image pairs
        return len(self.ct_imgs)

    def __getitem__(self, idx):
        # Load CT and MRI images as grayscale
        ct = Image.open(self.ct_imgs[idx]).convert('L')
        mri = Image.open(self.mri_imgs[idx]).convert('L')
        if self.transform:
            ct = self.transform(ct)
            mri = self.transform(mri)
        return ct, mri