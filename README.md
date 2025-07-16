# AMMF-Net: Multimodal Medical Image Fusion

## Overview
AMMF-Net is a deep learning model for fusing multimodal medical images (e.g., CT and MRI) using an encoder-decoder architecture with channel attention. This repository provides the core code for dataset loading, model definition, training, and evaluation.

## Structure
- `fusion_dataset.py`: Dataset loader for paired CT/MRI images.
- `fusion_model.py`: AMMF-Net model definition (PyTorch).
- `train.py`: Training and evaluation script.
- `utils.py`: Image quality metrics (MSE, PSNR, SSIM, MI).

## Usage
### Environment & Dependencies
Recommended Python version: >=3.8
Required libraries:
```
torch>=1.12.0
torchvision>=0.13.0
numpy
matplotlib
Pillow
scikit-image
pytorch-msssim
```
You can install all dependencies via:
```bash
pip install torch torchvision numpy matplotlib Pillow scikit-image pytorch-msssim
```

### Steps
1. Prepare paired CT and MRI images in separate folders.
2. Edit `ct_dir` and `mri_dir` in `train.py` to your data paths.
3. Run training:
   ```bash
   python train.py
   ```
4. The trained model and loss curve will be saved automatically.

## Citation
If you use this code for research, please cite our paper:
```
@article{AMMFNet2025,
  title={AMMF-Net: An End-to-End Adaptive Multimodal Medical Image Fusion Method},
  author={Dening Luo, Xinran Liu, Xiaozhao Jin},
  journal={The 8th International Conference on Biological Information and Biomedical Engineering (BIBE 2025)},
  year={2025}
}
```

## License
MIT
