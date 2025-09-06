
# Soft-Threshold Attention Enhanced Transformer-UNet (STAE-UNet)

This repository contains the implementation of **STAE-UNet**, a 3D deep learning architecture designed for permeability prediction in multi-component carbonate rocks. The model introduces:

- Multi-Soft-Threshold Attention Mechanism  
- Transformer modules for long-range dependencies  
- 3D UNet backbone for spatial feature extraction  

## Installation
```bash
git clone https://github.com/YourName/STAE-UNet.git
cd STAE-UNet
pip install -r requirements.txt
```

## Usage
```bash
python train.py --config configs/config.yaml
```

## Dataset
- Input: Multi-channel 3D digital rock cores (256×256×256)
- Channels: Mineral components (calcite, dolomite, quartz, pores, etc.)
- Labels: Experimental permeability from core flooding

## Citation
If you use this code, please cite our paper:
```
@article{yourpaper2025,
  title={A 3D Permeability Prediction Model for Multi-Component Carbonate Rocks: Multi-Soft-Threshold Transformer-UNet},
  journal={SPE Journal},
  year={2025}
}
```
