
# Soft-Threshold Attention Enhanced Transformer-UNet (STAE-UNet)

This repository provides a **reference implementation** of the **STAE-UNet** model, proposed in our paper:  

> *A 3D Permeability Prediction Model for Multi-Component Carbonate Rocks: Multi-Soft-Threshold Transformer-UNet*  
> SPE Journal, 2025  

The STAE-UNet is a 3D deep learning architecture designed for permeability prediction in **multi-component carbonate rocks**. It integrates:

- **Multi-Soft-Threshold Attention Mechanism**: adaptively filters weak activations and emphasizes strong pore–mineral interactions.  
- **Transformer Modules**: capture long-range dependencies between heterogeneous pore structures.  
- **3D UNet Backbone**: extracts local spatial features and restores hierarchical structure information.  

⚠️ **Note:**  
This repository provides a **simplified and illustrative code release**. Certain implementation details described in the paper (e.g., specialized preprocessing of QEMSCAN data, proprietary oilfield-specific workflows) have been **deliberately omitted or replaced with conventional placeholder methods** to ensure data security and comply with oilfield confidentiality requirements. As such, the code here should be considered **conceptual and not directly executable** without further adaptation.

---

## Installation

```bash
git clone https://github.com/YourName/STAE-UNet.git
cd STAE-UNet
pip install -r requirements.txt
```

---

## Usage (Conceptual Example)

```bash
python train.py --config configs/config.yaml
```

The scripts illustrate the **model structure, training loop, and evaluation metrics**.  
They are intended for **educational and reference purposes**, not for production use.

---

## Dataset

- **Input:** Multi-channel 3D digital rock cubes (nominal resolution: 256×256×256 voxels).  
- **Channels:** Mineral components (calcite, dolomite, quartz, pore space, etc.).  
- **Labels:** Experimental permeability values obtained via steady-state core flooding.  

⚠️ **Data availability:**  
Due to **data security restrictions and oilfield confidentiality requirements**, the dataset used in the paper cannot be publicly released.  
Interested researchers may **contact the corresponding author via email** to request access under appropriate agreements.

---

## Citation

If you find this work useful, please cite our paper:

```
@article{yourpaper2025,
  title={A 3D Permeability Prediction Model for Multi-Component Carbonate Rocks: Multi-Soft-Threshold Transformer-UNet},
  journal={SPE Journal},
  year={2025}
}
```

---

## Disclaimer

This repository serves as an **academic reference**.  
- The code provided here is **illustrative** and not guaranteed to reproduce the full experimental results reported in the paper.  
- Certain components (e.g., detailed preprocessing pipelines, parameter fine-tuning, and post-processing steps) have been **simplified or omitted** for confidentiality and clarity.  
