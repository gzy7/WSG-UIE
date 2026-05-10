# Wavelet and Structure Guided Lightweight Network for Underwater Image Enhancement

This paper is currently being submitted to The Visual Computer

## Environment

- OS: Ubuntu 22.04
- Python: 3.10
- CUDA: 11.8
- Miniconda3

## Create Environment

```bash
conda create -n WSG-UIE python=3.10 -y
conda activate WSG-UIE
```

## Dataset Preparation

Please download the datasets from the following links and place them in the `datasets/` directory.

- UIEB: [Google Drive](https://drive.google.com/file/d/1TZge0v5OzWWC8ZTxG7-NE7nVl0a4IeLb/view?usp=drive_link)
- LSUI: [Google Drive](YOUR_LINK)
- UFO: [Google Drive](YOUR_LINK)
- EUVP: [Google Drive](YOUR_LINK)

Example:

```text
datasets/
├── UIEB/
│   ├── train/
│   │   ├── low/
│   │   └── high/
│   └── test/
│       ├── low/
│       └── high/
├── LSUI/
│   ├── train/
│   │   ├── low/
│   │   └── high/
│   └── test/
│       ├── low/
│       └── high/
├── UFO/
│   ├── train/
│   │   ├── low/
│   │   └── high/
│   └── test/
│       ├── low/
│       └── high/
├── EUVP/
│   ├── train/
│   │   ├── low/
│   │   └── high/
│   └── test/
│       ├── low/
│       └── high/
├── C60/
├── U45/
├── RUIE/
└── Color-Check7/
```

