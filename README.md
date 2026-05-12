# Wavelet and Structure Guided Lightweight Network for Underwater Image Enhancement

This paper is currently being submitted to The Visual Computer

![WSG-UIE](Figs/WSG.png)

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

Paired Datasets:

- UIEB: [Google Drive](https://drive.google.com/file/d/1TZge0v5OzWWC8ZTxG7-NE7nVl0a4IeLb/view?usp=drive_link)
- LSUI: [Google Drive](https://drive.google.com/file/d/159oI-U4y1tB5XjohQF5Yh87Mivq7IXv0/view?usp=drive_link)
- UFO: [Google Drive](https://drive.google.com/file/d/1Ij-fOthwBzxV9kScDSvmImeDcWQ7zJqE/view?usp=drive_link)
- EUVP: [Google Drive](https://drive.google.com/file/d/1LaUCWq__csU7-u8zdd7gfmlpWGUmSZAD/view?usp=drive_link)
```text
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
└── EUVP/
    ├── train/
    │   ├── low/
    │   └── high/
    └── test/
        ├── low/
        └── high/
```

Unpaired Datasets:

- C60: [Google Drive](https://drive.google.com/file/d/1wencN8I9-VSDbZL0U5jgJyAIbUbtrgDX/view?usp=drive_link)
- U45: [Google Drive](https://drive.google.com/file/d/1iigjP9nbQvE3JeVRsKBPgkda53mG6vtT/view?usp=drive_link)
- RUIE: [Google Drive](https://drive.google.com/file/d/1U_RbNaaSjkrIxz-r-lvzKXUnMaLwgRTM/view?usp=drive_link)
- Color-Check7: [Google Drive](https://drive.google.com/file/d/17BP9LbK6Qq24fsIgAsoZampzj0B80Rhu/view?usp=drive_link)
```text
├── C60/
├── U45/
├── RUIE/
└── Color-Check7/
```

## Results

We provide all enhanced results [Google Drive](https://drive.google.com/file/d/1EqSA8Rgn1birkYwG25gPsi9MKBSHAZRN/view?usp=drive_link) for easy comparison and visualization.

## Training

```bash
python train.py
```

## Testing (paired datasets)

```bash
python test.py --ckpt "your checkpoint" --dataset 'your dataset' --test_batch_size 'your size'
```

## Testing (unpaired datasets)
When testing unpaired datasets, please select the second annotated code in the metrics.py to run.
```bash
python test_unpaired.py --ckpt "your checkpoint" --dataset 'your dataset'
```

## Acknowledgement

This work is based on the implementation of WWE-UIE (https://github.com/chingheng0808/WWE-UIE).
We gratefully acknowledge the authors for providing the codebase, which we use as the baseline for our experiments.
