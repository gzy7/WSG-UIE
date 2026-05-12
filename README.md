# Wavelet and Structure Guided Lightweight Network for Underwater Image Enhancement

[![DOI](https://zenodo.org/badge/1211449073.svg)](https://doi.org/10.5281/zenodo.20142586)

This repository contains the source code, pretrained weights, dataset links, testing result links, and detailed training/testing instructions for our The Visual Computer submission.

## Abstract

Underwater image enhancement is critical for ocean exploration, underwater robotics, and marine vision systems. Wavelength-dependent absorption and scattering cause severe color casts, low contrast, and detail loss in underwater imaging. This work addresses degradation issues by developing a lightweight end-to-end framework unifying color prior correction, wavelet-domain modeling, and structure-guided modulation. The method uses adaptive color balance, multi-scale wavelet enhancement, and gradient-aware fusion to restore chromatic fidelity and fine details while suppressing noise. Here we show the approach achieves 24.14 dB PSNR and 0.917 SSIM on UIEB benchmark, outperforming competing methods with only 1.06M parameters. It also boosts performance in object detection and semantic segmentation, offering practical value for real-time underwater vision applications. Code is available at https://github.com/gzy7/WSG-UIE.

## Overview

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
pip install torch torchvision torchmetrics
pip install -r requirements.txt
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

```html
<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th rowspan="2">Venue</th>
<th colspan="2">Complexity</th>
<th colspan="2">UIEB</th>
<th colspan="2">LSUI</th>
<th colspan="2">UFO</th>
<th colspan="2">EUVP</th>
</tr>

<tr>
<th>Params↓</th>
<th>FLOPs↓</th>

<th>PSNR↑</th>
<th>SSIM↑</th>

<th>PSNR↑</th>
<th>SSIM↑</th>

<th>PSNR↑</th>
<th>SSIM↑</th>

<th>PSNR↑</th>
<th>SSIM↑</th>
</tr>
</thead>

<tbody>

<tr>
<td>FiveA+</td>
<td>BMVC'2023</td>
<td>0.009</td>
<td>8.33</td>
<td>22.39</td>
<td>0.907</td>
<td>23.95</td>
<td>0.851</td>
<td>25.91</td>
<td>0.853</td>
<td>24.79</td>
<td>0.820</td>
</tr>

<tr>
<td>CBLA</td>
<td>TIM'2024</td>
<td>--</td>
<td>--</td>
<td>16.11</td>
<td>0.677</td>
<td>13.87</td>
<td>0.567</td>
<td>12.87</td>
<td>0.521</td>
<td>12.96</td>
<td>0.523</td>
</tr>

<tr>
<td>WWPF</td>
<td>TCSVT'2024</td>
<td>--</td>
<td>--</td>
<td>18.31</td>
<td>0.793</td>
<td>17.47</td>
<td>0.700</td>
<td>15.53</td>
<td>0.624</td>
<td>16.09</td>
<td>0.642</td>
</tr>

<tr>
<td>CCMSRNet</td>
<td>TGRS'2024</td>
<td>21.13</td>
<td>43.60</td>
<td>21.70</td>
<td>0.884</td>
<td>25.30</td>
<td>0.865</td>
<td>26.45</td>
<td>0.861</td>
<td>25.67</td>
<td>0.834</td>
</tr>

<tr>
<td>SINET</td>
<td>ICASSP'2025</td>
<td>0.03</td>
<td>0.05</td>
<td>20.19</td>
<td>0.837</td>
<td>21.12</td>
<td>0.825</td>
<td>19.55</td>
<td>0.651</td>
<td>24.52</td>
<td>0.830</td>
</tr>

<tr>
<td>UIR-PolyKernel</td>
<td>ICASSP'2025</td>
<td>1.84</td>
<td>25.56</td>
<td>23.02</td>
<td>0.913</td>
<td>25.42</td>
<td>0.867</td>
<td>27.44</td>
<td>0.870</td>
<td>26.63</td>
<td>0.852</td>
</tr>

<tr>
<td>HUPE</td>
<td>IJCV'2025</td>
<td>--</td>
<td>--</td>
<td>22.12</td>
<td>0.850</td>
<td>20.38</td>
<td>0.801</td>
<td>18.68</td>
<td>0.757</td>
<td>18.62</td>
<td>0.745</td>
</tr>

<tr>
<td>SPMFormer</td>
<td>KBS'2025</td>
<td>1.33</td>
<td>12.58</td>
<td>23.33</td>
<td>0.911</td>
<td>27.58</td>
<td>0.890</td>
<td>27.93</td>
<td>0.877</td>
<td>27.31</td>
<td>0.859</td>
</tr>

<tr>
<td>UDNet</td>
<td>ESWA'2025</td>
<td>1.40</td>
<td>46.76</td>
<td>19.20</td>
<td>0.839</td>
<td>20.12</td>
<td>0.806</td>
<td>19.89</td>
<td>0.778</td>
<td>18.73</td>
<td>0.738</td>
</tr>

<tr>
<td><b>WSG-UIE (Ours)</b></td>
<td>--</td>
<td><b>1.06</b></td>
<td><b>8.32</b></td>
<td><b>24.14</b></td>
<td><b>0.917</b></td>
<td><b>26.80</b></td>
<td><b>0.870</b></td>
<td><b>27.85</b></td>
<td><b>0.871</b></td>
<td><b>26.34</b></td>
<td><b>0.843</b></td>
</tr>

</tbody>
</table>
```


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

## Citation
Cite our work if WSG-UIE is useful to your research.

```bash
@article{WSG-UIE,
  title={Wavelet and Structure Guided Lightweight Network for Underwater Image Enhancement},
  author={Guo, Zengyang and Cai, Zhidan and Yuan, Zhaozhu and Xia, Haoyao},
  journal={The Visual Computer},
  pages={xxx--xxx},
  year={2026},
  publisher={Springer}
}
```

## Acknowledgement

This work is based on the implementation of WWE-UIE (https://github.com/chingheng0808/WWE-UIE).
We gratefully acknowledge the authors for providing the codebase, which we use as the baseline for our experiments.

## Contact
If you have any questions, please contact the email 2020000011@mails.cust.edu.cn
