# RSDet

## Removal Then Selection: A Coarse-to-Fine Fusion Perspective for RGB-Infrared Object Detection

[IEEE T-ITS Paper](https://ieeexplore.ieee.org/document/11278552)

This repository provides an open-source implementation of **RSDet** for RGB-Infrared object detection. It is built on top of **[MMDetection](https://github.com/open-mmlab/mmdetection) 3.1.0** and extends the original detection pipeline to support paired visible-infrared inputs.

This open-source version is organized around the **15th core implementation** and aligns the method naming with the paper.

## Highlights

- Paper-aligned implementation of the **Removal Then Selection** framework
- Based on the `15th` core version used in the project
- Supports paired RGB-Infrared detection on datasets such as **FLIR**, **LLVIP**, and **KAIST**
- Keeps the original MMDetection training and evaluation workflow

## Method Overview

RSDet follows a coarse-to-fine fusion pipeline with four main stages:

1. **Frequency Removal Module**
   Removes redundant modality-specific noise in the frequency domain and preserves informative unique content.
2. **Shared Feature Generator**
   Builds modality-shared representations from the filtered RGB and infrared inputs.
3. **Coarse-to-Fine Selection Fusion**
   Generates exclusive features for each modality and performs dynamic selection between shared and exclusive representations.
4. **Two-Stage Detector**
   Feeds the fused features into Faster R-CNN for final object detection.

## Paper-Aligned Naming in This Repository

To make the open-source version easier to understand, the key modules are renamed to match the paper terminology:

- `RSDet_15th` -> `RSDet`
- `UniqueMaskGenerator4` -> `FrequencyRemovalModule`
- `CommonFeatureGenerator3` -> `SharedFeatureGenerator`
- `Conv11_Fusion4` -> `DynamicFeatureSelection`

Main implementation files:

- [`mmdet/models/detectors/rsdet.py`](./mmdet/models/detectors/rsdet.py)
- [`mmdet/models/custom/common_unique/rsdet_frequency_removal.py`](./mmdet/models/custom/common_unique/rsdet_frequency_removal.py)
- [`mmdet/models/custom/common_unique/rsdet_shared_feature_generator.py`](./mmdet/models/custom/common_unique/rsdet_shared_feature_generator.py)
- [`mmdet/models/custom/common_unique/rsdet_selection_fusion.py`](./mmdet/models/custom/common_unique/rsdet_selection_fusion.py)

## Getting Started

### Installation

Please first follow the official MMDetection installation guide:

[MMDetection Installation](https://mmdetection.readthedocs.io/en/latest/get_started.html)

Then clone this repository:

```bash
git clone https://github.com/Zhao-Tian-yi/RSDet.git
cd RSDet-opensource
```

Create and activate a conda environment:

```bash
conda create -n RSDet python=3.9 -y
conda activate RSDet
```

Install the package in editable mode:

```bash
pip install -v -e .
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements_rgbt.txt
```

## Configuration

The paper-aligned FLIR open-source configuration is provided at:

- [`configs/fusion/RSDet/faster_rcnn_r50_rsdet_FLIR.py`](./configs/fusion/RSDet/faster_rcnn_r50_rsdet_FLIR.py)

## Training

Single-GPU training:

```bash
python tools/train.py configs/fusion/RSDet/faster_rcnn_r50_rsdet_FLIR.py
```

Multi-GPU training:

```bash
./tools/dist_train.sh configs/fusion/RSDet/faster_rcnn_r50_rsdet_FLIR.py 2
```

Training commands for all supported datasets:

```bash
# FLIR
python tools/train.py configs/fusion/RSDet/faster_rcnn_r50_rsdet_FLIR.py

# KAIST
python tools/train.py configs/fusion/RSDet/faster_rcnn_r50_rsdet_KAIST.py

# LLVIP
python tools/train.py configs/fusion/RSDet/faster_rcnn_r50_rsdet_LLVIP.py

# M3FD
python tools/train.py configs/fusion/RSDet/faster_rcnn_r50_rsdet_M3FD.py

# MFAD
python tools/train.py configs/fusion/RSDet/faster_rcnn_r50_rsdet_MFAD.py
```

Multi-GPU examples:

```bash
# FLIR
./tools/dist_train.sh configs/fusion/RSDet/faster_rcnn_r50_rsdet_FLIR.py 2

# KAIST
./tools/dist_train.sh configs/fusion/RSDet/faster_rcnn_r50_rsdet_KAIST.py 2

# LLVIP
./tools/dist_train.sh configs/fusion/RSDet/faster_rcnn_r50_rsdet_LLVIP.py 2

# M3FD
./tools/dist_train.sh configs/fusion/RSDet/faster_rcnn_r50_rsdet_M3FD.py 2

# MFAD
./tools/dist_train.sh configs/fusion/RSDet/faster_rcnn_r50_rsdet_MFAD.py 2
```

## Testing

Single-GPU testing:

```bash
python tools/test.py configs/fusion/RSDet/faster_rcnn_r50_rsdet_FLIR.py /path/to/checkpoint.pth
```

Multi-GPU testing:

```bash
./tools/dist_test.sh configs/fusion/RSDet/faster_rcnn_r50_rsdet_FLIR.py /path/to/checkpoint.pth 2
```

Testing commands for all supported datasets:

```bash
# FLIR
python tools/test.py configs/fusion/RSDet/faster_rcnn_r50_rsdet_FLIR.py /path/to/checkpoint.pth

# KAIST
python tools/test.py configs/fusion/RSDet/faster_rcnn_r50_rsdet_KAIST.py /path/to/checkpoint.pth

# LLVIP
python tools/test.py configs/fusion/RSDet/faster_rcnn_r50_rsdet_LLVIP.py /path/to/checkpoint.pth

# M3FD
python tools/test.py configs/fusion/RSDet/faster_rcnn_r50_rsdet_M3FD.py /path/to/checkpoint.pth

# MFAD
python tools/test.py configs/fusion/RSDet/faster_rcnn_r50_rsdet_MFAD.py /path/to/checkpoint.pth
```

Multi-GPU test examples:

```bash
# FLIR
./tools/dist_test.sh configs/fusion/RSDet/faster_rcnn_r50_rsdet_FLIR.py /path/to/checkpoint.pth 2

# KAIST
./tools/dist_test.sh configs/fusion/RSDet/faster_rcnn_r50_rsdet_KAIST.py /path/to/checkpoint.pth 2

# LLVIP
./tools/dist_test.sh configs/fusion/RSDet/faster_rcnn_r50_rsdet_LLVIP.py /path/to/checkpoint.pth 2

# M3FD
./tools/dist_test.sh configs/fusion/RSDet/faster_rcnn_r50_rsdet_M3FD.py /path/to/checkpoint.pth 2

# MFAD
./tools/dist_test.sh configs/fusion/RSDet/faster_rcnn_r50_rsdet_MFAD.py /path/to/checkpoint.pth 2
```

## Datasets

This project is designed for paired RGB-Infrared detection datasets. The current codebase includes support for datasets such as:

- FLIR
- LLVIP
- KAIST

Please prepare dataset annotations and paired image paths according to the dataset configs under `configs/_base_/datasets/`.

## Notes

- This repository keeps the original project structure for compatibility with MMDetection.
- The open-source implementation uses the **15th** core version as the main RSDet path.
- In this open-source version, the frequency removal stage uses `topk = 320` to match the paper setting.

## Citation

If you find this project useful, please cite:

```bibtex
@ARTICLE{11278552,
  author={Zhao, Tianyi and Yuan, Maoxun and Jiang, Feng and Wang, Nan and Wei, Xingxing},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  title={Removal Then Selection: A Coarse-to-Fine Fusion Perspective for RGB-Infrared Object Detection},
  year={2025},
  doi={10.1109/TITS.2025.3638627}
}
```

## Updates

- `2026-04`: Released a paper-aligned open-source version based on the `15th` core implementation

## Acknowledgment

This project is built upon the excellent open-source framework:

- [MMDetection](https://github.com/open-mmlab/mmdetection)
