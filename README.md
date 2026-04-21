# RSDet

## Removal Then Selection: A Coarse-to-Fine Fusion Perspective for RGB-Infrared Object Detection

[IEEE T-ITS Paper](https://ieeexplore.ieee.org/document/11278552)

RSDet is an RGB-Infrared object detector built on top of [MMDetection](https://github.com/open-mmlab/mmdetection). This repository provides a paper-aligned open-source implementation centered on the final `15th` core version of the method.

## Highlights

- Paper-aligned `RSDet` implementation based on the final project version
- Coarse-to-fine pipeline with frequency-domain removal and dynamic feature selection
- Multi-dataset support for RGB-Infrared detection
- Compatible with the MMDetection training and evaluation workflow

## Method Overview

RSDet follows four main stages:

1. `FrequencyRemovalModule`
   Removes redundant information in the frequency domain and preserves informative modality-unique content.
2. `SharedFeatureGenerator`
   Builds shared RGB-Infrared representations from the filtered inputs.
3. `DynamicFeatureSelection`
   Produces modality-exclusive features and dynamically fuses them with shared features.
4. `Faster R-CNN Detector`
   Consumes the fused multi-scale features for final detection.

## Paper-Aligned Naming

The open-source implementation renames key modules to match the paper terminology:

- `RSDet_15th` -> `RSDet`
- `UniqueMaskGenerator4` -> `FrequencyRemovalModule`
- `CommonFeatureGenerator3` -> `SharedFeatureGenerator`
- `Conv11_Fusion4` -> `DynamicFeatureSelection`

Core implementation files:

- [`mmdet/models/detectors/rsdet.py`](./mmdet/models/detectors/rsdet.py)
- [`mmdet/models/custom/common_unique/rsdet_frequency_removal.py`](./mmdet/models/custom/common_unique/rsdet_frequency_removal.py)
- [`mmdet/models/custom/common_unique/rsdet_shared_feature_generator.py`](./mmdet/models/custom/common_unique/rsdet_shared_feature_generator.py)
- [`mmdet/models/custom/common_unique/rsdet_selection_fusion.py`](./mmdet/models/custom/common_unique/rsdet_selection_fusion.py)

## Installation

Please first install MMDetection and its dependencies by following the official guide:

[MMDetection Installation Guide](https://mmdetection.readthedocs.io/en/latest/get_started.html)

Then clone this repository and install it in editable mode:

```bash
git clone https://github.com/Zhao-Tian-yi/RSDet.git
cd RSDet

conda create -n rsdet python=3.9 -y
conda activate rsdet

pip install -v -e .
pip install -r requirements.txt
pip install -r requirements_rgbt.txt
```

## Data Preparation

This repository uses relative dataset paths by default. Place datasets under `data/` or update the corresponding `data_root` in the dataset config files under [`configs/_base_/datasets`](./configs/_base_/datasets).

Suggested directory layout:

```text
RSDet/
├── data/
│   ├── FLIR_align/
│   ├── LLVIP/
│   ├── M3FD/
│   ├── MFAD/
│   └── KAIST/
└── pretrain/
    └── resnet50_cityscape.pth
```

Current dataset base configs:

- [`configs/_base_/datasets/FLIR.py`](./configs/_base_/datasets/FLIR.py)
- [`configs/_base_/datasets/LLVIP.py`](./configs/_base_/datasets/LLVIP.py)
- [`configs/_base_/datasets/M3FD.py`](./configs/_base_/datasets/M3FD.py)
- [`configs/_base_/datasets/MFAD.py`](./configs/_base_/datasets/MFAD.py)
- [`configs/_base_/datasets/KAIST.py`](./configs/_base_/datasets/KAIST.py)

## Pretrained Backbone

The released RSDet configs expect the backbone checkpoint at:

```text
pretrain/resnet50_cityscape.pth
```

If you want to use a different pretrained checkpoint, update the `pretrained_backbone` variable in the corresponding config file.

## Available RSDet Configs

This branch currently includes the following paper-aligned RSDet configs:

- [`configs/fusion/RSDet/faster_rcnn_r50_rsdet_FLIR.py`](./configs/fusion/RSDet/faster_rcnn_r50_rsdet_FLIR.py)
- [`configs/fusion/RSDet/faster_rcnn_r50_rsdet_KAIST.py`](./configs/fusion/RSDet/faster_rcnn_r50_rsdet_KAIST.py)
- [`configs/fusion/RSDet/faster_rcnn_r50_rsdet_LLVIP.py`](./configs/fusion/RSDet/faster_rcnn_r50_rsdet_LLVIP.py)
- [`configs/fusion/RSDet/faster_rcnn_r50_rsdet_M3FD.py`](./configs/fusion/RSDet/faster_rcnn_r50_rsdet_M3FD.py)
- [`configs/fusion/RSDet/faster_rcnn_r50_rsdet_M3FD_classaware.py`](./configs/fusion/RSDet/faster_rcnn_r50_rsdet_M3FD_classaware.py)
- [`configs/fusion/RSDet/faster_rcnn_r50_rsdet_MFAD.py`](./configs/fusion/RSDet/faster_rcnn_r50_rsdet_MFAD.py)
- [`configs/fusion/RSDet/faster_rcnn_r50_rsdet_MFAD_classaware.py`](./configs/fusion/RSDet/faster_rcnn_r50_rsdet_MFAD_classaware.py)

## Training

Single-GPU examples:

```bash
python tools/train.py configs/fusion/RSDet/faster_rcnn_r50_rsdet_FLIR.py
python tools/train.py configs/fusion/RSDet/faster_rcnn_r50_rsdet_KAIST.py
python tools/train.py configs/fusion/RSDet/faster_rcnn_r50_rsdet_LLVIP.py
python tools/train.py configs/fusion/RSDet/faster_rcnn_r50_rsdet_M3FD.py
python tools/train.py configs/fusion/RSDet/faster_rcnn_r50_rsdet_MFAD.py
```

Multi-GPU examples:

```bash
./tools/dist_train.sh configs/fusion/RSDet/faster_rcnn_r50_rsdet_FLIR.py 4
./tools/dist_train.sh configs/fusion/RSDet/faster_rcnn_r50_rsdet_KAIST.py 4
./tools/dist_train.sh configs/fusion/RSDet/faster_rcnn_r50_rsdet_LLVIP.py 4
./tools/dist_train.sh configs/fusion/RSDet/faster_rcnn_r50_rsdet_M3FD.py 4
./tools/dist_train.sh configs/fusion/RSDet/faster_rcnn_r50_rsdet_MFAD.py 4
```

Class-aware sampler variants:

```bash
python tools/train.py configs/fusion/RSDet/faster_rcnn_r50_rsdet_M3FD_classaware.py
python tools/train.py configs/fusion/RSDet/faster_rcnn_r50_rsdet_MFAD_classaware.py
```

## Testing

Single-GPU examples:

```bash
python tools/test.py configs/fusion/RSDet/faster_rcnn_r50_rsdet_FLIR.py /path/to/checkpoint.pth
python tools/test.py configs/fusion/RSDet/faster_rcnn_r50_rsdet_KAIST.py /path/to/checkpoint.pth
python tools/test.py configs/fusion/RSDet/faster_rcnn_r50_rsdet_LLVIP.py /path/to/checkpoint.pth
python tools/test.py configs/fusion/RSDet/faster_rcnn_r50_rsdet_M3FD.py /path/to/checkpoint.pth
python tools/test.py configs/fusion/RSDet/faster_rcnn_r50_rsdet_MFAD.py /path/to/checkpoint.pth
```

Multi-GPU examples:

```bash
./tools/dist_test.sh configs/fusion/RSDet/faster_rcnn_r50_rsdet_FLIR.py /path/to/checkpoint.pth 4
./tools/dist_test.sh configs/fusion/RSDet/faster_rcnn_r50_rsdet_KAIST.py /path/to/checkpoint.pth 4
./tools/dist_test.sh configs/fusion/RSDet/faster_rcnn_r50_rsdet_LLVIP.py /path/to/checkpoint.pth 4
./tools/dist_test.sh configs/fusion/RSDet/faster_rcnn_r50_rsdet_M3FD.py /path/to/checkpoint.pth 4
./tools/dist_test.sh configs/fusion/RSDet/faster_rcnn_r50_rsdet_MFAD.py /path/to/checkpoint.pth 4
```

## Notes

- This open-source branch keeps the MMDetection project structure for compatibility.
- The main RSDet release path is aligned with the final `15th` implementation.
- The frequency removal stage uses `topk = 320` to match the paper setting.
- Several legacy experimental modules are retained for reference, but the configs listed above are the recommended starting points.

## Dataset Layout

The repository assumes that each dataset contains paired visible and infrared images, together with COCO-style annotation files.

Recommended directory layout:

```text
RSDet/
├── data/
│   ├── FLIR_align/
│   │   ├── train/
│   │   ├── test/
│   │   ├── Annotation_train.json
│   │   └── Annotation_test.json
│   ├── LLVIP/
│   │   ├── train/
│   │   ├── test/
│   │   ├── Annotation_train.json
│   │   └── Annotation_test.json
│   ├── M3FD/
│   │   ├── train/
│   │   ├── test/
│   │   ├── Annotation_train.json
│   │   └── Annotation_test.json
│   ├── MFAD/
│   │   ├── train/
│   │   ├── test/
│   │   ├── Annotation_train.json
│   │   └── Annotation_test.json
│   └── KAIST/
│       ├── kaist_train/
│       ├── kaist_test/
│       ├── kaist_train_data.json
│       └── kaist_test_data.json
└── pretrain/
    └── resnet50_cityscape.pth
```

For paired loading, the annotation file should store both visible and infrared image paths in the format expected by `MultispectralDataset` and `LoadPairedImageFromFile`.

## Paths To Modify

If your local directory layout is different, update the following paths before training:

- Dataset root:
  Modify `data_root` in the dataset base config you use, for example:
  - [`configs/_base_/datasets/FLIR.py`](./configs/_base_/datasets/FLIR.py)
  - [`configs/_base_/datasets/KAIST.py`](./configs/_base_/datasets/KAIST.py)
  - [`configs/_base_/datasets/LLVIP.py`](./configs/_base_/datasets/LLVIP.py)
  - [`configs/_base_/datasets/M3FD.py`](./configs/_base_/datasets/M3FD.py)
  - [`configs/_base_/datasets/MFAD.py`](./configs/_base_/datasets/MFAD.py)

- Annotation filenames and image subfolders:
  If your dataset does not use the default names, modify `ann_file` and `data_prefix` in the same dataset config files.

- Pretrained backbone checkpoint:
  Modify `pretrained_backbone` in the RSDet config you use under [`configs/fusion/RSDet`](./configs/fusion/RSDet).
  The default value is `pretrain/resnet50_cityscape.pth`.

- Optional debug or analysis scripts:
  If you use files under [`utils`](./utils), pass your own input paths or update the script arguments accordingly.

In most cases, preparing `data/...` and `pretrain/...` with the default names is the easiest option, because then no additional config edits are needed.

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

## Acknowledgment

RSDet is built upon the excellent open-source framework:

- [MMDetection](https://github.com/open-mmlab/mmdetection)
