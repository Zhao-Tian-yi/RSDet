
**Noted:  The paper is under review, and this code repository will be updated after it is accepted**

<div align="center">
<h1>RSDet </h1>
<h3>Removal and Selection: Improving RGB Infrared Object Detection via Coarse to Fine Fusion</h3>
  
Paper: ([arXiv 2401.10731](https://arxiv.org/abs/2401.10731))

</div>

Based on the **[MMdetection](https://github.com/open-mmlab/mmdetection) 3.1.0** framework, this project modifies its data flow and related classes and functions, and changes the MMdetection to a multi-modal detection framework to facilitate **RGBT Object Detection**.

## **Overview**

<p align="center">
  <img src="README.assets\image-20240312011746031.png" alt="overview" width="90%">
</p>

## **Main Results**

**`Dataset: FLIR, LLVIP and Kaist`**


## **Getting Started**

### Installation

ref : [mmdetection installation](https://mmdetection.readthedocs.io/en/latest/get_started.html)

**Step 1: Clone the RSDet repository:**

To get started, first clone the RSDet repository and navigate to the project directory:

```bash
git clone https://github.com/Zhao-Tian-yi/RSDet.git
cd RSDet
```

**Step 2: Environment Setup:**

RSDet recommends setting up a conda environment and installing dependencies via pip. Use the following commands to set up your environment:

***Create and activate a new conda environment***

```bash
conda create -n RSDet
conda activate RSDet
```

***If you develop and run mmdet directly, install it from source***

```
pip install -v -e .
```

***Install Dependencies***

```bash
pip install -r requirements.txt
pip install -r requirements_rgbt.txt
```

## **Result**

[Kaist Result](https://drive.google.com/file/d/11tiHFCRG8ubt23g-BN1wY3W94pYNimL0/view?usp=sharing)



## **Citation**

```
@article{zhao2024removal,
  title={Removal and Selection: Improving RGB-Infrared Object Detection via Coarse-to-Fine Fusion},
  author={Zhao, Tianyi and Yuan, Maoxun and Wei, Xingxing},
  journal={arXiv preprint arXiv:2401.10731},
  year={2024}
}
```
## :white_check_mark: Updates
* **` March. 19th, 2024`**: Update: we have updated the source about one version bug. 

* **` March. 12th, 2024`:** The source code is provided. 
## **Acknowledgment**
