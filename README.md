# Fusion from Decomposition

## Prerequisites
- Linux
- Python 3
- NVIDIA GPU + CUDA cuDNN
- PyTorch 1.9


## Getting Started
### Installation
- Install python libraries and requests.

- Clone this repo:
```bash
git clone https://github.com/xxxxx/XXXXXXX
cd DecompositionForFusion
```




## Start run
----------
1. Download [COCO](https://github.com/cocodataset/cocoapi): https://cocodataset.org/
2. Put your training images into any floder and modify the `option/train/SelfTrained_SDataset.yaml' to retarget the path.
3. Train DeFusion
    ```bash
    python selftrain.py --opt options/train/SelfTrained_SDataset.yaml
    ```


## Start evaluation
----------

1. Download test dataset:
    1. Multi-exposure image fusion: [MEFB](https://github.com/xingchenzhang/MEFB):https://github.com/xingchenzhang/MEFB, [SICE](https://github.com/csjcai/SICE):https://github.com/csjcai/SICE.
    2. Multi-focus image fusion: [Real-MFF](https://githubmemory.com/repo/Zancelot/Real-MFF):https://githubmemory.com/repo/Zancelot/Real-MFF, [Dataset](https://github.com/xingchenzhang/MFIFB):https://github.com/xingchenzhang/MFIFB.
    3. Visible-infrared image fusion: [RoadScene](https://github.com/jiayi-ma/RoadScene):https://github.com/jiayi-ma/RoadScene, [TNO](https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029):https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029.
3. Modify [test.py](test.py) to select the data preprocess files for different tasks: 
4. (Option) Our pretrained model is avaliable at [Google Drive](https://drive.google.com/file/d/1CUoFLiV3mugvbfBcMcwgXbDF6bWPhdd9/view?usp=sharing)
5. Test DeFusion
    1. Test multi-exposure image fusion task on [MEFB](https://github.com/xingchenzhang/MEFB) or [SICE](https://github.com/csjcai/SICE)
    ```bash
    python test.py --opt options/test/MEF_Test_Dataset.yaml or
    python test.py --opt options/test/SMEF_Test_Dataset.yaml
    ```
    2. Test multi-focus image fusion task on [Real-MFF](https://githubmemory.com/repo/Zancelot/Real-MFF) or [Dataset](https://github.com/xingchenzhang/MFIFB)
    ```bash
    python test.py --opt options/test/MFF_Test_Dataset.yaml or
    python test.py --opt options/test/EMFF_Test_Dataset.yaml
    ```
    3. Test visible infrared image fusion task on [RoadScene](https://github.com/jiayi-ma/RoadScene) or [TNO](https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029)
    ```bash
    python test.py --opt options/test/IVF_Test_Dataset.yaml or
    python test.py --opt options/test/TIVF_Test_Dataset.yaml
    ```
    ```
