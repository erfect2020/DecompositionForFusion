# DecompositionForFusion

Training
----------
1. Download [COCO](https://github.com/cocodataset/cocoapi): https://cocodataset.org/
2. Put your training images into any floder and modify the `option/train/SelfTrained_SDataset.yaml' to retarget the path.
3. Train DeFusion
    1. Training
    ```bash
    python selftrain.py --opt options/train/SelfTrained_SDataset.yaml
    ```


Testing code
----------

1. Download test dataset [MEFB](https://github.com/xingchenzhang/MEFB):https://github.com/xingchenzhang/MEFB [TNO](https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029):https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029
2. Modify [test.py](test.py) to select the data preprocess files for different tasks: 
3. Test DeFusion
    1. Test multi-exposure image fusion
    ```bash
    python test.py --opt options/test/MEF_Test_Dataset.yaml
    ```
    2. Test multi-focus image fusion
    ```bash
    python test.py --opt options/test/MFF_Test_Dataset.yaml
    ```
    3. Test visible infrared image fusion
    ```bash
    python test.py --opt options/test/IVF_Test_Dataset.yaml
    ```
