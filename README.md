# DecompositionForFusion

Training
----------
1. Download [COCO](https://github.com/cocodataset/cocoapi): https://cocodataset.org/
2. Put your training images into any floder and modify the `option/train/SelfTrained_SDataset.yaml' to retarget it.
3. Train DeFusion
    1. Training
    ```bash
    python selftrain.py --opt options/train/SelfTrained_SDataset.yaml
    ```


Testing code
----------

* [test.py](test.py)
 
