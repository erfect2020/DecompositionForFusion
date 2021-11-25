from torch.utils.data import Dataset
import os
import json
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import Compose, RandomResizedCrop, ToTensor
import cv2
from tqdm import tqdm
from PIL import Image
import random

class TestDataset(Dataset):
    def __init__(self, valopt):
        super(TestDataset, self).__init__()
        path = valopt['dataroot']
        self.img_transform = ToTensor()
        part_imgs = os.path.join(os.path.expanduser(path), valopt['part_name'])

        def not_dir(x):
            return '_MDF' not in x and '.DS_Store' not in x and '.txt' not in x

        part_imgs = [os.path.join(part_imgs, os_dir) for os_dir in filter(not_dir, os.listdir(part_imgs))]

        over_imgs = []
        under_imgs = []
        for img_seq in tqdm(part_imgs):
            ov_seq = {}
            for img_name in filter(not_dir, os.listdir(img_seq)):
                img = cv2.imread(os.path.join(img_seq, img_name), 1)
                ov_seq[img_name] = img.mean()
            over_imgs.append(os.path.join(img_seq, max(ov_seq, key=ov_seq.get)))
            under_imgs.append(os.path.join(img_seq, min(ov_seq, key=ov_seq.get)))

        over_imgs.sort()
        under_imgs.sort()

        self.iget_imgs = {}
        for o_img, u_img in zip(over_imgs, under_imgs):
            self.iget_imgs[o_img] = [o_img, u_img]

        self.iget_imgs = [(key, values) for key, values in self.iget_imgs.items()]
        self.iget_imgs = sorted(self.iget_imgs, key=lambda x: x[0])

    def __len__(self):
        return len(self.iget_imgs)

    def __getitem__(self, index):
        c_img, (o_img, u_img) = self.iget_imgs[index]

        o_img = Image.open(o_img)
        u_img = Image.open(u_img)

        o_img = self.img_transform(o_img)
        u_img = self.img_transform(u_img)


        c_img = os.path.split(os.path.split(c_img)[-2])[-1]
        return o_img, u_img, c_img

