from torch.utils.data import Dataset
import os
import json
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import Compose, RandomResizedCrop, ToTensor
import torch.nn.functional as F
import cv2
from tqdm import tqdm
from PIL import Image


class TestDataset(Dataset):
    def __init__(self, valopt):
        super(TestDataset, self).__init__()
        path = valopt['dataroot']
        self.img_transform = ToTensor()
        part_imgs = os.path.join(os.path.expanduser(path), valopt['part_name'])

        def not_dir(x):
            return '_MDF' not in x and '.DS_Store' not in x and '.txt' not in x

        part_imgs = [os.path.join(part_imgs, os_dir) for os_dir in filter(not_dir, os.listdir(part_imgs))]

        up_imgs = []
        low_imgs = []
        gt_imgs = []
        for img_seq in tqdm(part_imgs):
            gt, up, low = sorted(os.listdir(img_seq))
            # print('gt: ', gt, 'up: ', up, 'low: ', low)
            up_imgs.append(os.path.join(img_seq, up))
            low_imgs.append(os.path.join(img_seq, low))
            gt_imgs.append(os.path.join(img_seq, gt))

        up_imgs.sort()
        low_imgs.sort()
        gt_imgs.sort()

        self.iget_imgs = {}
        for o_img, u_img, g_img in zip(up_imgs, low_imgs, gt_imgs):
            self.iget_imgs[o_img] = [o_img, u_img, g_img]

        self.iget_imgs = [(key, values) for key, values in self.iget_imgs.items()]
        self.iget_imgs = sorted(self.iget_imgs, key=lambda x: x[0])

    def __len__(self):
        return len(self.iget_imgs)

    def __getitem__(self, index):
        c_img, (up_img, low_img, gt_img) = self.iget_imgs[index]

        up_img = Image.open(up_img)
        low_img = Image.open(low_img)

        up_img = self.img_transform(up_img)
        low_img = self.img_transform(low_img)

        c_img = os.path.split(c_img)[-1].split('.')[0]
        return up_img, low_img, c_img



class TestMFFDataset(Dataset):
    def __init__(self, valopt):
        super(TestMFFDataset, self).__init__()
        path = valopt['dataroot']
        self.img_transform = ToTensor()
        part_imgs = os.path.join(os.path.expanduser(path), valopt['part_name'])

        def not_dir(x):
            return '_MDF' not in x and '.DS_Store' not in x and '.txt' not in x

        part_imgs = [os.path.join(part_imgs, os_dir) for os_dir in filter(not_dir, os.listdir(part_imgs))]

        up_imgs = []
        low_imgs = []
        gt_imgs = []
        for img_seq in tqdm(part_imgs):
            up, low = sorted(os.listdir(img_seq))
            up_imgs.append(os.path.join(img_seq, up))
            low_imgs.append(os.path.join(img_seq, low))

        up_imgs.sort()
        low_imgs.sort()

        self.iget_imgs = {}
        for o_img, u_img in zip(up_imgs, low_imgs):
            self.iget_imgs[o_img] = [o_img, u_img]

        self.iget_imgs = [(key, values) for key, values in self.iget_imgs.items()]
        self.iget_imgs = sorted(self.iget_imgs, key=lambda x: x[0])

    def __len__(self):
        return len(self.iget_imgs)

    def __getitem__(self, index):
        c_img, (up_img, low_img) = self.iget_imgs[index]

        up_img = Image.open(up_img)
        low_img = Image.open(low_img)

        up_img = self.img_transform(up_img)
        low_img = self.img_transform(low_img)


        c_img = os.path.split(c_img)[-1].split('.')[0]
        return up_img, low_img, c_img

