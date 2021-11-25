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
        inf_imgs = os.path.join(os.path.expanduser(path), valopt['infrare_name'])
        vis_imgs = os.path.join(os.path.expanduser(path), valopt['visible_name'])

        def not_dir(x):
            return '_MDF' not in x and '.DS_Store' not in x and '.txt' not in x

        inf_imgs = [os.path.join(inf_imgs, os_dir) for os_dir in filter(not_dir, os.listdir(inf_imgs))]
        vis_imgs = [os.path.join(vis_imgs, os_dir) for os_dir in filter(not_dir, os.listdir(vis_imgs))]

        inf_imgs.sort()
        vis_imgs.sort()

        self.iget_imgs = {}
        for i_img, v_img in zip(inf_imgs, vis_imgs):
            self.iget_imgs[v_img] = [i_img, v_img]

        self.iget_imgs = [(key, values) for key, values in self.iget_imgs.items()]
        self.iget_imgs = sorted(self.iget_imgs, key=lambda x: x[0])

    def __len__(self):
        return len(self.iget_imgs)

    def __getitem__(self, index):
        c_img, (up_img, low_img) = self.iget_imgs[index]

        # print('infrare image', up_img)
        # print('visible image', low_img)

        up_img = Image.open(up_img).convert('L').convert('RGB')
        low_img = Image.open(low_img).convert('L').convert('RGB')

        up_img = self.img_transform(up_img)
        low_img = self.img_transform(low_img)
        # u_img = torch.rand_like(u_img)

        # o_img = cv2.imread(o_img, 1)
        # u_img = cv2.imread(u_img, 1)
        #
        # o_img = torch.tensor(o_img / 65535.).float().permute(2, 0, 1)
        # u_img = torch.tensor(u_img / 65535.).float().permute(2, 0, 1)

        c_img = os.path.split(c_img)[-1].split('.')[0]
        return up_img, low_img, c_img


class TestTNODataset(Dataset):
    def __init__(self, valopt):
        super(TestTNODataset, self).__init__()
        path = valopt['dataroot']
        self.img_transform = ToTensor()
        part_imgs = os.path.expanduser(path)

        def not_dir(x):
            return '_MDF' not in x and '.DS_Store' not in x and '.txt' not in x

        part_imgs = [os.path.join(part_imgs, os_dir) for os_dir in filter(not_dir, os.listdir(part_imgs))]

        ir_imgs = []
        vi_imgs = []
        gt_imgs = []
        for img_seq in tqdm(part_imgs):
            up, low = sorted(os.listdir(img_seq))
            # print('gt: ', gt, 'up: ', up, 'low: ', low)
            if 'ir' in up.lower():
                ir_imgs.append(os.path.join(img_seq, up))
                vi_imgs.append(os.path.join(img_seq, low))
            else:
                vi_imgs.append(os.path.join(img_seq, up))
                ir_imgs.append(os.path.join(img_seq, low))
                            # gt_imgs.append(os.path.join(img_seq, gt))

        ir_imgs.sort()
        vi_imgs.sort()
        # gt_imgs.sort()

        self.iget_imgs = {}
        for o_img, u_img in zip(ir_imgs, vi_imgs):
            self.iget_imgs[o_img] = [o_img, u_img]

        self.iget_imgs = [(key, values) for key, values in self.iget_imgs.items()]
        self.iget_imgs = sorted(self.iget_imgs, key=lambda x: x[0])

    def __len__(self):
        return len(self.iget_imgs)

    def __getitem__(self, index):
        c_img, (up_img, low_img) = self.iget_imgs[index]

        up_img = Image.open(up_img).convert('RGB')
        low_img = Image.open(low_img).convert("RGB")

        up_img = self.img_transform(up_img)
        low_img = self.img_transform(low_img)
        # u_img = torch.rand_like(u_img)


        # o_img = cv2.imread(o_img, 1)
        # u_img = cv2.imread(u_img, 1)
        #
        # o_img = torch.tensor(o_img / 65535.).float().permute(2, 0, 1)
        # u_img = torch.tensor(u_img / 65535.).float().permute(2, 0, 1)

        c_img = os.path.split(os.path.split(c_img)[-2])[-1]
        return up_img, low_img, c_img

