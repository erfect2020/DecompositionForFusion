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


class TrainDataset(Dataset):
    def __init__(self, trainopt):
        super(TrainDataset, self).__init__()
        path = trainopt['dataroot']
        self.img_transform = ToTensor()
        self.img_resize = RandomResizedCrop(224)
        self.image_size = trainopt['image_size']
        self.batch_size = trainopt['iter_size']
        self.max_iter = trainopt["max_iter"]
        self.epoch_num = 0
        self.same_radio = 0.2
        self.noise_radio = 0.2
        self.train_type = ['supervised', 'self_supervised_common',
                           'self_supervised_upper', 'self_supervised_lower']

        trainpairs_forreading = str(trainopt['trainpairs'])
        if not os.path.exists(trainpairs_forreading):
            part1_imgs = os.path.join(os.path.expanduser(path), trainopt["part1_name"])
            part2_imgs = os.path.join(os.path.expanduser(path), trainopt["part2_name"])

            def not_dir(x):
                return 'Label' not in x and '.json' not in x and '.DS_Store' not in x
            part1_imgs = [os.path.join(part1_imgs, os_dir) for os_dir in filter(not_dir, os.listdir(part1_imgs)) ]
            part2_imgs = [os.path.join(part2_imgs, os_dir) for os_dir in filter(not_dir, os.listdir(part2_imgs)) ]

            over_imgs = []
            under_imgs = []
            for img_seq in tqdm(part1_imgs + part2_imgs):
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
                rand_state = torch.rand(2, 1)
                self.iget_imgs[str(o_img)+':'+self.train_type[0]] = [o_img, u_img, self.train_type[0]]
                if rand_state[0] > 1 - self.same_radio:
                    if rand_state[1] > 0.5:
                        self.iget_imgs[str(o_img)+':'+self.train_type[1]] = [o_img, o_img, self.train_type[1]]
                    else:
                        self.iget_imgs[str(o_img)+':'+self.train_type[1]] = [u_img, u_img, self.train_type[1]]
                elif rand_state[0] < self.noise_radio:
                    if rand_state[1] > 0.5:
                        self.iget_imgs[str(o_img)+':'+self.train_type[2]] = [o_img, o_img, self.train_type[2]]
                    else:
                        self.iget_imgs[str(o_img)+':'+self.train_type[3]] = [u_img, u_img, self.train_type[3]]

            extra_imgs = os.path.expanduser(trainopt['extra_name'])
            extra_imgs = [os.path.join(extra_imgs, os_dir) for os_dir in os.listdir(extra_imgs)]
            extra_imgs = random.choices(extra_imgs, k=3000)

            for self_img in extra_imgs:
                rand_state= torch.rand(1)
                if rand_state < 0.5:
                    self.iget_imgs[str(self_img) + ':' + self.train_type[1]] = [self_img, self_img, self.train_type[1]]
                elif rand_state < 0.75:
                    self.iget_imgs[str(self_img) + ':' + self.train_type[2]] = [self_img, self_img, self.train_type[2]]
                else:
                    self.iget_imgs[str(self_img) + ':' + self.train_type[3]] = [self_img, self_img, self.train_type[3]]

            with open(trainpairs_forreading, 'w') as f:
                json.dump(self.iget_imgs, f)
        else:
            with open(trainpairs_forreading, 'r') as f:
                self.iget_imgs = json.load(f)
        self.iget_imgs = [(key, values) for key, values in self.iget_imgs.items()]
        self.iget_imgs = sorted(self.iget_imgs, key=lambda x: x[0])
        self.iget_imgs = list(filter(lambda x: self.train_type[0] != x[-1][-1] , self.iget_imgs))

    def __len__(self):
        return len(self.iget_imgs)

    def random_augmentation(self, img):
        c, w, h = img.shape
        w_start = w - self.image_size
        h_start = h - self.image_size

        random_w = 1 if w_start <= 1 else torch.randint(low=1, high=w_start, size=(1, 1)).item()
        random_h = 1 if h_start <= 1 else torch.randint(low=1, high=h_start, size=(1, 1)).item()
        return random_w, random_h

    def __getitem__(self, index):
        c_img, (o_img, u_img, train_type) = self.iget_imgs[index]

        o_img = Image.open(o_img).convert("RGB")
        u_img = Image.open(u_img).convert("RGB")

        o_img = self.img_transform(o_img)
        u_img = self.img_transform(u_img)
        combime_img = self.img_resize(torch.cat((o_img, u_img), dim=0))
        o_img = combime_img[:3, :, :]
        u_img = combime_img[3:, :, :]

        if torch.rand(1) > 0.5:
            o_img, u_img = u_img, o_img
        if train_type == self.train_type[2]:
            u_img = torch.rand_like(u_img)
        elif train_type == self.train_type[3]:
            o_img = torch.rand_like(o_img)

        return o_img, u_img, train_type


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

