from torch.utils.data import Dataset
import os
import json
import torch
from torchvision.transforms import Compose, RandomResizedCrop, ToTensor, RandomCrop, ColorJitter
from torch.distributions.bernoulli import Bernoulli
import torch.nn.functional as F
import cv2
from tqdm import tqdm
from PIL import Image
from kornia.filters import GaussianBlur2d, gaussian_blur2d
import random


class TrainDataset(Dataset):
    def __init__(self, trainopt):
        super(TrainDataset, self).__init__()
        path = trainopt['dataroot']
        self.img_transform = Compose([ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2), ToTensor()])
        self.piltotensor = ToTensor()
        self.colorjitter = ColorJitter(brightness=(0.1, 1.0), contrast=(0.1, 1.0), saturation=(0.1, 1.0), hue=0)
        # self.colorjitter = ColorJitter()
        # self.brightness =  ColorJitter(brightness=(0.4, 1))
        self.brightness = ColorJitter()
        self.img_resize = RandomResizedCrop(trainopt['image_size'])
        self.image_size = trainopt['image_size']
        self.batch_size = trainopt['iter_size']
        self.max_iter = trainopt["max_iter"]
        self.epoch_num = 0
        self.same_radio = 0.2
        self.noise_radio = 0.2
        self.grid_num1 = 14 #14
        self.rand_dist1 = Bernoulli(probs=torch.ones(self.grid_num1 ** 2) * 0.99)
        self.grid_num2 = 28 #56 # 28
        self.rand_dist2 = Bernoulli(probs=torch.ones(self.grid_num2 ** 2) * 0.95)
        self.grid_num3 = 28 # 224 # 28
        self.rand_dist3 = Bernoulli(probs=torch.ones(self.grid_num3 ** 2) * 0.5)
        self.train_type = ['self_supervised_mixup', 'self_supervised_common_easy',
                           'self_supervised_upper', 'self_supervised_lower']

        trainpairs_forreading = str(trainopt['trainpairs'])
        if not os.path.exists(trainpairs_forreading):
            self.iget_imgs = {}
            extra_imgs = []
            extra_imgs = os.path.join(os.path.expanduser(path), trainopt["train_name"])
            extra_imgs = [os.path.join(extra_imgs, os_dir) for os_dir in os.listdir(extra_imgs)]
            extra_imgs = random.sample(extra_imgs, k=50000)

            lowlight_imgs = []

            def not_dir(x):
                return '_MDF' not in x and '.DS_Store' not in x and '.txt' not in x

            lowlight_imgs = list(filter(not_dir, lowlight_imgs))
            extra_imgs = extra_imgs + lowlight_imgs

            for self_img in extra_imgs:
                rand_state= torch.rand(1)
                if rand_state < 1.0:
                    self.iget_imgs[str(self_img) + ':' + self.train_type[0]] = [self_img, self_img, self.train_type[0]]
                elif rand_state < 0.8:
                    self.iget_imgs[str(self_img) + ':' + self.train_type[1]] = [self_img, self_img, self.train_type[1]]
                elif rand_state < 0.9:
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

    def __len__(self):
        return len(self.iget_imgs)

    def reassign_mask(self, radio):
        self.rand_dist1 = Bernoulli(probs=torch.ones(self.grid_num1 ** 2) * radio)
        self.rand_dist2 = Bernoulli(probs=torch.ones(self.grid_num2 ** 2) * radio)
        self.rand_dist3 = Bernoulli(probs=torch.ones(self.grid_num3 ** 2) * radio)

    def __getitem__(self, index):
        c_img, (o_img, u_img, train_type) = self.iget_imgs[index]


        o_img = Image.open(o_img).convert("RGB")
        # o_img = self.img_transform(o_img)
        o_img = self.piltotensor(o_img)
        o_img = self.colorjitter(o_img.unsqueeze(0)).squeeze()

        u_img = o_img.clone()
        # print("size", o_img.shape, u_img.shape)
        combime_img = self.img_resize(torch.cat((o_img, u_img), dim=0))
        o_img = combime_img[:3, :, :]
        u_img = combime_img[3:, :, :]
        gt_img = torch.cat((torch.zeros_like(o_img[:2, :, :]), combime_img), dim=0)

        if train_type == self.train_type[0]:
            self.grid_num1, self.grid_num2, self.grid_num3 = torch.randint(13, 14, [1]).item(), \
                                                             torch.randint(26, 28, [1]).item(), \
                                                             torch.randint(26, 28, [1]).item() ## 222 226 # 27 28
            self.reassign_mask(0.1 + torch.rand(1).item()/1.12)
            grid1 = F.interpolate(self.rand_dist1.sample().reshape(1, 1, self.grid_num1, self.grid_num1),
                                  size=self.image_size, mode='nearest').squeeze()
            grid1 *= F.interpolate(self.rand_dist2.sample().reshape(1, 1, self.grid_num2, self.grid_num2),
                                  size=self.image_size, mode='nearest').squeeze()
            # grid1 *= F.interpolate(self.rand_dist3.sample().reshape(1, 1, self.grid_num3, self.grid_num3),
            #                        size=self.image_size, mode='nearest').squeeze()
            grid2 = F.interpolate(self.rand_dist1.sample().reshape(1, 1, self.grid_num1, self.grid_num1),
                                  size=self.image_size, mode='nearest').squeeze()
            grid2 *= F.interpolate(self.rand_dist2.sample().reshape(1, 1, self.grid_num2, self.grid_num2),
                                  size=self.image_size, mode='nearest').squeeze()
            # grid2 *= F.interpolate(self.rand_dist3.sample().reshape(1, 1, self.grid_num3, self.grid_num3),
            #                       size=self.image_size, mode='nearest').squeeze()

            sample_rand = torch.rand(1).item()
            if sample_rand < 0.33:
                grid3 = F.interpolate(self.rand_dist1.sample().reshape(1, 1, self.grid_num1, self.grid_num1),
                                   size=self.image_size, mode='nearest').squeeze()
            elif sample_rand < 0.66:
                grid3 = F.interpolate(self.rand_dist2.sample().reshape(1, 1, self.grid_num2, self.grid_num2),
                                  size=self.image_size, mode='nearest').squeeze()
            else:
                grid3 = F.interpolate(self.rand_dist3.sample().reshape(1, 1, self.grid_num3, self.grid_num3),
                                   size=self.image_size, mode='nearest').squeeze()

            none_grid = ((grid1 == 0.0) & (grid2 == 0.0)).float()
            none_grid1 = grid3 * none_grid
            none_grid2 = (1 - grid3) * none_grid
            grid1 += none_grid1
            grid2 += none_grid2

            if torch.rand(1).item() < 0.5:
                mask1, mask2 = grid1, grid2
            else:
                mask1, mask2 = grid2, grid1

            o_rand = torch.randn_like(o_img).abs().clamp(0,1)
            u_rand = torch.randn_like(u_img).abs().clamp(0,1)
            o_img = o_img * mask1 + o_rand * (1.0 - mask1) #* torch.rand(1).item()
            u_img = u_img * mask2 + u_rand * (1.0 - mask2) #* torch.rand(1).item()
            gt_img = torch.cat((mask1.unsqueeze(0), mask2.unsqueeze(0), combime_img), dim=0)
            # print("gt_img shape", gt_img.shape)


        return o_img, u_img, gt_img, train_type
