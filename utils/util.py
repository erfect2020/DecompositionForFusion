import os
import logging
from datetime import datetime
import numpy as np
import random
import torch
import math
# from kornia.losses import ssim
from torchvision.transforms import ToTensor
from skimage.metrics import structural_similarity as compare_ssim


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        logger = logging.getLogger('base')
        logger.info('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = ToTensor()(img1)
    img2 = ToTensor()(img2)
    img1 = img1.squeeze().permute(1, 2, 0).cpu().numpy()
    img2 = img2.squeeze().permute(1, 2, 0).cpu().numpy()
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))


def calculate_ssim(img1, img2):
    # ssim_value = ssim(img1, img2, 11, 'mean')
    # return 1 - ssim_value.item()
    img1 = img1.squeeze().permute(1, 2, 0).cpu().numpy()
    img2 = img2.squeeze().permute(1, 2, 0).cpu().numpy()
    ssim_value = compare_ssim(img1, img2, data_range=1, multichannel=True)
    return ssim_value




def calculate_mae(img1, img2):
    mae = torch.mean((img1 - img2).abs(), dim=[2, 3, 1])
    return mae.squeeze().item()


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.random.manual_seed(seed)


def squeeze2d(input, factor):
    if factor == 1:
        return input

    B, C, H, W = input.size()

    assert H % factor == 0 and W % factor == 0, "H or W modulo factor is not 0"

    x = input.view(B, C, H // factor, factor, W // factor, factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(B, C * factor * factor, H // factor, W // factor)

    return x