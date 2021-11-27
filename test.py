import argparse
from utils import  util
from data.multi_exposure_dataset import TestDataset
# from data.multi_focus_dataset import TestDataset
# from data.multi_focus_dataset import TestMFFDataset as TestDataset
# from data.visir_fusion_dataset import TestDataset
# from data.visir_fusion_dataset import TestTNODataset as TestDataset
from torch.utils.data import DataLoader
from models.UCTestShareModelProCommon import UCTestSharedNetPro
from tqdm import tqdm
from torchvision.transforms import ToPILImage
import os
import torch
import option.options as option
import logging
import torch.nn.functional as F



parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Multi Data Fusion: Path to option ymal file.')
test_args = parser.parse_args()

opt = option.parse(test_args.opt, is_train=False)
util.mkdir_and_rename(opt['path']['results_root'])  # rename results folder if exists
util.mkdirs((path for key, path in opt['path'].items() if not key == 'results_root'
                     and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)

logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

torch.backends.cudnn.deterministic = True
# convert to NoneDict, which returns None for missing keys
opt = option.dict_to_nonedict(opt)


dataset_opt = opt['dataset']['test']
test_dataset = TestDataset(dataset_opt)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                        num_workers=dataset_opt['workers'], pin_memory=True)
logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_dataset)))


model = UCTestSharedNetPro()
device_id = torch.cuda.current_device()
resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))

model.load_state_dict(resume_state['state_dict'])
model = model.cuda()
model.eval()
torch.cuda.empty_cache()


avg_psnr = 0.0
avg_ssim = 0.0
avg_mae = 0.0
avg_lpips = 0.0
idx = 0
model.eval()
for test_data in tqdm(test_loader):
    with torch.no_grad():
        o_img, u_img, root_name = test_data

        padding_number = 16

        o_img = F.pad(o_img, (padding_number, padding_number, padding_number, padding_number), mode='reflect')
        u_img = F.pad(u_img, (padding_number, padding_number, padding_number, padding_number), mode='reflect')
        o_img = o_img.cuda()
        u_img = u_img.cuda()

        common_part, upper_part, lower_part, fusion_part = model(o_img, u_img)

        o_img = o_img[:, :, padding_number:-padding_number, padding_number:-padding_number]
        u_img = u_img[:, :, padding_number:-padding_number, padding_number:-padding_number]
        common_part = common_part[:, :, padding_number:-padding_number, padding_number:-padding_number]
        upper_part = upper_part[:, :, padding_number:-padding_number, padding_number:-padding_number]
        lower_part = lower_part[:, :, padding_number:-padding_number, padding_number:-padding_number]
        fusion_part = fusion_part[:, :, padding_number:-padding_number, padding_number:-padding_number]
        print("ou img", o_img.shape, u_img.shape, fusion_part.shape, root_name)

        recover = fusion_part
        # Save ground truth
        img_dir = opt['path']['test_images']

        common_img = ToPILImage()(common_part.clamp(0,1)[0])
        c_img_path = os.path.join(img_dir, "{:s}_common.png".format(root_name[0]))
        common_img.save(c_img_path)

        upper_img = ToPILImage()(upper_part.clamp(0,1)[0])
        upper_img_path = os.path.join(img_dir, "{:s}_upper.png".format(root_name[0]))
        upper_img.save(upper_img_path)

        lower_img = ToPILImage()(lower_part.clamp(0,1)[0])
        lower_img_path = os.path.join(img_dir, "{:s}_lower.png".format(root_name[0]))
        lower_img.save(lower_img_path)

        over_img = ToPILImage()(o_img[0])#.convert('L')
        o_img_path = os.path.join(img_dir, "{:s}_over.png".format(root_name[0]))
        over_img.save(o_img_path)

        under_img = ToPILImage()(u_img[0])#.convert('L')
        u_img_path = os.path.join(img_dir, "{:s}_under.png".format(root_name[0]))
        under_img.save(u_img_path)

        recover_img = ToPILImage()(recover.clamp(0,1)[0])#.convert('L')
        save_img_path = os.path.join(img_dir, "{:s}_recover.png".format(root_name[0]))
        recover_img.save(save_img_path)
        # calculate psnr
        idx += 1

        avg_ssim += util.calculate_ssim(o_img, recover) + util.calculate_ssim(u_img, recover)
        logger.info("current {} over ssim is {:.4e} under ssim is {: .4e}".format(root_name[0] ,
                                                       util.calculate_ssim(o_img, recover),
                                                       util.calculate_ssim(u_img, recover)
                                                       ))


avg_ssim = avg_ssim / idx
# log
logger.info('# Test #ssim: {:e}.'.format(avg_ssim))
logger_test = logging.getLogger('test')  # validation logger
logger_test.info('Test ssim: {:e}.'.format(avg_ssim))
logger.info('End of testing.')
