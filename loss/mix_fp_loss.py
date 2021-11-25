import torch.nn as nn
import torch

class SelfTrainLoss(nn.Module):
    def __init__(self):
        super(SelfTrainLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.is_train = False
        self.iteres = {
            'self_supervise_common_difficult': 0,
            'self_supervised_common_mix': 0,
            'self_supervised_upper_mix': 0,
            'self_supervised_lower_mix': 0,
            'self_supervised_fusion_mix': 0,
            'self_similar': 0,
            'zero_ul_constraint': 0,
            'self_supervised_common_easy': 0,
            'self_upper_constraint': 0,
            'zero_common_constraint': 0,
            'self_supervised_upper': 0,
            'self_lower_constraint': 0,
            'self_supervised_lower': 0,
            'total_loss': 0
        }

    def inital_losses(self, b_input_type, losses, compute_num):
        if 'self_supervised_mixup' in b_input_type:
            tmp = b_input_type.count('self_supervised_mixup')
            losses['self_supervised_common_mix'] = 0
            compute_num['self_supervised_common_mix'] = tmp
            losses['self_supervised_upper_mix'] = 0
            compute_num['self_supervised_upper_mix'] = tmp
            losses['self_supervised_lower_mix'] = 0
            compute_num['self_supervised_lower_mix'] = tmp
            losses['self_supervised_fusion_mix'] = 0
            compute_num['self_supervised_fusion_mix'] = tmp


    def forward(self, img1, img2, gt_img, common_part, upper_part, lower_part, fusion_part, b_input_type):
        losses = {}
        compute_num = {}
        losses['total_loss'] = 0

        self.inital_losses(b_input_type, losses, compute_num)
        for index, input_type in enumerate(b_input_type):
            common_part_i = common_part[index].unsqueeze(0)
            upper_part_i = upper_part[index].unsqueeze(0)
            lower_part_i = lower_part[index].unsqueeze(0)
            fusion_part_i = fusion_part[index].unsqueeze(0)
            img1_i = img1[index].unsqueeze(0)
            img2_i = img2[index].unsqueeze(0)
            gt_i = gt_img[index].unsqueeze(0)
            if input_type == 'self_supervised_mixup':
                mask1 = gt_i[:, 0:1, :, :]
                mask2 = gt_i[:, 1:2, :, :]
                gt_img1_i = gt_i[:, 2:5, :, :]
                gt_img2_i = gt_i[:, 5:8, :, :]
                common_mask = ((mask1 == 1.) & (mask2 == 1.)).float()
                gt_common_part = common_mask * gt_img1_i
                gt_upper_part = (mask1 - common_mask).abs() * gt_img1_i
                gt_lower_part = (mask2 - common_mask).abs() * gt_img2_i

                if self.iteres['total_loss'] < 3000:
                    common_part_pre = common_part_i * common_mask
                    upper_part_pre = upper_part_i * (mask1 - common_mask).abs()
                    lower_part_pre = lower_part_i * (mask2 - common_mask).abs()
                    common_part_post = 0
                    upper_part_post = 0
                    lower_part_post = 0
                else:
                    annel_alpha = min(self.iteres['total_loss'], 7000) / 7000
                    annel_alpha = annel_alpha ** 2
                    annel_alpha = annel_alpha * 0.15
                    lower_annel_beta = 1
                    if self.iteres['total_loss'] > 40000:
                        annel_alpha *= 0.1
                    common_part_pre = common_part_i * annel_alpha + common_part_i * common_mask * (1 - annel_alpha)
                    upper_part_pre = upper_part_i * annel_alpha +  upper_part_i * (mask1 - common_mask).abs() * (1 - annel_alpha)
                    lower_part_pre = lower_part_i * annel_alpha * lower_annel_beta + lower_part_i * (mask2 - common_mask).abs() * (1 - annel_alpha * lower_annel_beta)

                self_supervised_common_mix_loss =  self.l1_loss(common_part_pre, gt_common_part) #\
                losses['self_supervised_common_mix'] += self_supervised_common_mix_loss #+ self_supervised_common_mix_loss_a_channel
                self_supervised_upper_mix_loss = self.l1_loss(upper_part_pre, gt_upper_part) #+ 5 * \
                losses['self_supervised_upper_mix'] += self_supervised_upper_mix_loss #+ self_supervised_upper_mix_loss_a_channel
                self_supervised_lower_mix_loss = self.l1_loss(lower_part_pre, gt_lower_part) #+ 5 * \
                losses['self_supervised_lower_mix'] += self_supervised_lower_mix_loss #+ self_supervised_lower_mix_loss_a_channel

                if self.iteres['total_loss'] >= 17000:
                    annel_beta = min(self.iteres['total_loss'] - 10000, 14000) / 14000
                    annel_beta = annel_beta ** 2
                    self_supervised_fusion_mix_loss = 1 * self.l1_loss(gt_img1_i, fusion_part_i) * annel_beta
                                                           #+ 0 * self.ssim_loss(gt_img1_i, fusion_part_i))
                else:
                    self_supervised_fusion_mix_loss = torch.tensor(0.0).cuda()
                losses['self_supervised_fusion_mix'] += self_supervised_fusion_mix_loss
                losses['total_loss'] += self_supervised_common_mix_loss + self_supervised_upper_mix_loss \
                                         + self_supervised_lower_mix_loss + self_supervised_fusion_mix_loss #\


        for k, v in losses.items():
            if k in self.iteres.keys():
                self.iteres[k] += 1
            if k != 'total_loss':
                losses[k] = v / compute_num[k]

        return losses, self.iteres.copy()
