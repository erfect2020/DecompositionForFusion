import torch.nn as nn
import torch
import torch.nn.init as init
from .resnet import ResNestLayer, Bottleneck
import math
import torch.nn.functional as F


class UCTestSharedNetPro(nn.Module):
    def __init__(self):
        super(UCTestSharedNetPro, self).__init__()
        encoder_upper = [nn.Conv2d(3, 16, 3, 1, 1, bias=True),
                         nn.ReLU(inplace=True),
                         ResNestLayer(Bottleneck, 8, 6, stem_width=8, norm_layer=None),
                         ]
        self.encoder_upper = nn.Sequential(*encoder_upper)
        # self.encoder_upper_in = nn.InstanceNorm2d(64,affine=True)
        self.maxpool_upper = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.upper_encoder_layer1 = ResNestLayer(Bottleneck, 16, 6, stem_width=16, norm_layer=None, is_first=False)
        self.upper_encoder_layer2 = ResNestLayer(Bottleneck, 32, 4, stem_width=32, stride=2, norm_layer=None)
        self.upper_encoder_layer3 = ResNestLayer(Bottleneck, 64, 4, stem_width=64, stride=2, norm_layer=None)

        self.encoder_lower = self.encoder_upper
        self.maxpool_lower = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.lower_encoder_layer1 = self.upper_encoder_layer1
        self.lower_encoder_layer2 = self.upper_encoder_layer2
        self.lower_encoder_layer3 = self.upper_encoder_layer3

        encoder_body_fusion = [
            ResNestLayer(Bottleneck, 256, 4, stem_width=256, norm_layer=None, is_first=False)
        ]
        self.common_encoder = nn.Sequential(*encoder_body_fusion)

        self.decoder_common_layer1 = ResNestLayer(Bottleneck, 64, 2, stem_width=512, avg_down=False, avd=False, stride=1, norm_layer=None)
        self.decoder_common_up1 = nn.Upsample(scale_factor=2, mode='bilinear') # nn.PixelShuffle(2)
        self.decoder_common_layer2 = ResNestLayer(Bottleneck, 16, 2, stem_width=128, avg_down=False, avd=False, stride=1, norm_layer=None)
        self.decoder_common_up2 = nn.Upsample(scale_factor=2, mode='bilinear') # nn.PixelShuffle(2)
        self.decoder_common_layer3 = ResNestLayer(Bottleneck, 4, 2, stem_width=32, avg_down=False, avd=False, stride=1, norm_layer=None)
        self.decoder_common_up3 = nn.Upsample(scale_factor=2, mode='bilinear') # nn.PixelShuffle(2)
        decoder_common_layer4 = [
            ResNestLayer(Bottleneck, 4, 2, stem_width=8, avg_down=False, avd=False, stride=1, norm_layer=None),
        ]
        self.decoder_common_layer4 = nn.Sequential(*decoder_common_layer4)
        decoder_projection_layer = [nn.Conv2d(16, 3, 3, 1, 1, bias=True),
                                    nn.ReLU(inplace=True)]
        self.decoder_common_projection_layer = nn.Sequential(*decoder_projection_layer)

        self.decoder_upper_layer1 = ResNestLayer(Bottleneck, 96, 4, stem_width=640, avg_down=False, avd=False, stride=1, norm_layer=None)
        self.decoder_upper_up1 = nn.Upsample(scale_factor=2, mode='bilinear') # nn.PixelShuffle(2)
        self.decoder_upper_layer2 = ResNestLayer(Bottleneck, 32, 4, stem_width=256, avg_down=False, avd=False, stride=1, norm_layer=None)
        self.decoder_upper_up2 = nn.Upsample(scale_factor=2, mode='bilinear') # nn.PixelShuffle(2)
        self.decoder_upper_layer3 = ResNestLayer(Bottleneck, 16, 6, stem_width=96, avg_down=False, avd=False, stride=1, norm_layer=None)
        self.decoder_upper_up3 = nn.Upsample(scale_factor=2, mode='bilinear') # nn.PixelShuffle(2)
        decoder_upper_layer4 = [
            ResNestLayer(Bottleneck, 4, 6, stem_width=32, avg_down=False, avd=False, stride=1, norm_layer=None),
        ]
        self.decoder_upper_layer4 = nn.Sequential(*decoder_upper_layer4)
        upper_decoder_projection_layer = [nn.Conv2d(16, 3, 3, 1, 1, bias=True),
                                          nn.ReLU(inplace=True)]
        self.decoder_upper_projection_layer = nn.Sequential(*upper_decoder_projection_layer)

        self.decoder_lower_layer1 = self.decoder_upper_layer1
        self.decoder_lower_up1 = nn.Upsample(scale_factor=2, mode='bilinear')  # nn.PixelShuffle(2)
        self.decoder_lower_layer2 = self.decoder_upper_layer2
        self.decoder_lower_up2 = nn.Upsample(scale_factor=2, mode='bilinear')  # nn.PixelShuffle(2)
        self.decoder_lower_layer3 = self.decoder_upper_layer3
        self.decoder_lower_up3 = nn.Upsample(scale_factor=2, mode='bilinear')  # nn.PixelShuffle(2)
        self.decoder_lower_layer4 = self.decoder_upper_layer4
        self.decoder_lower_projection_layer = self.decoder_upper_projection_layer

        self.fusion_rule = nn.Sequential(*[
            nn.Conv2d(16, 3, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True)
        ])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, img1, img2):

        feature_upper = self.encoder_upper(img1)
        feature_upper0 = self.maxpool_upper(feature_upper)
        feature_upper1 = self.upper_encoder_layer1(feature_upper0)
        # print("feature upper1", feature_upper1.shape)
        feature_upper2 = self.upper_encoder_layer2(feature_upper1)
        # print("feature upper2", feature_upper2.shape)
        feature_upper3 = self.upper_encoder_layer3(feature_upper2)
        # print("feature upper3", feature_upper3.shape)

        feature_lower = self.encoder_lower(img2)
        feature_lower0 = self.maxpool_lower(feature_lower)
        feature_lower1 = self.lower_encoder_layer1(feature_lower0)
        feature_lower2 = self.lower_encoder_layer2(feature_lower1)
        feature_lower3 = self.lower_encoder_layer3(feature_lower2)

        feature_concat = torch.cat((feature_upper3, feature_lower3), dim=1)
        feature_common = self.common_encoder(feature_concat)

        common_part = self.decoder_common_layer1(feature_common)
        common_part = self.decoder_common_up1(common_part)
        common_part = F.interpolate(common_part, size=feature_upper2.shape[2:])
        common_part = self.decoder_common_layer2(common_part)
        common_part = self.decoder_common_up2(common_part)
        # print("common part2", common_part.shape)
        common_part = F.interpolate(common_part, size=feature_upper1.shape[2:])
        common_part = self.decoder_common_layer3(common_part)
        common_part = self.decoder_common_up3(common_part)
        # print("common part3", common_part.shape)
        common_part = F.interpolate(common_part, size=feature_upper.shape[2:])
        common_part = self.decoder_common_layer4(common_part)
        common_part_embedding = common_part
        common_part = self.decoder_upper_projection_layer(common_part)
        # print("common part4", common_part.shape)

        # print("decode feature upper", feature_de_upper.shape)
        feature_de_upper = torch.cat((feature_common, feature_upper3), dim=1)
        upper_part = self.decoder_upper_layer1(feature_de_upper)
        upper_part = self.decoder_upper_up1(upper_part)
        upper_part = F.interpolate(upper_part, size=feature_upper2.shape[2:])
        upper_part = torch.cat((upper_part, feature_upper2), dim=1)
        # print("upper part1", upper_part.shape)
        upper_part = self.decoder_upper_layer2(upper_part)
        upper_part = self.decoder_upper_up2(upper_part)
        upper_part = F.interpolate(upper_part, size=feature_upper1.shape[2:])
        upper_part = torch.cat((upper_part, feature_upper1), dim=1)
        # print("upper part2", upper_part.shape)
        upper_part = self.decoder_upper_layer3(upper_part)
        upper_part = self.decoder_upper_up3(upper_part)
        # print("upper part3", upper_part.shape)
        upper_part = F.interpolate(upper_part, size=feature_upper.shape[2:])
        upper_part = self.decoder_upper_layer4(upper_part)
        # print("upper part4", upper_part.shape)
        upper_part_embeding = upper_part
        upper_part = self.decoder_upper_projection_layer(upper_part)

        # print("decode feature lower", feature_de_lower.shape)
        feature_de_lower = torch.cat((feature_common, feature_lower3), dim=1)
        lower_part = self.decoder_lower_layer1(feature_de_lower)
        lower_part = self.decoder_lower_up1(lower_part)
        lower_part = F.interpolate(lower_part, size=feature_upper2.shape[2:])
        lower_part = torch.cat((lower_part, feature_lower2), dim=1)
        # print("lower part1", lower_part.shape)
        lower_part = self.decoder_lower_layer2(lower_part)
        lower_part = self.decoder_lower_up2(lower_part)
        lower_part = F.interpolate(lower_part, size=feature_upper1.shape[2:])
        lower_part = torch.cat((lower_part, feature_lower1), dim=1)
        # print("lower part2", lower_part.shape)
        lower_part = self.decoder_lower_layer3(lower_part)
        lower_part = self.decoder_lower_up3(lower_part)
        lower_part = F.interpolate(lower_part, size=feature_upper.shape[2:])
        # print("lower part3", lower_part.shape)
        lower_part = self.decoder_lower_layer4(lower_part)
        # print("lower part4", lower_part.shape)
        lower_part_embeddding = lower_part
        lower_part = self.decoder_lower_projection_layer(lower_part)

        # upper_part_f = torch.cat((upper_part[:, 0:1, :, :], lower_part[:, 0:1, :, :], common_part[:, 0:1, :, :]), dim=1)
        # lower_part_f = torch.cat((upper_part[:, 1:2, :, :], lower_part[:, 1:2, :, :], common_part[:, 1:2, :, :]), dim=1)
        # common_part_f = torch.cat((upper_part[:, 2:3, :, :], lower_part[:, 2:3, :, :], common_part[:, 2:3, :, :]), dim=1)
        # fusion_part = self.fusion_rule(torch.cat((upper_part_f, lower_part_f, common_part_f), dim=1))
        # fusion_partart = self.fusion_rule(
        #     torch.cat((upper_part_embeding, lower_part_embeddding, common_part_embedding), dim=1))
        fusion_part = self.fusion_rule(upper_part_embeding+lower_part_embeddding+ common_part_embedding)
        # fusion_part = self.fusion_rule(torch.cat((upper_part, lower_part, common_part), dim=1))

        # return common_part, upper_part, lower_part, fusion_part
        # return common_part, upper_part, lower_part, fusion_part
        return common_part_embedding, upper_part_embeding, lower_part_embeddding, fusion_part
