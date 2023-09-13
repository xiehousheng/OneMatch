import torch.nn.functional as F
import copy
from torch import nn
from model.utils import OMBlock, APC, GAC, DWConv
import torch
from einops.einops import rearrange
class Feature_Extraction(nn.Module):

    def __init__(self,dim=256, depths=4, input_size=(360,480)):
        super().__init__()
        self.embedding = nn.Sequential(nn.Conv2d(1, dim, kernel_size=8, stride=8), nn.BatchNorm2d(dim))
        self.backbone = nn.ModuleList([copy.deepcopy(OMBlock(dim)) for _ in range(depths)])
        self.gridencoder = nn.Conv2d(2, dim, kernel_size=3, stride=1, groups=2, padding=1)
        self.apc = nn.ModuleList([copy.deepcopy(APC(dim)) for _ in range(depths+1)])
        self.fpn = nn.ModuleList([DWConv(dim, stride=2), DWConv(dim, stride=2), DWConv(dim), DWConv(dim), DWConv(dim)])
        self.gac = GAC(dim, num_heads=4, sr_ratio=8)

        feat_h, feat_w = int(input_size[1]/8), int(input_size[0]/8)

        xs = torch.linspace(0, feat_h - 1, feat_h)
        ys = torch.linspace(0, feat_w - 1, feat_w)
        self.xs = xs / (feat_h - 1)
        self.ys = ys / (feat_w - 1)

        self.is_cuda=torch.cuda.is_available()
        if self.is_cuda:
            self.grid = torch.stack(torch.meshgrid([self.xs, self.ys]), dim=-1).unsqueeze(0).repeat(1, 1, 1, 1).cuda()
        else:
            self.grid = torch.stack(torch.meshgrid([self.xs, self.ys]), dim=-1).unsqueeze(0).repeat(1, 1, 1, 1)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        grid = rearrange(self.grid, 'n h w c -> n c h w')
        grid = self.gridencoder(grid)

        x = self.embedding(x)

        for i in range(len(self.backbone)):
            at=self.apc[i](x)
            x = at*grid+x
            x=self.backbone[i](x)

        x_16 = self.fpn[0](x)
        x_32 = self.fpn[1](x_16)


        _, _, H, W = x_16.shape
        x32_out = F.interpolate(x_32, size=(H, W), mode='bilinear', align_corners=True)
        x32_out_ = self.fpn[2](x_16 + x32_out)

        _, _, H, W = x.shape
        x16_out = F.interpolate(x32_out_, size=(H, W), mode='bilinear', align_corners=True)
        x = self.fpn[3](x + x16_out)
        x = self.fpn[4](x) + x

        at = self.apc[-1](x)
        x = at * grid + x

        x = x * self.gac(x)

        return x