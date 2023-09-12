import torch
import torch.nn as nn
from einops.einops import rearrange

from .backbone import Feature_Extraction
class OneMatch_G(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = Feature_Extraction(dim=256)

    def forward(self, img_vi, img_ir):
        regfeat = self.backbone(
            torch.cat((img_vi, img_ir), dim=0))

        bs = regfeat.shape[0]
        (regfeat_vi, regfeat_ir) = regfeat.split(int(bs / 2))


        regfeat_vi = rearrange(regfeat_vi, 'n c h w -> n (h w) c')
        regfeat_ir = rearrange(regfeat_ir, 'n c h w -> n (h w) c')

        feat_reg_vi, feat_reg_ir = map(lambda feat: feat / feat.shape[-1] ** .5,
                                       [regfeat_vi, regfeat_ir])

        conf = torch.einsum("nlc,nsc->nls", feat_reg_vi,
                            feat_reg_ir) / 0.1

        mask = conf > 0.2
        mask = mask \
               * (conf == conf.max(dim=2, keepdim=True)[0]) \
               * (conf == conf.max(dim=1, keepdim=True)[0])

        mask_v, all_j_ids = mask.max(dim=2)
        b_ids, i_ids = torch.where(mask_v)
        j_ids = all_j_ids[b_ids, i_ids]

        shape=regfeat.shape[3]

        mkpts0 = torch.stack(
            [i_ids % shape, i_ids // shape],
            dim=1) * 8
        mkpts1 = torch.stack(
            [j_ids % shape, j_ids // shape],
            dim=1) * 8

        return mkpts0, mkpts1
