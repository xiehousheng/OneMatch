from torch import nn
from einops.einops import rearrange

class DWCBR(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()

        self.conv = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, groups=in_planes, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        return self.relu(self.norm(self.conv(x)))

class DWConv(nn.Module):
    def __init__(self, out_channels,stride=1):
        super(DWConv, self).__init__()

        self.group_conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                                       padding=1, groups=out_channels, bias=False)
        self.norm = nn.BatchNorm2d(out_channels, eps=1e-5)
        self.act = nn.ReLU(inplace=True)
        self.projection = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.group_conv3x3(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.projection(out)

        return out


class MLP(nn.Module):
    def __init__(self, out_channels, mlp_ratio=2, bias=True):
        super().__init__()

        self.conv1 = nn.Conv2d(out_channels, out_channels * mlp_ratio, kernel_size=1, bias=bias)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels * mlp_ratio, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)

        return x


class OMBlock(nn.Module):
    def __init__(self, in_channels, mlp_ratio=2):
        super(OMBlock, self).__init__()
        self.embed = DWCBR(in_channels, in_channels)
        self.dwconv = DWConv(in_channels)
        self.norm = nn.BatchNorm2d(in_channels, eps=1e-5)
        self.mlp = MLP(in_channels, mlp_ratio=mlp_ratio, bias=True)

    def forward(self, x):
        x = self.embed(x)
        x = x + self.dwconv(x)
        out = self.norm(x)
        x = x + self.mlp(out)
        return x


class APC(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1,padding=1,groups=in_channels)
        self.norm = nn.BatchNorm2d(in_channels, eps=1e-5)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))



class GAC(nn.Module):
    def __init__(self, dim, num_heads=8, sr_ratio=4):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.proj = nn.Linear(dim, dim)

        self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.Sigmoid()

    def forward(self, x):
        _, _, H, W = x.shape
        x = rearrange(x, 'n c h w -> n (h w) c')
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = x.permute(0, 2, 1).reshape(B, C, H, W)
        kv = self.sr(kv).reshape(B, C, -1).permute(0, 2, 1)
        kv = self.norm(kv)
        kv = self.kv(kv).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.act(x)
        x = rearrange(x, 'n (h w) c -> n c h w', h=H, w=W)

        return x
