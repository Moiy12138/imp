import torch
from torch import nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import dataloader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from utils import *

import timm
from timm.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
from pdb import set_trace as st

from kan import KANLinear, KAN
from torch.nn import init

from einops import rearrange
import torch.utils.checkpoint as checkpoint


class KANLayer(nn.Module):
    def __init__(self, 
                in_features,
                hidden_features=None,
                out_features=None,
                act_layer=nn.GELU,
                drop=0.,
                no_kan=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features

        grid_size = 5
        spline_order = 3
        scale_noise = 0.1
        scale_base = 1.0
        scale_spline=1.0
        base_activation = torch.nn.SiLU
        grid_eps = 0.02
        grid_range = [-1, 1]

        if not no_kan:
            self.fc1 = KANLinear(
                in_features,
                hidden_features,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
            self.fc2 = KANLinear(
                hidden_features,
                out_features,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
            self.fc3 = KANLinear(
                hidden_features,
                out_features,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.fc3 = nn.Linear(hidden_features, out_features)

        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(hidden_features)
        self.dwconv_3 = DW_bn_relu(hidden_features)

        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    

    def forward(self, x, H, W):
        # pdb.set_trace()
        B, N, C = x.shape

        x = self.fc1(x.reshape(B*N,C))
        x = x.reshape(B,N,C).contiguous()
        x = self.dwconv_1(x, H, W)
        x = self.fc2(x.reshape(B*N,C))
        x = x.reshape(B,N,C).contiguous()
        x = self.dwconv_2(x, H, W)
        x = self.fc3(x.reshape(B*N,C))
        x = x.reshape(B,N,C).contiguous()
        x = self.dwconv_3(x, H, W)

        return x

class KANBlock(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, no_kan=False):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim)

        self.layer = KANLayer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, no_kan=no_kan)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.layer(self.norm2(x), H, W))

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super(DW_bn_relu, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    """

    def __init__(
            self,
            img_size=224,
            patch_size=7,
            stride=4,
            in_chans=3,
            embed_dim=768,
        ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2)
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W  

class STPatchEmbed(nn.Module):
    """
    Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 2.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 48.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=2, in_chans=3, embed_dim=48, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class D_ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(D_ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

def window_partition(x, window_size):
    """
    Args:
        Batch, Hight, Width, Chennal
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        # hidden_dim is 4x rate by default
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DWSConv(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., norm_layer=nn.BatchNorm2d):
        super().__init__()
        out_features = out_features or in_features
        # hidden_dim is 4x rate by default
        hidden_features = hidden_features or in_features
        self.pw_conv1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=True)
        self.norm1 = norm_layer(hidden_features)
        self.dw_conv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, 
                                stride=1, padding=1, groups=hidden_features, bias=True)
        self.norm2 = norm_layer(hidden_features)
        self.act = act_layer()
        self.pw_conv2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # B, N, C = x.shape
        # H = W = int(N ** 0.5)
        # x = x.transpose(1, 2).view(B, C, H, W)
        res = x
        x = self.pw_conv1(x)
        x = self.norm1(x)
        x = self.dw_conv(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.pw_conv2(x)
        x = self.drop(x)
        x = x + res
        # x = x.flatten(2).transpose(1, 2)
        
        return x

class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
        

class WindowAttention(nn.Module):
    r""" 
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.
        proj_drop (float, optional): Dropout ratio of output. Default: 0.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        # TODO
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class STransformerBlock(nn.Module):
    """ 
    STransformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp or kan_layer hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.
        attn_drop (float, optional): Attention dropout rate. Default: 0.
        drop_path (float, optional): Stochastic depth rate. Default: 0.
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        #self.dwsc = DWSConv(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        #x = x + self.drop_path(self.dwsc(self.norm2(x), H, W))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class FinalPatchExpand_X2(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, (dim_scale ** 2) * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)

        return x

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x


class MY_Unet(nn.Module):
    def __init__(
            self,
            num_classes,
            in_chans=3,
            deep_supervision=False,
            img_size=224,
            patch_size=16,
            embed_dims=[192, 384, 768],
            no_kan=False,
            drop_rate=0.1,
            drop_patch_rate=0.1,
            norm_layer=nn.LayerNorm,
            depths=[1, 1, 1],
            # ST init
            STpatch_size=2,
            STdepths=[2, 2, 2],
            mlp_ratio=4.,
            patch_norm=True,
            final_upsample="expand_first",
            STembed_dim=48,
            num_heads=[3, 6, 12, 24],
            window_size=7,
            qkv_bias=True,
            qk_scale=None,
            attn_drop_rate=0.,
            **kwargs,
        ):
        super().__init__()

        # # ST
        # self.num_classes = num_classes
        # self.embed_dim = STembed_dim
        # self.num_layers = len(depths)
        # self.mlp_ratio = mlp_ratio
        # self.patch_norm = patch_norm
        # self.final_upsample = final_upsample
        # self.norm_enc_out = norm_layer(embed_dims[0])

        # self.patch_embed = STPatchEmbed(
        #     img_size=img_size,
        #     patch_size=STpatch_size,
        #     in_chans=in_chans,
        #     embed_dim=STembed_dim,
        #     norm_layer=norm_layer,
        # )

        # num_patches = self.patch_embed.num_patches
        # patches_resolution = self.patch_embed.patches_resolution
        # self.patches_resolution = patches_resolution
        # self.pos_drop = nn.Dropout(p=drop_rate)
        # ST_dpr = [x.item() for x in torch.linspace(0, drop_patch_rate, sum(STdepths))]
        # dpr_s0 = ST_dpr[0:STdepths[0]]
        # dpr_s1 = ST_dpr[STdepths[0]:STdepths[0]+STdepths[1]]
        # dpr_s2 = ST_dpr[STdepths[0]+STdepths[1]:STdepths[0]+STdepths[1]+STdepths[2]]
        # # ==============================================================
        # # encoder
        # res0 = (patches_resolution[0], patches_resolution[1]) # 112 112    
        # dim0 = STembed_dim # 48
        # self.enc0_block0 = STransformerBlock(
        #     dim=dim0,
        #     input_resolution=res0,
        #     num_heads=num_heads[0],
        #     window_size=window_size,
        #     shift_size=0,
        #     mlp_ratio=mlp_ratio,
        #     qkv_bias=qkv_bias,
        #     qk_scale=qk_scale,
        #     drop=drop_rate,
        #     attn_drop=attn_drop_rate,
        #     drop_path=dpr_s0[0],
        #     act_layer=nn.GELU,
        #     norm_layer=norm_layer,
        # )
        # self.enc0_block1 = STransformerBlock(
        #     dim=dim0,
        #     input_resolution=res0,
        #     num_heads=num_heads[0],
        #     window_size=window_size,
        #     shift_size=window_size // 2,
        #     mlp_ratio=mlp_ratio,
        #     qkv_bias=qkv_bias,
        #     qk_scale=qk_scale,
        #     drop=drop_rate,
        #     attn_drop=attn_drop_rate,
        #     drop_path=dpr_s0[1],
        #     act_layer=nn.GELU,
        #     norm_layer=norm_layer,
        # )
        # self.enc0_down = PatchMerging(
        #     input_resolution=res0,
        #     dim=dim0,
        #     norm_layer=norm_layer,
        # )

        # # =========================================================
        # res1 = (res0[0]//2, res0[1]//2)   # 56 56
        # dim1 = dim0 * 2 # 96
        # self.enc1_block0 = STransformerBlock(
        #     dim=dim1,
        #     input_resolution=res1,
        #     num_heads=num_heads[1],
        #     window_size=window_size,
        #     shift_size=0,
        #     mlp_ratio=mlp_ratio,
        #     qkv_bias=qkv_bias,
        #     qk_scale=qk_scale,
        #     drop=drop_rate,
        #     attn_drop=attn_drop_rate,
        #     drop_path=dpr_s1[0],
        #     act_layer=nn.GELU,
        #     norm_layer=norm_layer,
        # )
        # self.enc1_block1 = STransformerBlock(
        #     dim=dim1,
        #     input_resolution=res1,
        #     num_heads=num_heads[1],
        #     window_size=window_size,
        #     shift_size=window_size // 2,
        #     mlp_ratio=mlp_ratio,
        #     qkv_bias=qkv_bias,
        #     qk_scale=qk_scale,
        #     drop=drop_rate,
        #     attn_drop=attn_drop_rate,
        #     drop_path=dpr_s1[1],
        #     act_layer=nn.GELU,
        #     norm_layer=norm_layer,
        # )
        # # 28 28 192
        # self.enc1_down = PatchMerging(
        #     input_resolution=res1,
        #     dim=dim1,
        #     norm_layer=norm_layer,
        # )
        # # =========================================================
        # res2 = (res1[0]//2, res1[1]//2)   # 28,28
        # dim2 = dim1 * 2 # 192
        # self.enc2_block0 = STransformerBlock(
        #     dim=dim2,
        #     input_resolution=res2,
        #     num_heads=num_heads[2],
        #     window_size=window_size,
        #     shift_size=0,
        #     mlp_ratio=mlp_ratio,
        #     qkv_bias=qkv_bias,
        #     qk_scale=qk_scale,
        #     drop=drop_rate,
        #     attn_drop=attn_drop_rate,
        #     drop_path=dpr_s2[0],
        #     act_layer=nn.GELU,
        #     norm_layer=norm_layer,
        # )
        # self.enc2_block1 = STransformerBlock(
        #     dim=dim2,
        #     input_resolution=res2,
        #     num_heads=num_heads[2],
        #     window_size=window_size,
        #     shift_size=window_size // 2,
        #     mlp_ratio=mlp_ratio,
        #     qkv_bias=qkv_bias,
        #     qk_scale=qk_scale,
        #     drop=drop_rate,
        #     attn_drop=attn_drop_rate,
        #     drop_path=dpr_s2[1],
        #     act_layer=nn.GELU,
        #     norm_layer=norm_layer,
        # )


        # self.dec2_concat_linear = nn.Linear(dim2+dim2, dim2)
        # self.dec2_block0 = STransformerBlock(
        #     dim=dim2,
        #     input_resolution=res2,
        #     num_heads=num_heads[2],
        #     window_size=window_size,
        #     shift_size=0,
        #     mlp_ratio=mlp_ratio,
        #     qkv_bias=qkv_bias,
        #     qk_scale=qk_scale,
        #     drop=drop_rate,
        #     attn_drop=attn_drop_rate,
        #     drop_path=drop_patch_rate,
        #     act_layer=nn.GELU,
        #     norm_layer=norm_layer,
        # )
        # self.dec2_block1 = STransformerBlock(
        #     dim=dim2,
        #     input_resolution=res2,
        #     num_heads=num_heads[2],
        #     window_size=window_size,
        #     shift_size=window_size // 2,
        #     mlp_ratio=mlp_ratio,
        #     qkv_bias=qkv_bias,
        #     qk_scale=qk_scale,
        #     drop=drop_rate,
        #     attn_drop=attn_drop_rate,
        #     drop_path=drop_patch_rate,
        #     act_layer=nn.GELU,
        #     norm_layer=norm_layer,
        # )
        # self.dec2_up = PatchExpand(
        #     input_resolution=res2,
        #     dim=dim2,
        #     dim_scale=2,
        #     norm_layer=norm_layer,
        # )

        # # Stage 1: (56x56 96) + skip0(56x56 96)
        # self.dec1_concat_linear = nn.Linear(dim1+dim1, dim1)
        # self.dec1_block0 = STransformerBlock(
        #     dim=dim1,
        #     input_resolution=res1,
        #     num_heads=num_heads[1],
        #     window_size=window_size,
        #     shift_size=0,
        #     mlp_ratio=mlp_ratio,
        #     qkv_bias=qkv_bias,
        #     qk_scale=qk_scale,
        #     drop=drop_rate,
        #     attn_drop=attn_drop_rate,
        #     drop_path=drop_patch_rate,
        #     act_layer=nn.GELU,
        #     norm_layer=norm_layer,
        # )
        # self.dec1_block1 = STransformerBlock(
        #     dim=dim1,
        #     input_resolution=res1,
        #     num_heads=num_heads[1],
        #     window_size=window_size,
        #     shift_size=window_size // 2,
        #     mlp_ratio=mlp_ratio,
        #     qkv_bias=qkv_bias,
        #     qk_scale=qk_scale,
        #     drop=drop_rate,
        #     attn_drop=attn_drop_rate,
        #     drop_path=drop_patch_rate,
        #     act_layer=nn.GELU,
        #     norm_layer=norm_layer,
        # )
        # self.dec1_up = PatchExpand(
        #     input_resolution=res1,
        #     dim=dim1,
        #     dim_scale=2,
        #     norm_layer=norm_layer,
        # )

        # # Stage 0: (112x112 48) + skip0(112x112 48)
        # self.dec0_concat_linear = nn.Linear(dim0+dim0, dim0)
        # self.dec0_block0 = STransformerBlock(
        #     dim=dim0,
        #     input_resolution=res0,
        #     num_heads=num_heads[0],
        #     window_size=window_size,
        #     shift_size=0,
        #     mlp_ratio=mlp_ratio,
        #     qkv_bias=qkv_bias,
        #     qk_scale=qk_scale,
        #     drop=drop_rate,
        #     attn_drop=attn_drop_rate,
        #     drop_path=drop_patch_rate,
        #     act_layer=nn.GELU,
        #     norm_layer=norm_layer,
        # )
        # self.dec0_block1 = STransformerBlock(
        #     dim=dim0,
        #     input_resolution=res0,
        #     num_heads=num_heads[0],
        #     window_size=window_size,
        #     shift_size=window_size // 2,
        #     mlp_ratio=mlp_ratio,
        #     qkv_bias=qkv_bias,
        #     qk_scale=qk_scale,
        #     drop=drop_rate,
        #     attn_drop=attn_drop_rate,
        #     drop_path=drop_patch_rate,
        #     act_layer=nn.GELU,
        #     norm_layer=norm_layer,
        # )
        # self.norm_up = norm_layer(STembed_dim)

        # # Final 2x
        # if self.final_upsample == "expand_first":
        #     self.up2 = FinalPatchExpand_X2(
        #         input_resolution=(img_size // STpatch_size, img_size // STpatch_size),
        #         dim=STembed_dim,
        #         dim_scale=2,
        #         norm_layer=norm_layer,
        #     )
        #     self.output = nn.Conv2d(
        #         in_channels=STembed_dim,
        #         out_channels=num_classes,
        #         kernel_size=1,
        #         bias=False,
        #     )

        # self.apply(self._init_weights)

        # # -----------------------end
        kan_input_dim = embed_dims[0]

        # self.encoder1 = ConvLayer(3, kan_input_dim//8)  # 3-24
        # self.encoder2 = ConvLayer(kan_input_dim//8, kan_input_dim//4)  # 24-48 
        # self.encoder3 = ConvLayer(kan_input_dim//4, kan_input_dim)  # 48-192
        self.encoder1 = ConvLayer(3, kan_input_dim//4)  # 3-48
        self.encoder1_x1 = nn.Conv2d(3, kan_input_dim//4, kernel_size=1, bias=True)
        self.encoder1_eca = eca_layer(channel=kan_input_dim//4)
        self.encoder1_reseca = eca_layer(channel=kan_input_dim//4)

        self.encoder2 = ConvLayer(kan_input_dim//4, kan_input_dim//2)  # 48-96 
        self.encoder2_x1 = nn.Conv2d(kan_input_dim//4, kan_input_dim//2, kernel_size=1, bias=True)
        self.encoder2_eca = eca_layer(channel=kan_input_dim//2)
        self.encoder2_reseca = eca_layer(channel=kan_input_dim//2)
        
        self.encoder3 = ConvLayer(kan_input_dim//2, kan_input_dim)  # 96-192
        self.encoder3_x1 = nn.Conv2d(kan_input_dim//2, kan_input_dim, kernel_size=1, bias=True)
        self.encoder3_eca = eca_layer(channel=kan_input_dim//2)
        self.encoder3_reseca = eca_layer(channel=kan_input_dim//2)
        
        self.norm2 = norm_layer(embed_dims[0]) # 192

        self.norm3 = norm_layer(embed_dims[1]) # 384
        self.tobottleneck_eca = eca_layer(channel=embed_dims[1])

        self.norm4 = norm_layer(embed_dims[2]) # 768
        self.bottleneck_eca = eca_layer(channel=embed_dims[2])
        self.norm4_b = norm_layer(embed_dims[2]) # 768
        # self.dwsc1 = DWSConv(in_features=kan_input_dim//4, hidden_features=kan_input_dim//2, act_layer=nn.ReLU)
        # self.dwsc2 = DWSConv(in_features=kan_input_dim//2, hidden_features=kan_input_dim, act_layer=nn.ReLU)
        # self.dwsc3 = DWSConv(in_features=kan_input_dim, hidden_features=kan_input_dim*2, act_layer=nn.ReLU)
        # self.dwsc4 = DWSConv(in_features=kan_input_dim*2, hidden_features=kan_input_dim*4, act_layer=nn.ReLU)

        self.dnorm3 = norm_layer(embed_dims[1]) # 384
        self.dnorm4 = norm_layer(embed_dims[0]) # 192

        dpr = [x.item() for x in torch.linspace(0, drop_patch_rate, sum(depths))]

        self.block0 = nn.ModuleList(
            [
                KANBlock(
                    dim=embed_dims[0],
                    drop=drop_rate,
                    drop_path=dpr[0],
                    norm_layer=norm_layer
                )
            ]
        )
        self.block1 = nn.ModuleList(
            [
                KANBlock(
                    dim=embed_dims[1],
                    drop=drop_rate,
                    drop_path=dpr[0],
                    norm_layer=norm_layer
                )
            ]
        )
        self.block2 = nn.ModuleList(
            [
                KANBlock(
                    dim=embed_dims[2],
                    drop=drop_rate,
                    drop_path=dpr[1],
                    norm_layer=norm_layer
                )
            ]
        )
        self.block2_b = nn.ModuleList(
            [
                KANBlock(
                    dim=embed_dims[2],
                    drop=drop_rate,
                    drop_path=dpr[1],
                    norm_layer=norm_layer
                )
            ]
        )
        self.dblock1 = nn.ModuleList(
            [
                KANBlock(
                    dim=embed_dims[1],
                    drop=drop_rate,
                    drop_path=dpr[0],
                    norm_layer=norm_layer
                )
            ]
        )
        self.dblock0 = nn.ModuleList(
            [
                KANBlock(
                    dim=embed_dims[0],
                    drop=drop_rate,
                    drop_path=dpr[1],
                    norm_layer=norm_layer
                )
            ]
        )
        self.patch_embed3 = PatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1]) 
        self.patch_embed4 = PatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[1]) # 768 => 384
        self.decoder1_concat_linear = nn.Linear(embed_dims[1]+embed_dims[1], embed_dims[1])

        self.decoder2 = D_ConvLayer(embed_dims[1], embed_dims[0]) # 384 => 192
        self.decoder2_eca = eca_layer(channel=embed_dims[1])
        self.decoder2_x1 = nn.Conv2d(embed_dims[1], embed_dims[0], kernel_size=1, bias=True)
        self.decoder2_concat_linear = nn.Linear(embed_dims[0]+embed_dims[0], embed_dims[0])

        self.decoder3 = D_ConvLayer(embed_dims[0], embed_dims[0]//2) # 192 => 96
        self.decoder3_eca = eca_layer(channel=embed_dims[0])
        self.decoder3_x1 = nn.Conv2d(embed_dims[0], embed_dims[0]//2, kernel_size=1, bias=True)
        self.decoder3_concat_linear = nn.Linear(embed_dims[0]//2+embed_dims[0]//2, embed_dims[0]//2)

        self.decoder4 = D_ConvLayer(embed_dims[0]//2, embed_dims[0]//4) # 96 => 48
        self.decoder4_eca = eca_layer(channel=embed_dims[0]//2)
        self.decoder4_x1 = nn.Conv2d(embed_dims[0]//2, embed_dims[0]//4, kernel_size=1, bias=True)
        self.decoder4_concat_linear = nn.Linear(embed_dims[0]//4+embed_dims[0]//4, embed_dims[0]//4)

        self.decoder5 = D_ConvLayer(embed_dims[0]//4, embed_dims[0]//4) # 96 => 96
        self.decoder5_eca = eca_layer(channel=embed_dims[0]//4)

        self.final = nn.Conv2d(embed_dims[0]//4, num_classes, kernel_size=1)
        self.soft = nn.Softmax(dim=1)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)        

    def forward_features(self, x):
        # input (B 3 224 224) output (B 56*56 96)
        # input (B 3 224 224) output (B 112*112 48)
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        skip0 = x

        # Stage 0
        # input (B 112*112 48)
        x = self.enc0_block0(x)
        x = self.enc0_block1(x)
        # output (B 56*56 96)
        x = self.enc0_down(x)
        skip1 = x

        # Stage 1
        x = self.enc1_block0(x)
        x = self.enc1_block1(x)
        # x = (B 28*28 192)
        x = self.enc1_down(x)
        skip2 = x

        # Stage 2
        x = self.enc2_block0(x)
        x = self.enc2_block1(x)
        # x = (B 28*28 192)
        x = self.norm_enc_out(x)
        return x, (skip0, skip1, skip2)
    
    def forward_bottleneck(self, x):
        # Tokenized KAN Stage 1
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        skip2 = x
        #  (B, 192, 28, 28) (B, 14*14, 384) H=14, W=14
        out, H, W = self.patch_embed3(x)
        # x.shape = (B 28*28 192)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # (B, 384, 14, 14)
        skip1 = out

        # Tokenized KAN Stage 2 Bottleneck
        # (B, 384, 14, 14) (B, 7*7, 768), H=7, W=7
        out, H, W = self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        # (B, 768, 7, 7)
        skip0 = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        for i, blk in enumerate(self.block2_b):
            out = blk(out, H, W)
        out = self.norm4_b(out)
        # (B, 768, 7, 7)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = torch.add(out, skip0)

        # Tokenized KAN Stage 3
        # decoder input (B, 768, 7, 7) out = (B, 384, 14, 14)
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2,2), mode='bilinear'))
        out = torch.add(out, skip1)
        # H 14 W 14
        _, _, H, W = out.shape
        # out (B, 14*14, 384)
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            # out (B, 14*14, 384)
            out = blk(out, H, W)
        out = self.dnorm3(out)
        # out (B, 384, 14, 14)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # decoder Stage 4
        # out (B, 192, 28, 28)
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2,2), mode='bilinear'))
        out = torch.add(out,skip2)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)
        out = self.dnorm4(out)
        # out (B, 192, 28, 28)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return out

    def forward_up_features(self, x, skips):
        skip0, skip1, skip2 = skips
        # x.shape = (B 192 28 28)
        # x = (B, 784, 192)
        x = x.flatten(2).transpose(1, 2)

        x = torch.cat([x, skip2], dim=-1)
        x = self.dec2_concat_linear(x)
        x = self.dec2_block0(x)
        x = self.dec2_block1(x)
        # (B, 3136, 96)
        x = self.dec2_up(x)

        # Decoder 1
        x = torch.cat([x, skip1], dim=-1)
        x = self.dec1_concat_linear(x)
        x = self.dec1_block0(x)
        x = self.dec1_block1(x)
        # (B, 112*112 48)
        x = self.dec1_up(x)

        # Decoder 0
        x = torch.cat([x, skip0], dim=-1)
        x = self.dec0_concat_linear(x)
        x = self.dec0_block0(x)
        x = self.dec0_block1(x)
        x = self.norm_up(x)            

        return x

    def final_x2(self, x):
        B, L, C = x.shape
        H, W = self.patches_resolution
        assert L == H * W
        if self.final_upsample == "expand_first":
            x = self.up2(x)              # 112->224
            x = x.view(B, 2*H, 2*W, -1).permute(0,3,1,2)
            x = self.output(x)
        return x


    def dual_branch(self, x):
        
        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1  3 224 224 -> 48 112 112
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
        out_x1 = self.encoder1_eca(F.relu(F.max_pool2d(self.encoder1_x1(x), 2, 2)))
        # out_x1 = F.relu(F.max_pool2d(self.encoder1_x1(x), 2, 2))
        x = torch.add(out, out_x1)
        # t1 = self.dwsc1(out)
        # t1 = x
        t1 = self.encoder1_reseca(x)

        ### Stage 2 48 112 112 -> 96 56 56
        out = F.relu(F.max_pool2d(self.encoder2(x), 2, 2))
        out_x1 = self.encoder2_eca(F.relu(F.max_pool2d(self.encoder2_x1(x), 2, 2)))
        # out_x1 = F.relu(F.max_pool2d(self.encoder2_x1(x), 2, 2))
        x = torch.add(out, out_x1)
        #t2 = self.dwsc2(out)
        t2 = self.encoder2_reseca(x)


        ### Stage 3 96 56 56 -> 192 28 28
        out = F.relu(F.max_pool2d(self.encoder3(x), 2, 2))
        out_x1 = self.encoder3_eca(F.relu(F.max_pool2d(self.encoder3_x1(x), 2, 2)))
        # out_x1 = F.relu(F.max_pool2d(self.encoder3_x1(x), 2, 2))
        x = torch.add(out, out_x1)
        #t3 = self.dwsc3(out)
        t3 = self.encoder3_reseca(x)

        _, _, H, W = x.shape
        # 28*28 192
        x = x.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.block0):
            x = blk(x, H, W)
        x = self.norm2(x)
        # 192 28 28
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Tokenized KAN Stage
        ### Stage 4
        # 192 28 28 -> 14*14 384
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm3(x)
        # 384 14 14
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # t4 = self.dwsc4(out)
        # t4 = x
        t4 = self.tobottleneck_eca(x)

        ### Bottleneck
        # 384 14 14 -> 7*7 768
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm4(x)
        # (B, 768, 7, 7)
        skip0 = self.bottleneck_eca(x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous())
        for i, blk in enumerate(self.block2_b):
            x = blk(x, H, W)
        x = self.norm4_b(x)
        # (B, 768, 7, 7)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = torch.add(x, skip0)

        ### Stage 4 768 7 7 -> 384 14 14
        x = F.relu(F.interpolate(self.decoder1(x), scale_factor=(2,2), mode ='bilinear'))
        
        x = torch.add(x, t4)
        # 14*14 384
        # _, _, H, W = x.shape
        # x = x.flatten(2).transpose(1, 2)
        # x = torch.cat([x, t4], dim=-1)
        # x = self.decoder1_concat_linear(x)
        # x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        _, _, H, W = x.shape
        # 14*14 384
        x = x.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock1):
            x = blk(x, H, W)
        x = self.dnorm3(x)
        # 14*14 384 -> 384 14 14
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # 384 14 14 -> 192 28 28
        out = F.relu(F.interpolate(self.decoder2(x),scale_factor=(2,2),mode ='bilinear'))
        out_x1 = F.relu(F.interpolate(self.decoder2_x1(self.decoder2_eca(x)),scale_factor=(2,2),mode ='bilinear'))
        # out_x1 = F.relu(F.interpolate(self.decoder2_x1(x),scale_factor=(2,2),mode ='bilinear'))
        x = torch.add(out, out_x1)
        x = torch.add(x, t3)
        # _, _, H, W = x.shape
        # x = x.flatten(2).transpose(1, 2)
        # x = torch.cat([x, t3], dim=-1)
        # x = self.decoder2_concat_linear(x)
        # x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()


        # redundant kan block
        _,_,H,W = x.shape
        # 192 28 28 -> 28*28 192
        x = x.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock0):
            x = blk(x, H, W)
        x = self.dnorm4(x)
        # 28*28 192 -> 192 28 28
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # 192 28 28 -> 96 56 56
        out = F.relu(F.interpolate(self.decoder3(x),scale_factor=(2,2),mode ='bilinear'))
        out_x1 = F.relu(F.interpolate(self.decoder3_x1(self.decoder3_eca(x)),scale_factor=(2,2),mode ='bilinear'))
        # out_x1 = F.relu(F.interpolate(self.decoder3_x1(x),scale_factor=(2,2),mode ='bilinear'))
        x = torch.add(out, out_x1)
        x = torch.add(x, t2)
        # _, _, H, W = x.shape
        # x = x.flatten(2).transpose(1, 2)
        # x = torch.cat([x, t2], dim=-1)
        # x = self.decoder3_concat_linear(x)
        # x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # 96 56 56 -> 48 112 112
        out = F.relu(F.interpolate(self.decoder4(x),scale_factor=(2,2),mode ='bilinear'))
        out_x1 = F.relu(F.interpolate(self.decoder4_x1(self.decoder4_eca(x)),scale_factor=(2,2),mode ='bilinear'))
        # out_x1 = F.relu(F.interpolate(self.decoder4_x1(x),scale_factor=(2,2),mode ='bilinear'))
        x = torch.add(out, out_x1)
        x = torch.add(x, t1)
        # _, _, H, W = x.shape
        # x = x.flatten(2).transpose(1, 2)
        # x = torch.cat([x, t1], dim=-1)
        # x = self.decoder4_concat_linear(x)
        # x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # 48 112 112 -> 48 224 224
        x = F.relu(F.interpolate(self.decoder5(self.decoder5_eca(x)),scale_factor=(2,2),mode ='bilinear'))
        # x = F.relu(F.interpolate(self.decoder5(x),scale_factor=(2,2),mode ='bilinear'))

        return self.final(x)

    def forward(self, x):
        # x, skips = self.forward_features(x)
        # x = self.forward_bottleneck(x)
        # x = self.forward_up_features(x, skips)
        # x = self.final_x2(x)


        x = self.dual_branch(x)
        return x
