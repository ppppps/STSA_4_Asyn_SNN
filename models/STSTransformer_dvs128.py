import os
import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from . import neuron
from .settings_dvs128 import args

gpu_list = args.gpu
gpu_list_str = ','.join(map(str, gpu_list))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_mask_upper_triangle_mat(head_num: int = 1, T: int = 1, N: int = 1):
    mask_mat_raw = torch.ones(T * N, T * N)
    for t in range(T - 1):
        mask_mat_raw[t * N:(t + 1) * N, (t + 1) * N:] = 0.
    mask_mat = mask_mat_raw.unsqueeze(0).repeat(head_num, 1, 1)
    return mask_mat


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, forward_drop=0., tau=0.5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv3d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_drop = nn.Dropout(p=forward_drop)
        self.fc1_bn = nn.BatchNorm3d(hidden_features)
        self.fc1_lif = neuron.LIFSpike(thresh=1.0, tau=tau, gama=1.0)

        self.fc2_conv = nn.Conv3d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_drop = nn.Dropout(p=forward_drop)
        self.fc2_bn = nn.BatchNorm3d(out_features)
        self.fc2_lif = neuron.LIFSpike(thresh=1.0, tau=tau, gama=1.0)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        x = x.permute(1, 2, 0, 3, 4).contiguous()
        x = self.fc1_conv(x)
        x = self.fc1_drop(x)
        x = self.fc1_bn(x).permute(2, 0, 1, 3, 4).contiguous()
        x = self.fc1_lif(x)

        x = x.permute(1, 2, 0, 3, 4).contiguous()
        x = self.fc2_conv(x)
        x = self.fc2_drop(x)
        x = self.fc2_bn(x).permute(2, 0, 1, 3, 4).contiguous()
        x = self.fc2_lif(x)
        return x


class STSA(nn.Module):
    def __init__(self, dim, num_heads=1, attn_drop=0., proj_drop=0., T=0, H=0, W=0, tau=0.5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = [T, H, W]

        mask_mat = generate_mask_upper_triangle_mat(num_heads, T, H * W)
        self.register_buffer('mask_mat', mask_mat)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1),
                        num_heads))

        trunc_normal_(self.relative_position_bias_table, std=.02)

        # get pair-wise relative position index for each token of a DVS sample
        coordinates_t = torch.arange(self.window_size[0])
        coordinates_h = torch.arange(self.window_size[1])
        coordinates_w = torch.arange(self.window_size[2])
        coordinates = torch.stack(torch.meshgrid(coordinates_t, coordinates_h, coordinates_w))
        coordinates_flatten = torch.flatten(coordinates, 1)
        relative_coordinates = coordinates_flatten[:, :, None] - coordinates_flatten[:, None, :]
        relative_coordinates = relative_coordinates.permute(1, 2, 0).contiguous()
        relative_coordinates[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coordinates[:, :, 1] += self.window_size[1] - 1
        relative_coordinates[:, :, 2] += self.window_size[2] - 1

        # Avoid having the same index number in different locations
        relative_coordinates[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coordinates[:, :, 1] *= (2 * self.window_size[2] - 1)

        # Generate the final location index
        relative_position_index = relative_coordinates.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv_conv = nn.Conv1d(dim, dim * 3, kernel_size=1, stride=1, bias=False)
        self.qkv_bn = nn.BatchNorm1d(dim * 3)
        self.qkv_lif = neuron.LIFSpike(tau=tau)

        self.to_qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.to_qkv_bn = nn.BatchNorm1d(dim * 3)
        self.to_qkv_lif = neuron.LIFSpike(tau=tau)

        self.attn_lif = neuron.LIFSpike(tau=tau)
        self.attn_drop = nn.Dropout(p=attn_drop)

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = neuron.LIFSpike(tau=tau)
        self.proj_drop = nn.Dropout(p=proj_drop)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.reshape(B, C, -1).contiguous()  # B, C, T * N

        qkv_out = self.qkv_conv(x_for_qkv)
        qkv_out = self.qkv_bn(qkv_out).reshape(B, C * 3, T, N).permute(2, 0, 1, 3).contiguous()
        qkv_out = self.qkv_lif(qkv_out).permute(1, 0, 3, 2).chunk(3, dim=-1)

        q, k, v = map(lambda z: rearrange(z, 'b t n (h d) -> b h (t n) d', h=self.num_heads), qkv_out)

        # compute the STRPB
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:T * N, :T * N].reshape(-1)].reshape(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1)

        attn = (q @ k.transpose(-2, -1))  # B, head_num, token_num, token_num

        # add STRPB
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn * self.mask_mat.unsqueeze(0)

        x = (attn @ v) * 0.125

        x = x.permute(0, 2, 1, 3).reshape(B, T * N, C).reshape(B, T, N, C).permute(1, 0, 3, 2).contiguous()
        x = self.attn_lif(x)  # T, B, C, N
        x = x.permute(1, 2, 0, 3).contiguous()  # B, C, T, N
        x = x.reshape(B, C, -1).contiguous()  # B, C, T*N
        x = self.proj_conv(x)
        x = self.proj_lif(self.proj_bn(x).reshape(B, C, T, N).permute(2, 0, 1, 3).reshape(T, B, C, H, W).contiguous())

        return x


class encoder_block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., proj_drop=0., attn_drop=0.,
                 forward_drop=0.,
                 drop_path=0., T=0, H=0, W=0, tau=0.5):
        super().__init__()
        self.attn = STSA(dim, num_heads=num_heads,
                         attn_drop=attn_drop, proj_drop=proj_drop,
                         T=T, H=H, W=W, tau=tau)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, forward_drop=forward_drop, tau=tau)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class Conv_stem(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, token_len=256, T=10, tau=0.5):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        print(self.image_size, patch_size)
        self.num_patches = self.H * self.W
        self.block1_conv = nn.Conv2d(in_channels, token_len // 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block1_bn = nn.BatchNorm2d(token_len // 16)
        self.block1_lif = neuron.LIFSpike(tau=tau)
        self.block1_mp = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.block2_conv = nn.Conv2d(token_len // 16, token_len // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.block2_bn = nn.BatchNorm2d(token_len // 8)
        self.block2_lif = neuron.LIFSpike(tau=tau)
        self.block2_mp = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.block3_conv = nn.Conv2d(token_len // 8, token_len // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.block3_bn = nn.BatchNorm2d(token_len // 4)
        self.block3_lif = neuron.LIFSpike(tau=tau)
        self.block3_mp = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.block4_conv = nn.Conv2d(token_len // 4, token_len, kernel_size=3, stride=1, padding=1, bias=False)
        self.block4_bn = nn.BatchNorm2d(token_len)
        self.block4_lif = neuron.LIFSpike(tau=tau)
        self.block4_mp = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.block5_conv = nn.Conv2d(token_len, token_len, kernel_size=3, stride=1, padding=1, bias=False)
        self.block5_bn = nn.BatchNorm2d(token_len)
        self.block5_lif = neuron.LIFSpike(tau=tau)
        print('tau=', tau)

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.block1_conv(x.flatten(0, 1))
        x = self.block1_bn(x).reshape(T, B, -1, 128, 128).contiguous()
        x = self.block1_lif(x).flatten(0, 1).contiguous()
        x = self.block1_mp(x)

        x = self.block2_conv(x)
        x = self.block2_bn(x).reshape(T, B, -1, 64, 64).contiguous()
        x = self.block2_lif(x).flatten(0, 1).contiguous()
        x = self.block2_mp(x)

        x = self.block3_conv(x)
        x = self.block3_bn(x).reshape(T, B, -1, 32, 32).contiguous()
        x = self.block3_lif(x).flatten(0, 1).contiguous()
        x = self.block3_mp(x)

        x = self.block4_conv(x)
        x = self.block4_bn(x).reshape(T, B, -1, 16, 16).contiguous()
        x = self.block4_lif(x).flatten(0, 1).contiguous()
        x = self.block4_mp(x)

        x_out = x.reshape(T, B, -1, 8, 8).contiguous()
        x = self.block5_conv(x)
        x = self.block5_bn(x).reshape(T, B, -1, 8, 8).contiguous()
        x = self.block5_lif(x)
        x = x + x_out

        return x


class STSTransformer(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11,
                 token_len=256, num_heads=1, mlp_ratios=4,
                 proj_drop_rate=0., attn_drop_rate=0., forward_drop_rate=0.,
                 depths=1, T=10, tau=0.5
                 ):
        super().__init__()
        self.H, self.W = img_size_h // patch_size, img_size_w // patch_size
        self.num_patches = self.H * self.W

        self.num_classes = num_classes
        self.token_len = token_len
        self.depths = depths
        self.T = T
        dpr = [forward_drop_rate] * depths
        print(dpr)
        self.conv_stem = Conv_stem(img_size_h=img_size_h,
                                   img_size_w=img_size_w,
                                   patch_size=patch_size,
                                   in_channels=in_channels,
                                   token_len=token_len, T=T, tau=tau)

        self.block = nn.ModuleList([encoder_block(
            dim=token_len, num_heads=num_heads, mlp_ratio=mlp_ratios,
            proj_drop=proj_drop_rate, attn_drop=attn_drop_rate, forward_drop=dpr[j],
            T=self.T, H=self.H, W=self.W, tau=tau)
            for j in range(depths)])

        # classification
        self.classification = nn.Linear(token_len, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = x.transpose(0, 1)

        x = self.conv_stem(x.float())
        for i in range(len(self.block)):
            x = self.block[i](x)

        x = x.flatten(3).mean(3)

        x = rearrange(x, 't b c -> (t b) c')
        x = self.classification(x)
        x = x.reshape(self.T, -1, self.num_classes)
        x = x.transpose(0, 1)
        return x
