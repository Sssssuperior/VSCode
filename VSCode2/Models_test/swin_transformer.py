# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
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


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, num, mask=None):

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)

        _, _, nH = relative_position_bias.shape
        prompt_PE = torch.zeros(1, 1, nH).cuda()
        relative_position_bias = torch.cat(
            (prompt_PE.repeat((num + 1), self.window_size[0] * self.window_size[1], 1), relative_position_bias),
            dim=0)
        relative_position_bias = torch.cat((prompt_PE.repeat(self.window_size[0] * self.window_size[1] + (num + 1),
                                                             (num + 1), 1), relative_position_bias), dim=1)

        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            prompt_mask = torch.zeros(nW, 1, 1).cuda()
            mask = torch.cat((prompt_mask.repeat(1, self.window_size[0] * self.window_size[1], (num + 1)), mask),
                             dim=2)
            mask = torch.cat(
                (prompt_mask.repeat(1, (num + 1), self.window_size[0] * self.window_size[1] + (num + 1)), mask),
                dim=1)

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
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
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

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
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

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x, task_prompt, num, domain_prompt):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                x_windows = window_partition(shifted_x, self.window_size)
            else:
                x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
        else:
            shifted_x = x
            x_windows = window_partition(shifted_x, self.window_size)

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        WB, WW, _ = x_windows.shape
        domain_prompt = domain_prompt.repeat_interleave(WB // B, dim=0).squeeze(-2)
        task_prompt = task_prompt.repeat_interleave(WB // B, dim=0)
        x_windows = torch.cat((domain_prompt, x_windows), dim=1)
        x_windows = torch.cat((task_prompt, x_windows), dim=1)
        attn_windows = self.attn(x_windows, num, mask=self.attn_mask)
        attn_windows = attn_windows[:, (num + 1):, :]

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size)
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        flops += self.dim * H * W
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

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


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, task_prompt, num, domain_prompt):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, task_prompt, num, domain_prompt)
            else:
                x = blk(x, task_prompt, num, domain_prompt)
        if self.downsample is not None:
            x_new = self.downsample(x)
            return x, x_new
        else:
            return x, None

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
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

        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               fused_window_process=fused_window_process)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, rgb, depth, domain_prompt, task_prompt, num,
                                                   gate_domain, gate_task, expert_domain, expert_task):
        rgb = self.patch_embed(rgb)
        if self.ape:
            rgb = rgb + self.absolute_pos_embed
        rgb = self.pos_drop(rgb)

        if depth != None:
            depth = self.patch_embed(depth)
            if self.ape:
                depth = depth + self.absolute_pos_embed
            depth = self.pos_drop(depth)

        output_rgb = []
        output_depth = []
        top = 2
        skwed_softmax = 0
        for i in range(4):
            B, HW, _ = rgb.shape

            rgb_p = F.avg_pool2d(rgb, kernel_size=(HW, 1))
            rgb_d = gate_domain[i](rgb_p).softmax(dim=-1)
            rgb_t = gate_task[i](rgb_p).softmax(dim=-1)
            topk_values_d, topk_indices_d = torch.topk(rgb_d, top, dim=-1)
            topk_values_t, topk_indices_t = torch.topk(rgb_t, top, dim=-1)

            domain_prompt_specific = []
            for k1 in range(B):
                indice = topk_indices_d[k1,:,:]
                selected_linears = [expert_domain[m.item()] for m in indice.flatten()]
                prompt1 = topk_values_d[k1,:,0]
                prompt2 = topk_values_d[k1,:,1]
                skwed_softmax += torch.abs(prompt1-prompt2)
                domain_prompt_tmp = (prompt1.unsqueeze(-1) * selected_linears[0][i](torch.stack([p.data for p in domain_prompt[i]])).sum(0) + (prompt2.unsqueeze(-1) *selected_linears[1][i](torch.stack([p.data for p in domain_prompt[i]])).sum(0))).unsqueeze(1)
                domain_prompt_specific.append(domain_prompt_tmp)
            domain_prompt_specific = torch.cat(domain_prompt_specific, dim=0)

            task_prompt_specific = []
            for k2 in range(B):
                indice = topk_indices_t[k2,:,:]
                selected_linears = [expert_task[m.item()] for m in indice.flatten()]
                prompt1 =  topk_values_t[k2,:,0].repeat(1,1,num[i]).flatten(1)
                prompt2 = topk_values_t[k2,:,1].repeat(1,1,num[i]).flatten(1)
                # skwed_softmax += torch.abs(prompt1-prompt2)
                task_prompt_tmp = (prompt1.unsqueeze(-1) * selected_linears[0][i](torch.stack([p.data for p in task_prompt[i]]))).sum(0) + (prompt2.unsqueeze(-1) * selected_linears[0][i](torch.stack([p.data for p in task_prompt[i]]))).sum(0)
                task_prompt_specific.append(task_prompt_tmp)
            task_prompt_specific = torch.cat(task_prompt_specific, dim=0)

            save, rgb = self.layers[i](rgb, task_prompt_specific, num[i], domain_prompt_specific)
            output_rgb.append(save)

            if depth != None:
                B1, _, _ = depth.shape
                depth_p = F.avg_pool2d(depth, kernel_size=(HW, 1))
                depth_d = gate_domain[i](depth_p).softmax(dim=-1)
                depth_t = gate_task[i](depth_p).softmax(dim=-1)
                topk_values_d, topk_indices_d = torch.topk(depth_d, top, dim=-1)
                topk_values_t, topk_indices_t = torch.topk(depth_t, top, dim=-1)

                domain_prompt_specific = []
                for k1 in range(B1):
                    indice = topk_indices_d[k1,:,:]
                    selected_linears = [expert_domain[m.item()] for m in indice.flatten()]
                    prompt1 = topk_values_d[k1,:,0]
                    prompt2 = topk_values_d[k1,:,1]
                    skwed_softmax += torch.abs(prompt1-prompt2)
                    domain_prompt_tmp = (prompt1.unsqueeze(-1) * selected_linears[0][i](torch.stack([p.data for p in domain_prompt[i]])).sum(0) + (prompt2.unsqueeze(-1) *selected_linears[1][i](torch.stack([p.data for p in domain_prompt[i]])).sum(0))).unsqueeze(1)
                    domain_prompt_specific.append(domain_prompt_tmp)
                domain_prompt_specific = torch.cat(domain_prompt_specific, dim=0)

                task_prompt_specific = []
                for k2 in range(B1):
                    indice = topk_indices_t[k2,:,:]
                    selected_linears = [expert_task[m.item()] for m in indice.flatten()]
                    prompt1 =  topk_values_t[k2,:,0].repeat(1,1,num[i]).flatten(1)
                    prompt2 = topk_values_t[k2,:,1].repeat(1,1,num[i]).flatten(1)
                    # skwed_softmax += torch.abs(prompt1-prompt2)
                    task_prompt_tmp = (prompt1.unsqueeze(-1) * selected_linears[0][i](torch.stack([p.data for p in task_prompt[i]]))).sum(0) + (prompt2.unsqueeze(-1) * selected_linears[0][i](torch.stack([p.data for p in task_prompt[i]]))).sum(0)
                    task_prompt_specific.append(task_prompt_tmp)
                task_prompt_specific = torch.cat(task_prompt_specific, dim=0)

                save, depth = self.layers[i](depth, task_prompt_specific, num[i], domain_prompt_specific)
                output_depth.append(save)

        return output_rgb, output_depth

    def forward(self, rgb, depth, domain_prompt, task_prompt, num,
                                                   gate_domain, gate_task, expert_domain, expert_task):
        output_rgb, output_depth = self.forward_features(rgb, depth, domain_prompt, task_prompt, num,
                                                   gate_domain, gate_task, expert_domain, expert_task)
        return output_rgb, output_depth

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


def swin_transformer(pretrained=True, **kwargs):
    model = SwinTransformer(img_size=352, embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24],
                            window_size=11, drop_path_rate=0.2)
    args = kwargs['args']
    if pretrained:
        pretrained_dict = torch.load(args.pretrained_model)["model"]
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model.state_dict()}
        for k, v in list(pretrained_dict.items()):
            if ('attn.relative_position_index' in k) or ('attn_mask' in k):
                pretrained_dict.pop(k)
        if pretrained_dict.get('absolute_pos_embed') is not None:
            absolute_pos_embed = pretrained_dict['absolute_pos_embed']
            N1, L, C1 = absolute_pos_embed.size()
            N2, C2, H, W = model.absolute_pos_embed.size()
            if N1 != N2 or C1 != C2 or L != H * W:
                print("no")
            else:
                pretrained_dict['absolute_pos_embed'] = absolute_pos_embed.view(N2, H, W, C2).permute(0, 3, 1, 2)

        relative_position_bias_table_keys = [k for k in pretrained_dict.keys() if
                                             "relative_position_bias_table" in k]
        for table_key in relative_position_bias_table_keys:
            table_pretrained = pretrained_dict[table_key]
            table_current = model.state_dict()[table_key]
            L1, nH1 = table_pretrained.size()
            L2, nH2 = table_current.size()
            if nH1 == nH2:
                if L1 != L2:
                    S1 = int(L1 ** 0.5)
                    S2 = int(L2 ** 0.5)
                    table_pretrained_resized = F.interpolate(
                        table_pretrained.permute(1, 0).view(1, nH1, S1, S1),
                        size=(S2, S2), mode='bicubic')
                    pretrained_dict[table_key] = table_pretrained_resized.view(nH2, L2).permute(1, 0)
        model.load_state_dict(pretrained_dict, strict=False)

        print('Model loaded from {}'.format(args.pretrained_model))
    return model