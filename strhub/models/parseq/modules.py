# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Optional, Sequence

import timm
import torch
from torch import Tensor, nn as nn
from torch.nn import functional as F
from torch.nn.modules import transformer

from timm.layers import build_sincos2d_pos_embed
from timm.models.vision_transformer import Block, PatchEmbed, VisionTransformer


class DecoderLayer(nn.Module):
    """A Transformer decoder layer supporting two-stream attention (XLNet)
    This implements a pre-LN decoder, as opposed to the post-LN default in PyTorch."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='gelu', layer_norm_eps=1e-5):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_q = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_c = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = transformer._get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.gelu
        super().__setstate__(state)

    def forward_stream(
        self,
        tgt: Tensor,
        tgt_norm: Tensor,
        tgt_kv: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor],
        tgt_key_padding_mask: Optional[Tensor],
    ):
        """Forward pass for a single stream (i.e. content or query)
        tgt_norm is just a LayerNorm'd tgt. Added as a separate parameter for efficiency.
        Both tgt_kv and memory are expected to be LayerNorm'd too.
        memory is LayerNorm'd by ViT.
        """
        tgt2, sa_weights = self.self_attn(
            tgt_norm, tgt_kv, tgt_kv, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout1(tgt2)

        tgt2, ca_weights = self.cross_attn(self.norm1(tgt), memory, memory)
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(tgt)))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, sa_weights, ca_weights

    def forward(
        self,
        query,
        content,
        memory,
        query_mask: Optional[Tensor] = None,
        content_mask: Optional[Tensor] = None,
        content_key_padding_mask: Optional[Tensor] = None,
        update_content: bool = True,
    ):
        query_norm = self.norm_q(query)
        content_norm = self.norm_c(content)
        query = self.forward_stream(query, query_norm, content_norm, memory, query_mask, content_key_padding_mask)[0]
        if update_content:
            content = self.forward_stream(
                content, content_norm, content_norm, memory, content_mask, content_key_padding_mask
            )[0]
        return query, content


class Decoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm):
        super().__init__()
        self.layers = transformer._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        query,
        content,
        memory,
        query_mask: Optional[Tensor] = None,
        content_mask: Optional[Tensor] = None,
        content_key_padding_mask: Optional[Tensor] = None,
    ):
        for i, mod in enumerate(self.layers):
            last = i == len(self.layers) - 1
            query, content = mod(
                query, content, memory, query_mask, content_mask, content_key_padding_mask, update_content=not last
            )
        query = self.norm(query)
        return query


class Encoder(VisionTransformer):

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        embed_layer=PatchEmbed,
    ):
        super().__init__(
            img_size,
            patch_size,
            in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            embed_layer=embed_layer,
            num_classes=0,  # These
            global_pool='',  # disable the
            class_token=False,  # classifier head.
        )

    def forward(self, x):
        # Return all tokens
        return self.forward_features(x)


class HybridBlock(nn.Module):
    """Hybrid Transformer Block with parallel / residual Depthwise Convolution.
    Combines local spatial mixing (DWConv 3x3) with global contextual mixing (MHSA) and FFN,
    as recommended in sections 7 & 8 of the hybrid architecture design.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        spatial_size: Sequence[int] = (8, 16),
        norm_layer: Optional[nn.Module] = None,
        act_layer: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.spatial_size = spatial_size
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.transformer_block = Block(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer or nn.LayerNorm,
            act_layer=act_layer or nn.GELU,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        H, W = self.spatial_size
        if H * W == N:
            feat = x.transpose(1, 2).reshape(B, C, H, W)
            feat = self.dwconv(feat)
            x = x + feat.flatten(2).transpose(1, 2)
        return self.transformer_block(x)


class ConvBlock(nn.Module):
    """Depthwise-Separable Inverted Residual Conv Block for pure CNN stages."""

    def __init__(self, dim: int, mlp_ratio: float = 2.0, drop: float = 0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.conv = nn.Sequential(
            # DWConv 3x3 local mixing
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            # 1x1 PWConv expansion
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            # 1x1 PWConv projection
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.drop = nn.Dropout2d(drop) if drop > 0.0 else nn.Identity()

        # Kaiming init
        nn.init.kaiming_normal_(self.conv[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv[3].weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv[6].weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.conv(x))


class HybridEncoder(nn.Module):
    """Hybrid CNN + Transformer (or Pure CNN) Encoder for PARSeq.

    Architecture pipeline:
      Input (B, 3, H, W) [e.g. 32x128]
              │
              ▼
      Conv 3×3 stride 2×2 -> 16×64 (stem)
              │
              ▼
      CNN Backbone (RepViT, MobileNetV3, ResNet, etc.) -> 8×32
              │
              ▼
      DWConv 3×3 stride 1×2 -> 8×16 × C
              │
              ▼
      1×1 Conv C -> embed_dim (e.g. 384) -> 8×16 × embed_dim
              │
              ▼
      CNN Blocks (cnn_depth layers of ConvBlock: DWConv 3x3 + PWConv)
              │
              ▼
      Flatten -> 128 × embed_dim
              │
              ▼
      2D Positional Embedding
              │
              ▼
      Transformer / Hybrid Encoder (enc_depth layers; if enc_depth == 0, pure CNN!)
              │
              ▼
      LayerNorm -> Decoder
    """

    def __init__(
        self,
        img_size: Sequence[int] = (32, 128),
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 0,
        cnn_depth: int = 0,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        backbone: str = 'repvit_m0_9',
        pretrained_backbone: bool = False,
        backbone_out_idx: Optional[int] = None,
        dw_stride: Sequence[int] = (1, 2),
        pos_embed_type: str = 'learned',
        block_type: str = 'transformer',
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Optional[nn.Module] = None,
        act_layer: Optional[nn.Module] = None,
    ):
        super().__init__()
        aliases = {
            'repvit': 'repvit_m0_9',
            'repvit_m0': 'repvit_m0_9',
            'repvit_m0_9': 'repvit_m0_9',
            'repvit_m1': 'repvit_m1_0',
            'repvit_m1_0': 'repvit_m1_0',
            'mobilenetv3': 'mobilenetv3_small_050',
            'mobilenetv3_small': 'mobilenetv3_small_050',
            'mobilenetv3_small_050': 'mobilenetv3_small_050',
            'mobilenetv3_large': 'mobilenetv3_large_100',
            'mobilenetv3_large_100': 'mobilenetv3_large_100',
            'repvgg': 'repvgg_a0',
            'repvgg_a0': 'repvgg_a0',
            'repvgg_a1': 'repvgg_a1',
            'ghostnet': 'ghostnet_100',
            'ghostnet_100': 'ghostnet_100',
            'resnet18': 'resnet18',
            'resnet34': 'resnet34',
            'mobilenetv2': 'mobilenetv2_100',
            'mobilenetv2_100': 'mobilenetv2_100',
        }
        resolved_backbone = aliases.get(str(backbone).lower(), backbone)

        if str(resolved_backbone).lower() in ('stem', 'conv_stem', 'none_stem'):
            in_channels = 64
            reduction = 4
            self.backbone = nn.Sequential(
                nn.Conv2d(in_chans, 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.GELU(),
                nn.Conv2d(32, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.GELU(),
            )
            nn.init.kaiming_normal_(self.backbone[0].weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.backbone[3].weight, mode='fan_out', nonlinearity='relu')
        else:
            # Inspect backbone feature info to get reduction 4 stage (e.g. 32x128 -> 8x32)
            probe_model = timm.create_model(resolved_backbone, features_only=True, in_chans=in_chans)
            if backbone_out_idx is None:
                red4 = [i for i, info in enumerate(probe_model.feature_info.info) if info.get('reduction') == 4]
                out_idx = red4[0] if red4 else 0
            else:
                out_idx = backbone_out_idx

            in_channels = probe_model.feature_info.info[out_idx]['num_chs']
            reduction = probe_model.feature_info.info[out_idx].get('reduction', 4)
            del probe_model

            self.backbone = timm.create_model(
                resolved_backbone,
                features_only=True,
                out_indices=(out_idx,),
                pretrained=pretrained_backbone,
                in_chans=in_chans,
            )

        dw_stride = tuple(dw_stride)
        self.dw_stride = dw_stride
        self.reduction = reduction

        # DWConv 3x3 with stride (1, 2)
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=dw_stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
        )

        # 1x1 Conv C -> embed_dim
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        # Stack of CNN blocks (cnn_depth layers of ConvBlock: DWConv 3x3 + PWConv)
        self.cnn_depth = cnn_depth
        if cnn_depth > 0:
            self.cnn_blocks = nn.Sequential(
                *[ConvBlock(embed_dim, mlp_ratio=2.0, drop=drop_rate) for _ in range(cnn_depth)]
            )
        else:
            self.cnn_blocks = nn.Identity()

        # Weight initialization for DWConv and 1x1 Conv
        nn.init.kaiming_normal_(self.dwconv[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.ones_(self.dwconv[1].weight)
        nn.init.zeros_(self.dwconv[1].bias)
        nn.init.kaiming_normal_(self.proj[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.ones_(self.proj[1].weight)
        nn.init.zeros_(self.proj[1].bias)

        # Spatial dimensions for 2D Positional Embedding
        feat_h = img_size[0] // reduction // dw_stride[0]
        feat_w = img_size[1] // reduction // dw_stride[1]
        self.feat_shape = (feat_h, feat_w)
        num_patches = feat_h * feat_w
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.pos_embed_type = pos_embed_type
        self.block_type = block_type

        # 2D Positional Embedding
        if pos_embed_type == 'sine':
            try:
                pos_emb = build_sincos2d_pos_embed((feat_h, feat_w), embed_dim).unsqueeze(0)
            except Exception:
                pos_emb = torch.zeros(1, num_patches, embed_dim)
            self.register_buffer('pos_embed', pos_emb)
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            try:
                pos_emb = build_sincos2d_pos_embed((feat_h, feat_w), embed_dim).unsqueeze(0)
                self.pos_embed.data.copy_(pos_emb)
            except Exception:
                nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer / Hybrid Encoder blocks (depth == 0 means Pure CNN)
        self.depth = depth
        norm_layer = norm_layer or nn.LayerNorm
        act_layer = act_layer or nn.GELU
        if depth > 0:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
            blocks = []
            for i in range(depth):
                if block_type == 'hybrid':
                    blocks.append(
                        HybridBlock(
                            dim=embed_dim,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            proj_drop=drop_rate,
                            attn_drop=attn_drop_rate,
                            drop_path=dpr[i],
                            spatial_size=(feat_h, feat_w),
                            norm_layer=norm_layer,
                            act_layer=act_layer,
                        )
                    )
                else:
                    blocks.append(
                        Block(
                            dim=embed_dim,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            proj_drop=drop_rate,
                            attn_drop=attn_drop_rate,
                            drop_path=dpr[i],
                            norm_layer=norm_layer,
                            act_layer=act_layer,
                        )
                    )
            self.blocks = nn.Sequential(*blocks)
        else:
            self.blocks = nn.Identity()

        self.norm = norm_layer(embed_dim)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        if isinstance(feat, (list, tuple)):
            feat = feat[0]
        feat = self.dwconv(feat)
        feat = self.proj(feat)
        feat = self.cnn_blocks(feat)

        B, C, H_f, W_f = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)  # (B, N, embed_dim)

        if tokens.shape[1] != self.pos_embed.shape[1]:
            orig_h, orig_w = self.feat_shape
            pe = self.pos_embed.reshape(1, orig_h, orig_w, -1).permute(0, 3, 1, 2)
            pe = F.interpolate(pe, size=(H_f, W_f), mode='bicubic', align_corners=False)
            pe = pe.permute(0, 2, 3, 1).flatten(1, 2)
        else:
            pe = self.pos_embed

        tokens = tokens + pe
        tokens = self.pos_drop(tokens)
        tokens = self.blocks(tokens)
        tokens = self.norm(tokens)
        return tokens


class TokenEmbedding(nn.Module):

    def __init__(self, charset_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(charset_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, tokens: torch.Tensor):
        return math.sqrt(self.embed_dim) * self.embedding(tokens)
