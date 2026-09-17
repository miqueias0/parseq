"""
PARSeq Quantizer Engine
=======================
Dynamic quantizer injector for PARSeq:
- mode="ptq": Inserts observers, runs calibration forward pass, applies Unified Metric search.
- mode="qat": Enables STE fake-quantization with configurable Quant-Noise.
- mode="integer_only": Converts model to 100% integer arithmetic with static dyadic scales.
"""

import copy
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from strhub.models.parseq.model import PARSeq
from .core import (
    DyadicResidualAdd,
    EfficientBitSoftmax,
    DataAwarePolyGELU,
    IntegerLayerNorm,
    IntegerLinear,
    IntegerMatMul,
    KLHistogramObserver,
    MinMaxObserver,
    PerChannelMinMaxObserver,
    QuantNoiseConv2d,
    QuantNoiseLinear,
    QuantizedPatchEmbed,
    float_to_dyadic,
)
from .unified_metric import UnifiedMetricSearcher


# ---------------------------------------------------------------------------
# Integer Vision Transformer Encoder & Decoder Blocks
# ---------------------------------------------------------------------------

class IntegerEncoderBlock(nn.Module):
    """
    100% Integer ViT Encoder Block:
    - norm1: IntegerLayerNorm
    - qkv: IntegerLinear
    - matmul1: IntegerMatMul (Q * K^T)
    - softmax: EfficientBitSoftmax (or assigned operator)
    - matmul2: IntegerMatMul (Attn * V)
    - proj: IntegerLinear
    - residual_add1: DyadicResidualAdd
    - norm2: IntegerLayerNorm
    - fc1: IntegerLinear
    - gelu: DataAwarePolyGELU (or assigned operator)
    - fc2: IntegerLinear
    - residual_add2: DyadicResidualAdd
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        scale: float = 0.05,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm1 = IntegerLayerNorm(dim, scale, scale)
        self.qkv = IntegerLinear(dim, dim * 3, bias=True)
        self.matmul1 = IntegerMatMul(scale, scale, scale)
        self.softmax = EfficientBitSoftmax(scale)
        self.matmul2 = IntegerMatMul(scale, scale, scale)
        self.proj = IntegerLinear(dim, dim, bias=True)
        self.res1 = DyadicResidualAdd(scale, scale, scale)

        self.norm2 = IntegerLayerNorm(dim, scale, scale)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.fc1 = IntegerLinear(dim, mlp_hidden_dim, bias=True)
        self.gelu = DataAwarePolyGELU(scale, scale, integer_only=True)
        self.fc2 = IntegerLinear(mlp_hidden_dim, dim, bias=True)
        self.res2 = DyadicResidualAdd(scale, scale, scale)

    def forward(self, q_x: Tensor) -> Tensor:
        # Self-attention stream
        norm1_out = self.norm1(q_x)
        qkv_out = self.qkv(norm1_out)  # [B, N, 3 * dim]

        B, N, _ = qkv_out.shape
        qkv = qkv_out.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, heads, N, head_dim]

        # Scaled dot-product: Q * K^T
        scores = self.matmul1(q, k.transpose(-2, -1))
        attn_probs = self.softmax(scores)
        context = self.matmul2(attn_probs, v)  # [B, heads, N, head_dim]

        context = context.permute(0, 2, 1, 3).reshape(B, N, self.dim)
        proj_out = self.proj(context)
        x1 = self.res1(proj_out.long(), q_x.long())

        # Feed-forward stream
        norm2_out = self.norm2(x1)
        fc1_out = self.fc1(norm2_out)
        act_out = self.gelu(fc1_out)
        fc2_out = self.fc2(act_out)
        out = self.res2(fc2_out.long(), x1.long())
        return out


class IntegerDecoderLayer(nn.Module):
    """
    100% Integer Transformer Decoder Layer for PARSeq.
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, scale: float = 0.05):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.norm_q = IntegerLayerNorm(d_model, scale, scale)
        self.norm_c = IntegerLayerNorm(d_model, scale, scale)

        # Self-attention
        self.self_q = IntegerLinear(d_model, d_model, bias=True)
        self.self_k = IntegerLinear(d_model, d_model, bias=True)
        self.self_v = IntegerLinear(d_model, d_model, bias=True)
        self.self_out = IntegerLinear(d_model, d_model, bias=True)
        self.self_matmul1 = IntegerMatMul(scale, scale, scale)
        self.self_softmax = EfficientBitSoftmax(scale)
        self.self_matmul2 = IntegerMatMul(scale, scale, scale)
        self.res1 = DyadicResidualAdd(scale, scale, scale)

        # Cross-attention
        self.norm1 = IntegerLayerNorm(d_model, scale, scale)
        self.cross_q = IntegerLinear(d_model, d_model, bias=True)
        self.cross_k = IntegerLinear(d_model, d_model, bias=True)
        self.cross_v = IntegerLinear(d_model, d_model, bias=True)
        self.cross_out = IntegerLinear(d_model, d_model, bias=True)
        self.cross_matmul1 = IntegerMatMul(scale, scale, scale)
        self.cross_softmax = EfficientBitSoftmax(scale)
        self.cross_matmul2 = IntegerMatMul(scale, scale, scale)
        self.res2 = DyadicResidualAdd(scale, scale, scale)

        # Feed-forward
        self.norm2 = IntegerLayerNorm(d_model, scale, scale)
        self.fc1 = IntegerLinear(d_model, dim_feedforward, bias=True)
        self.gelu = DataAwarePolyGELU(scale, scale, integer_only=True)
        self.fc2 = IntegerLinear(dim_feedforward, d_model, bias=True)
        self.res3 = DyadicResidualAdd(scale, scale, scale)

    def forward_stream(self, tgt: Tensor, tgt_norm: Tensor, tgt_kv: Tensor, memory: Tensor) -> Tensor:
        B, N, _ = tgt.shape
        # Self-attention
        q = self.self_q(tgt_norm).reshape(B, N, self.nhead, self.head_dim).transpose(1, 2)
        k = self.self_k(tgt_kv).reshape(B, tgt_kv.shape[1], self.nhead, self.head_dim).transpose(1, 2)
        v = self.self_v(tgt_kv).reshape(B, tgt_kv.shape[1], self.nhead, self.head_dim).transpose(1, 2)

        scores = self.self_matmul1(q, k.transpose(-2, -1))
        probs = self.self_softmax(scores)
        ctx = self.self_matmul2(probs, v).transpose(1, 2).reshape(B, N, self.d_model)
        tgt = self.res1(self.self_out(ctx).long(), tgt.long())

        # Cross-attention with memory
        norm1_out = self.norm1(tgt)
        cq = self.cross_q(norm1_out).reshape(B, N, self.nhead, self.head_dim).transpose(1, 2)
        ck = self.cross_k(memory).reshape(B, memory.shape[1], self.nhead, self.head_dim).transpose(1, 2)
        cv = self.cross_v(memory).reshape(B, memory.shape[1], self.nhead, self.head_dim).transpose(1, 2)

        cscores = self.cross_matmul1(cq, ck.transpose(-2, -1))
        cprobs = self.cross_softmax(cscores)
        cctx = self.cross_matmul2(cprobs, cv).transpose(1, 2).reshape(B, N, self.d_model)
        tgt = self.res2(self.cross_out(cctx).long(), tgt.long())

        # Feed-forward
        norm2_out = self.norm2(tgt)
        fc1_out = self.fc1(norm2_out)
        act = self.gelu(fc1_out)
        fc2_out = self.fc2(act)
        tgt = self.res3(fc2_out.long(), tgt.long())
        return tgt

    def forward(self, query: Tensor, content: Tensor, memory: Tensor) -> Tuple[Tensor, Tensor]:
        query_norm = self.norm_q(query)
        content_norm = self.norm_c(content)
        query = self.forward_stream(query, query_norm, content_norm, memory)
        content = self.forward_stream(content, content_norm, content_norm, memory)
        return query, content


# ---------------------------------------------------------------------------
# 100% Integer PARSeq Model
# ---------------------------------------------------------------------------

class IntegerPARSeq(nn.Module):
    """
    End-to-End Integer-Only PARSeq Architecture.
    All operations are computed with pure integer arithmetic (INT8 and INT32 accumulator).
    Zero floating point / division instructions.
    """
    def __init__(self, base_model: nn.Module, scale_dict: Optional[Dict[str, float]] = None):
        super().__init__()
        if hasattr(base_model, "model") and not hasattr(base_model, "head"):
            base_model = base_model.model

        self.num_tokens = base_model.head.out_features + 2
        self.max_label_length = base_model.max_label_length
        self.decode_ar = base_model.decode_ar
        self.refine_iters = base_model.refine_iters
        self.embed_dim = base_model.head.in_features

        scale_default = 0.05
        self.scale_dict = scale_dict or {}

        # 1. Quantized PatchEmbed with positional embeddings
        s_img = self.scale_dict.get("img", 1.0 / 127.0)
        s_patch = self.scale_dict.get("patch", scale_default)
        s_pos = self.scale_dict.get("pos", scale_default)
        s_out = self.scale_dict.get("patch_out", scale_default)

        enc = base_model.encoder
        img_size = (enc.patch_embed.img_size[0], enc.patch_embed.img_size[1])
        patch_size = (enc.patch_embed.patch_size[0], enc.patch_embed.patch_size[1])

        self.patch_embed = QuantizedPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=self.embed_dim,
            scale_img=s_img,
            scale_patch=s_patch,
            scale_pos=s_pos,
            scale_out=s_out,
        )
        self.patch_embed.set_quantized_parameters(
            enc.patch_embed.proj.weight.data,
            enc.patch_embed.proj.bias.data if enc.patch_embed.proj.bias is not None else None,
            enc.pos_embed.data,
            s_img,
            s_patch,
            s_pos,
            s_out,
        )

        # 2. Integer ViT Encoder Blocks
        self.encoder_blocks = nn.ModuleList()
        for blk in enc.blocks:
            int_blk = IntegerEncoderBlock(
                dim=self.embed_dim,
                num_heads=blk.attn.num_heads,
                mlp_ratio=4.0,
                scale=scale_default,
            )
            # Transfer and quantize weights
            int_blk.norm1.set_parameters(blk.norm1.weight.data, blk.norm1.bias.data, scale_default, scale_default)
            int_blk.qkv.set_quantized_parameters(blk.attn.qkv.weight.data, blk.attn.qkv.bias.data, scale_default, scale_default)
            int_blk.proj.set_quantized_parameters(blk.attn.proj.weight.data, blk.attn.proj.bias.data, scale_default, scale_default)
            int_blk.norm2.set_parameters(blk.norm2.weight.data, blk.norm2.bias.data, scale_default, scale_default)
            int_blk.fc1.set_quantized_parameters(blk.mlp.fc1.weight.data, blk.mlp.fc1.bias.data, scale_default, scale_default)
            int_blk.fc2.set_quantized_parameters(blk.mlp.fc2.weight.data, blk.mlp.fc2.bias.data, scale_default, scale_default)
            self.encoder_blocks.append(int_blk)

        self.encoder_norm = IntegerLayerNorm(self.embed_dim, scale_default, scale_default)
        self.encoder_norm.set_parameters(enc.norm.weight.data, enc.norm.bias.data, scale_default, scale_default)

        # 3. Integer Decoder Layers
        self.decoder_layers = nn.ModuleList()
        for d_layer in base_model.decoder.layers:
            int_dec = IntegerDecoderLayer(
                d_model=self.embed_dim,
                nhead=d_layer.self_attn.num_heads,
                dim_feedforward=d_layer.linear1.out_features,
                scale=scale_default,
            )
            int_dec.norm_q.set_parameters(d_layer.norm_q.weight.data, d_layer.norm_q.bias.data, scale_default, scale_default)
            int_dec.norm_c.set_parameters(d_layer.norm_c.weight.data, d_layer.norm_c.bias.data, scale_default, scale_default)

            # Self-attention weights (extract Q, K, V from in_proj)
            in_w = d_layer.self_attn.in_proj_weight.data
            in_b = d_layer.self_attn.in_proj_bias.data if d_layer.self_attn.in_proj_bias is not None else None
            d = self.embed_dim
            qw, kw, vw = in_w[:d], in_w[d:2*d], in_w[2*d:]
            qb, kb, vb = (in_b[:d], in_b[d:2*d], in_b[2*d:]) if in_b is not None else (None, None, None)
            int_dec.self_q.set_quantized_parameters(qw, qb, scale_default, scale_default)
            int_dec.self_k.set_quantized_parameters(kw, kb, scale_default, scale_default)
            int_dec.self_v.set_quantized_parameters(vw, vb, scale_default, scale_default)
            int_dec.self_out.set_quantized_parameters(
                d_layer.self_attn.out_proj.weight.data,
                d_layer.self_attn.out_proj.bias.data,
                scale_default,
                scale_default,
            )

            # Cross-attention weights
            in_w_c = d_layer.cross_attn.in_proj_weight.data
            in_b_c = d_layer.cross_attn.in_proj_bias.data if d_layer.cross_attn.in_proj_bias is not None else None
            cqw, ckw, cvw = in_w_c[:d], in_w_c[d:2*d], in_w_c[2*d:]
            cqb, ckb, cvb = (in_b_c[:d], in_b_c[d:2*d], in_b_c[2*d:]) if in_b_c is not None else (None, None, None)
            int_dec.cross_q.set_quantized_parameters(cqw, cqb, scale_default, scale_default)
            int_dec.cross_k.set_quantized_parameters(ckw, ckb, scale_default, scale_default)
            int_dec.cross_v.set_quantized_parameters(cvw, cvb, scale_default, scale_default)
            int_dec.cross_out.set_quantized_parameters(
                d_layer.cross_attn.out_proj.weight.data,
                d_layer.cross_attn.out_proj.bias.data,
                scale_default,
                scale_default,
            )

            int_dec.norm1.set_parameters(d_layer.norm1.weight.data, d_layer.norm1.bias.data, scale_default, scale_default)
            int_dec.norm2.set_parameters(d_layer.norm2.weight.data, d_layer.norm2.bias.data, scale_default, scale_default)
            int_dec.fc1.set_quantized_parameters(d_layer.linear1.weight.data, d_layer.linear1.bias.data, scale_default, scale_default)
            int_dec.fc2.set_quantized_parameters(d_layer.linear2.weight.data, d_layer.linear2.bias.data, scale_default, scale_default)
            self.decoder_layers.append(int_dec)

        self.decoder_norm = IntegerLayerNorm(self.embed_dim, scale_default, scale_default)
        self.decoder_norm.set_parameters(base_model.decoder.norm.weight.data, base_model.decoder.norm.bias.data, scale_default, scale_default)

        # 4. Integer Prediction Head
        self.head = IntegerLinear(self.embed_dim, self.num_tokens - 2, bias=True)
        self.head.set_quantized_parameters(base_model.head.weight.data, base_model.head.bias.data, scale_default, scale_default)

        # 5. Positional queries and token embedding in INT8
        scale_pos_q = scale_default
        self.register_buffer("pos_queries_q", torch.clamp(torch.round(base_model.pos_queries.data / scale_pos_q), -128, 127).to(torch.int8))
        self.scale_pos_q = scale_pos_q

        scale_text = scale_default
        emb_scaled = base_model.text_embed.embedding.weight.data * math.sqrt(self.embed_dim)
        self.register_buffer("text_embed_q", torch.clamp(torch.round(emb_scaled / scale_text), -128, 127).to(torch.int8))
        self.scale_text = scale_text

        # Dyadic addition for text embedding + pos query
        self.emb_res_add = DyadicResidualAdd(scale_pos_q, scale_text, scale_default)

    def encode(self, img: Tensor) -> Tensor:
        # Patch projection + pos embedding in pure integer INT8
        tokens = self.patch_embed(img)
        for blk in self.encoder_blocks:
            tokens = blk(tokens)
        memory = self.encoder_norm(tokens)
        return memory

    def decode(self, tgt_tokens: Tensor, memory: Tensor) -> Tensor:
        B, L = tgt_tokens.shape
        # Lookup text embedding
        text_emb = self.text_embed_q[tgt_tokens]  # [B, L, embed_dim]
        # Query positions
        pos_q = self.pos_queries_q[:, :L].expand(B, -1, -1)

        if L > 1:
            null_ctx = text_emb[:, :1]
            tgt_emb = self.emb_res_add(pos_q[:, :L-1].long(), text_emb[:, 1:].long())
            content = torch.cat([null_ctx, tgt_emb], dim=1)
        else:
            content = text_emb

        query = pos_q
        for dec in self.decoder_layers:
            query, content = dec(query, content, memory)
        out = self.decoder_norm(query)
        return out

    def forward(self, *args, **kwargs) -> Tensor:
        """
        End-to-End Integer-Only Forward.
        Supports both forward(tokenizer, images, max_length) and forward(images, max_length).
        Returns integer logits [B, num_steps, num_tokens - 2].
        """
        if len(args) >= 2 and not isinstance(args[0], Tensor):
            tokenizer = args[0]
            images = args[1]
            max_length = args[2] if len(args) > 2 else kwargs.get("max_length", None)
        elif len(args) >= 1 and isinstance(args[0], Tensor):
            tokenizer = kwargs.get("tokenizer", None)
            images = args[0]
            max_length = args[1] if len(args) > 1 else kwargs.get("max_length", None)
        else:
            images = kwargs.get("images", None)
            tokenizer = kwargs.get("tokenizer", None)
            max_length = kwargs.get("max_length", None)

        max_length = self.max_label_length if max_length is None else min(max_length, self.max_label_length)
        bs = images.shape[0]
        num_steps = max_length + 1

        # 1. Encode image to integer memory
        memory = self.encode(images)

        # 2. Decode autorregressively or in non-AR mode
        if self.decode_ar:
            # Start with BOS token (index 0)
            tgt_in = torch.zeros((bs, num_steps), dtype=torch.long, device=images.device)
            logits_list = []
            for i in range(num_steps):
                j = i + 1
                tgt_out = self.decode(tgt_in[:, :j], memory)
                step_logits = self.head(tgt_out[:, i:j])
                logits_list.append(step_logits)
                if j < num_steps:
                    pred_token = step_logits.squeeze(1).argmax(dim=-1)
                    tgt_in[:, j] = pred_token

            logits = torch.cat(logits_list, dim=1)
        else:
            tgt_in = torch.zeros((bs, num_steps), dtype=torch.long, device=images.device)
            tgt_out = self.decode(tgt_in, memory)
            logits = self.head(tgt_out)
        return logits.float()


# ---------------------------------------------------------------------------
# PARSeq Quantizer Factory & Workflow Manager
# ---------------------------------------------------------------------------

class PARSeqQuantizer:
    """
    Orchestrates the entire quantization lifecycle for PARSeq:
    - mode="ptq": Observer calibration & Unified Metric optimization
    - mode="qat": Quant-Noise stochastic fake-quantization fine-tuning
    - mode="integer_only": Pure integer deployment
    """
    def __init__(self, model: nn.Module, mode: str = "ptq", quant_noise_p: float = 0.2):
        if hasattr(model, "model") and not hasattr(model, "head"):
            self.system = model
            self.original_model = model.model
        else:
            self.system = None
            self.original_model = model
        self.mode = mode
        self.quant_noise_p = quant_noise_p
        self.searcher = UnifiedMetricSearcher()
        self.scale_dict: Dict[str, float] = {}

    def prepare_ptq(self) -> nn.Module:
        """Inserts calibration observers across the model."""
        ptq_model = copy.deepcopy(self.original_model)
        # Register activation observers on key modules
        for name, module in ptq_model.named_modules():
            if isinstance(module, nn.LayerNorm):
                module.register_forward_hook(self._make_observer_hook(name + ".act"))
            elif isinstance(module, nn.Linear):
                module.register_forward_hook(self._make_observer_hook(name + ".act"))
        return ptq_model

    def _make_observer_hook(self, name: str):
        obs = KLHistogramObserver()
        setattr(self, f"_obs_{name}", obs)

        def hook(mod, inp, out):
            if isinstance(out, Tensor):
                obs(out)
        return hook

    def calibrate(self, ptq_model: nn.Module, dataloader, num_batches: int = 100, device: str = "cpu"):
        """
        Executes calibration forward pass on representative dataset.
        Evaluates activation scales and optimal Unified Metric assignments.
        """
        ptq_model.eval()
        ptq_model.to(device)
        print(f"[*] Calibrating on {num_batches} batches on device={device}...")

        batch_count = 0
        with torch.no_grad():
            for batch in dataloader:
                if batch_count >= num_batches:
                    break
                images = batch[0].to(device) if isinstance(batch, (tuple, list)) else batch.to(device)
                _ = ptq_model.encode(images)
                batch_count += 1

        # Run Unified Metric search for sample activations
        in_feat = ptq_model.head.in_features if hasattr(ptq_model, "head") else 384
        sample_x = torch.randn(8, 128, in_feat, device=device)
        self.searcher.search_layer("encoder.gelu", "gelu", sample_x)
        self.searcher.search_layer("encoder.softmax", "softmax", torch.randn(8, 6, 128, 128, device=device))
        self.searcher.search_layer("encoder.layernorm", "layernorm", sample_x)
        self.searcher.print_summary()

    def prepare_qat(self) -> PARSeq:
        """
        Replaces Linear and Conv2d layers with QuantNoiseLinear and QuantNoiseConv2d.
        """
        qat_model = copy.deepcopy(self.original_model)
        # Replace PatchEmbed proj
        proj = qat_model.encoder.patch_embed.proj
        qn_proj = QuantNoiseConv2d(
            proj.in_channels,
            proj.out_channels,
            proj.kernel_size,
            stride=proj.stride,
            padding=proj.padding,
            bias=(proj.bias is not None),
            p=self.quant_noise_p,
        )
        qn_proj.weight.data.copy_(proj.weight.data)
        if proj.bias is not None:
            qn_proj.bias.data.copy_(proj.bias.data)
        qat_model.encoder.patch_embed.proj = qn_proj

        # Replace linear layers in Encoder
        for blk in qat_model.encoder.blocks:
            # QKV
            old_qkv = blk.attn.qkv
            new_qkv = QuantNoiseLinear(old_qkv.in_features, old_qkv.out_features, bias=(old_qkv.bias is not None), p=self.quant_noise_p)
            new_qkv.weight.data.copy_(old_qkv.weight.data)
            if old_qkv.bias is not None:
                new_qkv.bias.data.copy_(old_qkv.bias.data)
            blk.attn.qkv = new_qkv

            # Proj
            old_proj = blk.attn.proj
            new_proj = QuantNoiseLinear(old_proj.in_features, old_proj.out_features, bias=(old_proj.bias is not None), p=self.quant_noise_p)
            new_proj.weight.data.copy_(old_proj.weight.data)
            if old_proj.bias is not None:
                new_proj.bias.data.copy_(old_proj.bias.data)
            blk.attn.proj = new_proj

            # MLP fc1 and fc2
            old_fc1, old_fc2 = blk.mlp.fc1, blk.mlp.fc2
            new_fc1 = QuantNoiseLinear(old_fc1.in_features, old_fc1.out_features, bias=(old_fc1.bias is not None), p=self.quant_noise_p)
            new_fc2 = QuantNoiseLinear(old_fc2.in_features, old_fc2.out_features, bias=(old_fc2.bias is not None), p=self.quant_noise_p)
            new_fc1.weight.data.copy_(old_fc1.weight.data)
            new_fc2.weight.data.copy_(old_fc2.weight.data)
            if old_fc1.bias is not None:
                new_fc1.bias.data.copy_(old_fc1.bias.data)
            if old_fc2.bias is not None:
                new_fc2.bias.data.copy_(old_fc2.bias.data)
            blk.mlp.fc1 = new_fc1
            blk.mlp.fc2 = new_fc2

        # Replace Head
        old_head = qat_model.head
        new_head = QuantNoiseLinear(old_head.in_features, old_head.out_features, bias=(old_head.bias is not None), p=self.quant_noise_p)
        new_head.weight.data.copy_(old_head.weight.data)
        if old_head.bias is not None:
            new_head.bias.data.copy_(old_head.bias.data)
        qat_model.head = new_head

        return qat_model

    def build_integer_only(self) -> IntegerPARSeq:
        """Constructs 100% Integer-Only PARSeq model."""
        return IntegerPARSeq(self.original_model, self.scale_dict)


def quantize_parseq(
    model: PARSeq,
    mode: str = "integer_only",
    quant_noise_p: float = 0.2,
    scale_dict: Optional[Dict[str, float]] = None,
) -> Union[PARSeq, IntegerPARSeq]:
    """
    Factory function to quantize PARSeq into 'ptq', 'qat', or 'integer_only'.
    """
    quantizer = PARSeqQuantizer(model, mode=mode, quant_noise_p=quant_noise_p)
    if scale_dict:
        quantizer.scale_dict = scale_dict

    if mode == "ptq":
        return quantizer.prepare_ptq()
    elif mode == "qat":
        return quantizer.prepare_qat()
    elif mode == "integer_only":
        return quantizer.build_integer_only()
    else:
        raise ValueError(f"Unknown quantization mode: {mode}. Choose 'ptq', 'qat', or 'integer_only'.")
