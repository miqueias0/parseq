# Scene Text Recognition Model Hub - SmoothQuant for PARSeq
# Mitigates activation outliers by migrating difficulty from activations to weights
# Reference: SmoothQuant (Xiao et al., 2023) and I-BERT (Kim et al., 2021)

import copy
from typing import Optional
import torch
import torch.nn as nn

from .calibrator import ActivationCalibrator


def compute_smooth_scale(
    act_max: torch.Tensor,
    weight: torch.Tensor,
    alpha: float = 0.5,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Computes channel-wise smoothing scale: s_j = (max|X_j|)^alpha / (max|W_j|)^(1-alpha).
    
    Args:
        act_max: Channel-wise max absolute activations [in_features].
        weight: Linear weight tensor [out_features, in_features].
        alpha: Migration parameter (0 = only weigh weights, 1 = only weigh activations, 0.5 = balanced).
        eps: Epsilon to prevent division by zero.
        
    Returns:
        Smoothing scale vector s [in_features].
    """
    device = weight.device
    act_max = act_max.to(device).float()
    w_max = torch.amax(torch.abs(weight.float()), dim=0).clamp(min=eps)
    act_max = act_max.clamp(min=eps)

    s = (torch.pow(act_max, alpha) / torch.pow(w_max, 1.0 - alpha)).clamp(min=1e-4)
    return s


def apply_smoothquant_to_parseq(
    model: nn.Module,
    calibrator: Optional[ActivationCalibrator] = None,
    alpha: float = 0.5,
    inplace: bool = False,
) -> nn.Module:
    """Applies SmoothQuant parameter transformation to PARSeq.
    
    Absorbs smoothing scales into LayerNorm parameters (weight and bias)
    and rescales downstream Linear weights.
    
    Args:
        model: PARSeq model (or System containing .model).
        calibrator: ActivationCalibrator instance containing collected statistics.
        alpha: Smoothing hyperparameter (default: 0.5).
        inplace: Whether to modify model in place or return a copy.
        
    Returns:
        Smoothed model.
    """
    m = model if inplace else copy.deepcopy(model)
    inner = getattr(m, "model", m)

    # 1. Smooth Encoder ViT Blocks
    if hasattr(inner, "encoder") and hasattr(inner.encoder, "blocks"):
        for i, block in enumerate(inner.encoder.blocks):
            # LN1 -> QKV
            if hasattr(block, "norm1") and hasattr(block, "attn") and hasattr(block.attn, "qkv"):
                ln = block.norm1
                qkv = block.attn.qkv
                stats = calibrator.get_layer_stats(f"encoder.blocks.{i}.norm1") if calibrator else None
                act_max = stats["ch_max"] if stats else torch.ones(ln.normalized_shape[0], device=ln.weight.device)
                act_max = act_max.to(ln.weight.device)
                s = compute_smooth_scale(act_max, qkv.weight, alpha=alpha)

                with torch.no_grad():
                    ln.weight.div_(s)
                    if ln.bias is not None:
                        ln.bias.div_(s)
                    qkv.weight.mul_(s.unsqueeze(0))

            # LN2 -> FC1
            if hasattr(block, "norm2") and hasattr(block, "mlp") and hasattr(block.mlp, "fc1"):
                ln = block.norm2
                fc1 = block.mlp.fc1
                stats = calibrator.get_layer_stats(f"encoder.blocks.{i}.norm2") if calibrator else None
                act_max = stats["ch_max"] if stats else torch.ones(ln.normalized_shape[0], device=ln.weight.device)
                act_max = act_max.to(ln.weight.device)
                s = compute_smooth_scale(act_max, fc1.weight, alpha=alpha)

                with torch.no_grad():
                    ln.weight.div_(s)
                    if ln.bias is not None:
                        ln.bias.div_(s)
                    fc1.weight.mul_(s.unsqueeze(0))

    # 2. Smooth Encoder Final Norm -> Decoder Cross Attention KV
    if hasattr(inner, "encoder") and hasattr(inner.encoder, "norm"):
        enc_norm = inner.encoder.norm
        stats = calibrator.get_layer_stats("encoder.norm") if calibrator else None
        act_max = stats["ch_max"] if stats else torch.ones(enc_norm.normalized_shape[0], device=enc_norm.weight.device)
        act_max = act_max.to(enc_norm.weight.device)

        if hasattr(inner, "decoder") and hasattr(inner.decoder, "layers"):
            for layer in inner.decoder.layers:
                if hasattr(layer, "cross_attn") and hasattr(layer.cross_attn, "in_proj_weight"):
                    embed_dim = enc_norm.normalized_shape[0]
                    # cross_attn in_proj_weight is [3*embed_dim, embed_dim] for Q, K, V
                    # In cross attention, memory is used for K and V (rows embed_dim to 3*embed_dim)
                    kv_weight = layer.cross_attn.in_proj_weight[embed_dim:, :]
                    s_kv = compute_smooth_scale(act_max, kv_weight, alpha=alpha)

                    with torch.no_grad():
                        enc_norm.weight.div_(s_kv)
                        if enc_norm.bias is not None:
                            enc_norm.bias.div_(s_kv)
                        layer.cross_attn.in_proj_weight[embed_dim:, :].mul_(s_kv.unsqueeze(0))
                    break

    # 3. Smooth Decoder Layers: LN2 -> Linear1
    if hasattr(inner, "decoder") and hasattr(inner.decoder, "layers"):
        for i, layer in enumerate(inner.decoder.layers):
            if hasattr(layer, "norm2") and hasattr(layer, "linear1"):
                ln = layer.norm2
                l1 = layer.linear1
                stats = calibrator.get_layer_stats(f"decoder.layers.{i}.norm2") if calibrator else None
                act_max = stats["ch_max"] if stats else torch.ones(ln.normalized_shape[0], device=ln.weight.device)
                act_max = act_max.to(ln.weight.device)
                s = compute_smooth_scale(act_max, l1.weight, alpha=alpha)

                with torch.no_grad():
                    ln.weight.div_(s)
                    if ln.bias is not None:
                        ln.bias.div_(s)
                    l1.weight.mul_(s.unsqueeze(0))

    return m
