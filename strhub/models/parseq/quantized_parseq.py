import copy
import os
import json
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from strhub.models.parseq.model import PARSeq
from strhub.quant.quant_utils import quantize_symmetric, dequantize_symmetric, quantize_naive
from strhub.quant.integer_gelu import IBERTGELU, IViTGELU, IPTQDataAwarePolyGELU, GELUFP32
from strhub.quant.integer_softmax import IBERTSoftmax, IViTShiftmax, IPTQBitSoftmax, SoftmaxFP32
from strhub.quant.integer_layernorm import IBERTLayerNorm, IPTQLayerNorm, LayerNormFP32
from strhub.quant.int_flashattention import INTFlashAttention
from strhub.quant.sage_attention import SageAttention
from strhub.quant.plugins.trt_plugins import (
    IntegerLayerNormPluginWrapper,
    IntegerGELUPluginWrapper,
    IntegerSoftmaxPluginWrapper,
)


class ONNXQDQ(torch.autograd.Function):
    """Inserts explicit ONNX QuantizeLinear and DequantizeLinear nodes during export for TensorRT 11 INT8 fusion."""
    @staticmethod
    def forward(ctx, x, scale, axis=0):
        if x.dim() == 2 and axis == 0 and scale.numel() > 1:
            s = scale.view(-1, 1)
        else:
            s = scale
        q = torch.clamp(torch.round(x / s), -128, 127)
        return q * s

    @staticmethod
    def symbolic(g, x, scale, axis=0):
        zp_shape = scale.type().sizes() if hasattr(scale, "type") and hasattr(scale.type(), "sizes") else ()
        zero_point = g.op("Constant", value_t=torch.zeros(zp_shape, dtype=torch.int8))
        q = g.op("QuantizeLinear", x, scale, zero_point, axis_i=axis)
        deq = g.op("DequantizeLinear", q, scale, zero_point, axis_i=axis)
        return deq


class QuantizedLinear(nn.Module):
    """Linear layer supporting:
    - naive: direct truncation negative control (no dynamic range scaling)
    - conventional_ptq: per-channel INT8 weights + calibrated/dynamic activation quantization
    - integer_only: INT8 weights/activations for integer-only inference
    - qat: Straight-Through Estimator (STE) for quantization-aware training
    """
    global_export_qdq: bool = False

    def __init__(
        self,
        original_linear: nn.Linear,
        mode: str = "conventional_ptq",
        bits: int = 8,
        channel_wise_weight: bool = True
    ):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.mode = mode
        self.bits = bits
        self.channel_wise_weight = channel_wise_weight
        self.export_qdq = False
        requires_grad = (mode == "qat")

        # Buffers for scales
        self.register_buffer("weight_scale", torch.tensor(1.0))
        self.register_buffer("weight_scale_1d", torch.tensor([1.0], dtype=torch.float32))
        self.register_buffer("act_scale", torch.tensor(0.05, dtype=torch.float32))
        self.calibrated = False

        qmin = -(1 << (bits - 1))
        qmax = (1 << (bits - 1)) - 1

        with torch.no_grad():
            w_data = original_linear.weight.data.clone()
            if mode == "naive":
                q_w, _ = quantize_naive(w_data, bits=self.bits)
                self.weight = nn.Parameter(q_w, requires_grad=requires_grad)
                self.weight_scale.fill_(1.0)
                self.weight_scale_1d = torch.tensor([1.0], dtype=torch.float32)
                self.calibrated = True
            elif mode in ["conventional_ptq", "integer_only", "qat"]:
                if channel_wise_weight and w_data.dim() >= 2:
                    s_w = w_data.abs().amax(dim=1, keepdim=True) / qmax
                    self.weight_scale_1d = (w_data.abs().amax(dim=1) / qmax).clamp(min=1e-8)
                else:
                    s_w = w_data.abs().max() / qmax
                    self.weight_scale_1d = (s_w.unsqueeze(0)).clamp(min=1e-8)
                s_w = torch.clamp(s_w, min=1e-8)
                q_w = torch.clamp(torch.round(w_data / s_w), qmin, qmax)
                # Store dequantized weight so functional operations (including MHA internal calls) have correct scale
                self.weight = nn.Parameter(q_w * s_w, requires_grad=requires_grad)
                self.register_buffer("q_w", q_w)
                self.weight_scale = s_w
                # self.calibrated remains False until calibration scales are loaded or calibrated
            else:
                self.weight = nn.Parameter(w_data, requires_grad=requires_grad)

        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone(), requires_grad=requires_grad)
        else:
            self.bias = None

    def recompute_weight_scale(self):
        qmax = (1 << (self.bits - 1)) - 1
        with torch.no_grad():
            w_data = self.weight.data
            if self.channel_wise_weight and w_data.dim() >= 2:
                s_w = w_data.abs().amax(dim=1, keepdim=True) / qmax
                self.weight_scale_1d = (w_data.abs().amax(dim=1) / qmax).clamp(min=1e-8)
            else:
                s_w = w_data.abs().max() / qmax
                self.weight_scale_1d = (s_w.unsqueeze(0)).clamp(min=1e-8)
            self.weight_scale = torch.clamp(s_w, min=1e-8)

    def set_activation_scale(self, scale: float):
        self.act_scale = torch.tensor(scale, dtype=torch.float32, device=self.weight.device)
        self.calibrated = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (getattr(self, "export_qdq", False) or QuantizedLinear.global_export_qdq) and self.mode in ["conventional_ptq", "integer_only", "qat"]:
            w_q = ONNXQDQ.apply(self.weight, self.weight_scale_1d, 0)
            s_act = self.act_scale.squeeze()
            x_q = ONNXQDQ.apply(x, s_act, 1)
            return F.linear(x_q, w_q, self.bias)

        qmin = -(1 << (self.bits - 1))
        qmax = (1 << (self.bits - 1)) - 1

        if self.mode == "naive":
            # Negative control: truncate activation directly without dynamic scaling
            q_x, _ = quantize_naive(x, bits=self.bits)
            return F.linear(q_x, self.weight, self.bias)

        elif self.mode in ["conventional_ptq", "integer_only"]:
            if self.calibrated:
                s_x = self.act_scale
            else:
                # Dynamic per-token or per-tensor scale fallback
                s_x = torch.clamp(x.abs().amax(dim=-1, keepdim=True) / qmax, min=1e-8)
            q_x = torch.clamp(torch.round(x / s_x), qmin, qmax)
            deq_x = q_x * s_x
            return F.linear(deq_x, self.weight, self.bias)

        elif self.mode == "qat":
            # Straight-Through Estimator (STE) for QAT
            # Weight quantization with STE:
            if self.channel_wise_weight and self.weight.dim() >= 2:
                s_w = self.weight.abs().amax(dim=1, keepdim=True) / qmax
            else:
                s_w = self.weight.abs().max() / qmax
            s_w = torch.clamp(s_w, min=1e-8)
            w_q = quantize_naive(self.weight, bits=self.bits)
            return F.linear(x, w_q, self.bias)

        elif self.mode in ["conventional_ptq", "integer_only", "qat"]:
            # Quantize weights per-channel
            w_q, _ = quantize_symmetric(self.weight, self.weight_scale, bits=self.bits)
            w_deq = dequantize_symmetric(w_q, self.weight_scale)

            # Quantize activations per-tensor if calibrated
            if self.calibrated or self.mode == "qat":
                x_q, _ = quantize_symmetric(x, self.activation_scale, bits=self.bits)
                x = dequantize_symmetric(x_q, self.activation_scale)

            return F.linear(x, w_deq, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)


class AttentionSoftmaxWrapper(nn.Module):
    """Wraps timm Attention to use candidate integer-only softmax, fused MHA, or INT-FlashAttention (arXiv:2409.16997v2)."""
    def __init__(
        self,
        original_attn,
        softmax_module: nn.Module,
        fuse_mha: bool = False,
        use_int_flashattention: bool = False,
        use_sage_attention: bool = False,
        sage_mode: str = "sageattn_b",
        block_r: int = 64,
        block_c: int = 64,
        bits: int = 8,
        v_quant_mode: str = "per_tensor",
        use_plugin: bool = False,
    ):
        super().__init__()
        if isinstance(original_attn, AttentionSoftmaxWrapper):
            self.attn = original_attn.attn
        else:
            self.attn = original_attn
        self.softmax = softmax_module
        self.fuse_mha = fuse_mha
        self.use_int_flashattention = use_int_flashattention
        self.use_sage_attention = use_sage_attention
        self.sage_mode = sage_mode
        self.v_quant_mode = v_quant_mode
        self.use_plugin = use_plugin

        # Disable fused_attn so explicit execution takes place
        if hasattr(self.attn, "fused_attn"):
            self.attn.fused_attn = False

        embed_dim = self.attn.attn_dim if hasattr(self.attn, "attn_dim") else self.attn.qkv.out_features // 3
        if self.use_sage_attention:
            self.sage_attn = SageAttention(
                embed_dim=embed_dim,
                num_heads=self.attn.num_heads,
                block_r=block_r,
                block_c=block_c,
                bits=bits,
                mode=sage_mode,
                smooth=True,
                use_plugin=use_plugin,
            )
        else:
            self.sage_attn = None

        if self.use_int_flashattention:
            self.int_flash_attn = INTFlashAttention(
                embed_dim=embed_dim,
                num_heads=self.attn.num_heads,
                block_r=block_r,
                block_c=block_c,
                bits=bits,
                v_quant_mode=v_quant_mode,
                use_plugin=use_plugin,
            )
        else:
            self.int_flash_attn = None

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if "attn" in self.__dict__:
                return getattr(self.attn, name)
            raise

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, is_causal: bool = False) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.attn.qkv(x).reshape(B, N, 3, self.attn.num_heads, self.attn.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.attn.q_norm(q), self.attn.k_norm(k)

        if self.use_sage_attention and self.sage_attn is not None:
            # SageAttention (arXiv:2410.02367v9)
            # Smooth K, INT8 GEMMs, Online Softmax
            x = self.sage_attn(q, k, v, attn_mask=attn_mask)
        elif self.use_int_flashattention and self.int_flash_attn is not None:
            # INT-FlashAttention (arXiv:2409.16997v2)
            # Fully INT8 Q@K^T, online softmax, and INT8 Attn@V fused in SRAM
            x = self.int_flash_attn(q, k, v, attn_mask=attn_mask)
        elif self.fuse_mha:
            # Canonical Softmax pattern allowing TensorRT to fuse entire MHA block into FMHA kernel
            q = q * self.attn.scale
            attn = q @ k.transpose(-2, -1)
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = F.softmax(attn, dim=-1)
            attn = self.attn.attn_drop(attn)
            x = attn @ v
        else:
            # Execute chosen candidate integer Softmax (e.g. IPTQBitSoftmax polynomial unrolling)
            q = q * self.attn.scale
            attn = q @ k.transpose(-2, -1)
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = self.softmax(attn)
            attn = self.attn.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, self.attn.attn_dim)
        x = self.attn.norm(x)
        x = self.attn.proj(x)
        x = self.attn.proj_drop(x)
        return x


def replace_linear_modules(module: nn.Module, mode: str, bits: int = 8) -> nn.Module:
    """Recursively replace nn.Linear modules with QuantizedLinear."""
    for name, child in list(module.named_children()):
        if isinstance(child, QuantizedLinear):
            child.mode = mode
            child.bits = bits
        elif isinstance(child, nn.Linear):
            setattr(module, name, QuantizedLinear(child, mode=mode, bits=bits))
        else:
            replace_linear_modules(child, mode=mode, bits=bits)
    return module


def replace_decoder_attention_weights(decoder: nn.Module, mode: str, bits: int = 8):
    """Quantize in_proj_weight of nn.MultiheadAttention in Decoder layers."""
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    if not hasattr(decoder, "layers"):
        return
    for layer in decoder.layers:
        for attn in [layer.self_attn, layer.cross_attn]:
            if hasattr(attn, "in_proj_weight") and attn.in_proj_weight is not None:
                w = attn.in_proj_weight.data
                s_w = torch.clamp(w.abs().amax(dim=1, keepdim=True) / qmax, min=1e-8)
                q_w = torch.clamp(torch.round(w / s_w), qmin, qmax)
                attn.in_proj_weight.data.copy_(q_w * s_w)


def replace_nonlinear_modules(
    module: nn.Module,
    gelu_name: str = "gelu_iptq",
    softmax_name: str = "softmax_iptq",
    layernorm_name: str = "layernorm_ibert",
    assignment: Optional[Dict[str, str]] = None,
    fuse_mha: bool = False,
    fuse_mlp: bool = False,
    fuse_layernorm: bool = False,
    use_int_flashattention: bool = False,
    use_sage_attention: bool = False,
    sage_mode: str = "sageattn_b",
    v_quant_mode: str = "per_tensor",
    use_plugin: bool = False,
) -> nn.Module:
    """Replace activation functions and LayerNorms with chosen integer-only approximations,
    or with canonical fused primitives when flags fuse_mha, fuse_mlp, fuse_layernorm,
    use_int_flashattention (arXiv:2409.16997v2) or use_sage_attention (arXiv:2410.02367v9) are active."""
    gelu_map = {
        "gelu_fp32": GELUFP32,
        "gelu_ibert": IBERTGELU,
        "gelu_ivit": IViTGELU,
        "gelu_iptq": IPTQDataAwarePolyGELU,
        "gelu_plugin": IntegerGELUPluginWrapper,
    }
    softmax_map = {
        "softmax_fp32": SoftmaxFP32,
        "softmax_ibert": IBERTSoftmax,
        "softmax_ivit": IViTShiftmax,
        "softmax_iptq": IPTQBitSoftmax,
        "softmax_plugin": IntegerSoftmaxPluginWrapper,
    }
    layernorm_map = {
        "layernorm_fp32": LayerNormFP32,
        "layernorm_ibert": IBERTLayerNorm,
        "layernorm_iptq": IPTQLayerNorm,
        "layernorm_plugin": IntegerLayerNormPluginWrapper,
    }

    if use_plugin and not use_int_flashattention and not use_sage_attention:
        softmax_name = "softmax_plugin"
    if use_plugin and not fuse_mlp:
        gelu_name = "gelu_plugin"
    if use_plugin and not fuse_layernorm:
        layernorm_name = "layernorm_plugin"

    if fuse_mlp:
        gelu_name = "gelu_fp32"
    if fuse_layernorm:
        layernorm_name = "layernorm_fp32"

    device = next(module.parameters()).device

    # 1. Replace in decoder layers
    if hasattr(module, "decoder") and hasattr(module.decoder, "layers"):
        for i, layer in enumerate(module.decoder.layers):
            layer_key_gelu = f"decoder.layer_{i}.gelu"
            layer_key_ln = f"decoder.layer_{i}.layernorm"
            chosen_gelu = "gelu_fp32" if fuse_mlp else (assignment or {}).get(layer_key_gelu, gelu_name)
            chosen_ln = "layernorm_fp32" if fuse_layernorm else (assignment or {}).get(layer_key_ln, layernorm_name)

            if hasattr(layer, "activation"):
                layer.activation = gelu_map.get(chosen_gelu, IPTQDataAwarePolyGELU)()

            # Replace all 4 LayerNorms in each Decoder layer
            for ln_attr in ["norm1", "norm2", "norm_q", "norm_c"]:
                if hasattr(layer, ln_attr):
                    old_ln = getattr(layer, ln_attr)
                    ln_dim = old_ln.normalized_shape[0] if isinstance(old_ln.normalized_shape, (tuple, list)) else old_ln.normalized_shape
                    new_ln = layernorm_map.get(chosen_ln, IBERTLayerNorm)(ln_dim, eps=old_ln.eps).to(device)
                    new_ln.weight.data.copy_(old_ln.weight.data)
                    new_ln.bias.data.copy_(old_ln.bias.data)
                    setattr(layer, ln_attr, new_ln)

        # Decoder final norm
        if hasattr(module.decoder, "norm") and module.decoder.norm is not None:
            old_ln = module.decoder.norm
            ln_dim = old_ln.normalized_shape[0] if isinstance(old_ln.normalized_shape, (tuple, list)) else old_ln.normalized_shape
            chosen_ln = "layernorm_fp32" if fuse_layernorm else layernorm_name
            new_ln = layernorm_map.get(chosen_ln, IBERTLayerNorm)(ln_dim, eps=old_ln.eps).to(device)
            new_ln.weight.data.copy_(old_ln.weight.data)
            new_ln.bias.data.copy_(old_ln.bias.data)
            module.decoder.norm = new_ln

    # 2. Replace in encoder blocks
    if hasattr(module, "encoder"):
        if hasattr(module.encoder, "blocks"):
            for i, block in enumerate(module.encoder.blocks):
                block_key_gelu = f"encoder.block_{i}.gelu"
                block_key_ln = f"encoder.block_{i}.layernorm"
                block_key_sm = f"encoder.block_{i}.softmax"
                chosen_gelu = "gelu_fp32" if fuse_mlp else (assignment or {}).get(block_key_gelu, gelu_name)
                chosen_ln = "layernorm_fp32" if fuse_layernorm else (assignment or {}).get(block_key_ln, layernorm_name)
                chosen_sm = (assignment or {}).get(block_key_sm, softmax_name)

                # MLP activation (GELU)
                if hasattr(block, "mlp") and hasattr(block.mlp, "act"):
                    block.mlp.act = gelu_map.get(chosen_gelu, IPTQDataAwarePolyGELU)()

                # Norm1 and Norm2
                for ln_attr in ["norm1", "norm2"]:
                    if hasattr(block, ln_attr):
                        old_ln = getattr(block, ln_attr)
                        ln_dim = old_ln.normalized_shape[0] if isinstance(old_ln.normalized_shape, (tuple, list)) else old_ln.normalized_shape
                        new_ln = layernorm_map.get(chosen_ln, IBERTLayerNorm)(ln_dim, eps=old_ln.eps).to(device)
                        new_ln.weight.data.copy_(old_ln.weight.data)
                        new_ln.bias.data.copy_(old_ln.bias.data)
                        setattr(block, ln_attr, new_ln)

                # Attention Softmax / INT-FlashAttention / SageAttention
                if hasattr(block, "attn"):
                    sm_mod = softmax_map.get(chosen_sm, IPTQBitSoftmax)(dim=-1)
                    block.attn = AttentionSoftmaxWrapper(
                        block.attn,
                        sm_mod,
                        fuse_mha=fuse_mha,
                        use_int_flashattention=use_int_flashattention,
                        use_sage_attention=use_sage_attention,
                        sage_mode=sage_mode,
                        v_quant_mode=v_quant_mode,
                        use_plugin=use_plugin,
                    )

        # Encoder final norm
        if hasattr(module.encoder, "norm") and module.encoder.norm is not None:
            old_ln = module.encoder.norm
            ln_dim = old_ln.normalized_shape[0] if isinstance(old_ln.normalized_shape, (tuple, list)) else old_ln.normalized_shape
            chosen_ln = "layernorm_fp32" if fuse_layernorm else layernorm_name
            new_ln = layernorm_map.get(chosen_ln, IBERTLayerNorm)(ln_dim, eps=old_ln.eps).to(device)
            new_ln.weight.data.copy_(old_ln.weight.data)
            new_ln.bias.data.copy_(old_ln.bias.data)
            module.encoder.norm = new_ln

    return module


def load_calibration_into_model(model: nn.Module, calibration_json_path: str, sample_size: str = "256"):
    """Loads calibrated activation scales from calibration_stats.json into QuantizedLinear modules."""
    if not os.path.exists(calibration_json_path):
        return
    with open(calibration_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Get sample size entry or first available
    stats = data.get(str(sample_size)) or (list(data.values())[0] if data else None)
    if not stats or "layers" not in stats:
        return

    layers_dict = stats["layers"]
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            clean_name = name.replace(".attn.attn.", ".attn.")
            wrapped_name = name.replace(".attn.", ".attn.attn.")
            scale_val = None
            if name in layers_dict:
                scale_val = layers_dict[name].get("scale")
            elif clean_name in layers_dict:
                scale_val = layers_dict[clean_name].get("scale")
            elif wrapped_name in layers_dict:
                scale_val = layers_dict[wrapped_name].get("scale")

            if scale_val:
                module.set_activation_scale(scale_val)


def create_model_variant(
    variant: str,
    base_model: PARSeq,
    assignment: Optional[Dict[str, str]] = None,
    gelu_candidate: str = "gelu_iptq",
    softmax_candidate: str = "softmax_iptq",
    layernorm_candidate: str = "layernorm_ibert",
    calibration_file: Optional[str] = None,
    fuse_mha: bool = False,
    fuse_mlp: bool = False,
    fuse_layernorm: bool = False,
    use_int_flashattention: bool = False,
    use_sage_attention: bool = False,
    sage_mode: str = "sageattn_b",
    v_quant_mode: str = "per_tensor",
    use_plugin: bool = False,
) -> nn.Module:
    """Build a specific model variant from the evaluation matrix:
    - M0: PARSeq FP32 AR (Autoregressive decoding, refine_iters=1)
    - M1: PARSeq FP32 NAR (Non-autoregressive, decode_ar=False, refine_iters=0)
    - M2: PARSeq FP16 NAR
    - M3: PARSeq Naive INT8 NAR (Negative Control)
    - M4: PARSeq W8A8 Conventional PTQ
    - M5: PARSeq INT8 Integer-Only PTQ (Unified Metric / IPTQ-ViT)
    - M6: PARSeq INT8 Integer-Only QAT
    """
    variant = variant.lower().strip()
    model = copy.deepcopy(base_model)

    if variant == "m0":
        model.decode_ar = True
        model.refine_iters = 1
        return model.float()

    # All variants M1-M6 are strictly Non-Autoregressive (NAR)
    model.decode_ar = False
    model.refine_iters = 0

    if variant == "m1":
        return model.float()

    if variant == "m2":
        return model.half()

    if variant == "m3":
        # M3: Naive INT8 NAR (Negative control)
        replace_linear_modules(model, mode="naive", bits=8)
        return model

    if variant == "m4":
        # M4: Conventional W8A8 PTQ (INT8 Linear, FP32 nonlinearities)
        replace_linear_modules(model, mode="conventional_ptq", bits=8)
        replace_decoder_attention_weights(model.decoder, mode="conventional_ptq", bits=8)
        if calibration_file and os.path.exists(calibration_file):
            load_calibration_into_model(model, calibration_file)
        return model

    if variant == "m5":
        # M5: Integer-Only PTQ
        replace_linear_modules(model, mode="integer_only", bits=8)
        replace_decoder_attention_weights(model.decoder, mode="integer_only", bits=8)
        replace_nonlinear_modules(
            model,
            gelu_name=gelu_candidate,
            softmax_name=softmax_candidate,
            layernorm_name=layernorm_candidate,
            assignment=assignment,
            fuse_mha=fuse_mha,
            fuse_mlp=fuse_mlp,
            fuse_layernorm=fuse_layernorm,
            use_int_flashattention=use_int_flashattention,
            use_sage_attention=use_sage_attention,
            sage_mode=sage_mode,
            v_quant_mode=v_quant_mode,
            use_plugin=use_plugin,
        )
        if calibration_file and os.path.exists(calibration_file):
            load_calibration_into_model(model, calibration_file)
        return model

    if variant == "m6":
        # M6: Integer-Only QAT
        replace_linear_modules(model, mode="qat", bits=8)
        replace_decoder_attention_weights(model.decoder, mode="qat", bits=8)
        replace_nonlinear_modules(
            model,
            gelu_name=gelu_candidate,
            softmax_name=softmax_candidate,
            layernorm_name=layernorm_candidate,
            assignment=assignment,
            fuse_mha=fuse_mha,
            fuse_mlp=fuse_mlp,
            fuse_layernorm=fuse_layernorm,
            use_int_flashattention=use_int_flashattention,
            use_sage_attention=use_sage_attention,
            sage_mode=sage_mode,
            v_quant_mode=v_quant_mode,
            use_plugin=use_plugin,
        )
        if calibration_file and os.path.exists(calibration_file):
            load_calibration_into_model(model, calibration_file)
        return model

    raise ValueError(f"Unknown variant '{variant}'. Expected one of: m0, m1, m2, m3, m4, m5, m6")
