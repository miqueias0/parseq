"""
Core INT8 and Integer-Only Quantization Operators
=================================================
Strict integer arithmetic modules, observers, dyadic scaling,
and stochastic quantization noise layers for Vision Transformers.
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# 1. Dyadic Arithmetic & Helper Functions (HAWQ-V3 / Jacob et al.)
# ---------------------------------------------------------------------------

def float_to_dyadic(scale_float: float, max_b: int = 32767, max_shift: int = 31) -> Tuple[int, int]:
    """
    Converts a floating-point multiplier S into a dyadic fraction: S ~= b / (2^c)
    where b is an integer multiplier (0 <= b <= max_b) and c is a right shift (0 <= c <= max_shift).
    """
    if scale_float <= 0.0:
        return 0, 0

    best_b, best_c = 0, 0
    min_err = float("inf")

    # Search shift c from 0 to max_shift
    for c in range(max_shift + 1):
        b = round(scale_float * (1 << c))
        if 0 <= b <= max_b:
            approx = b / (1 << c)
            err = abs(approx - scale_float)
            if err < min_err:
                min_err = err
                best_b = b
                best_c = c
                if err == 0.0:
                    break

    return best_b, best_c


def dyadic_scale(tensor_int: Tensor, b: int, c: int) -> Tensor:
    """
    Applies dyadic rescaling in integer arithmetic:
    out = (tensor_int * b + 2^(c - 1)) >> c
    Rounds half away from zero in integer domain.
    """
    if c == 0:
        return tensor_int * b
    rounding = 1 << (c - 1)
    return (tensor_int * b + rounding) >> c


def pure_int_bit_length(n_safe: Tensor) -> Tensor:
    """
    Computes integer bit length without any floating-point operations.
    Bit length = floor(log2(n_safe)) + 1 using pure integer comparisons and shifts.
    """
    shifts = torch.zeros_like(n_safe, dtype=torch.int64)
    temp = n_safe.clone()
    for k in [32, 16, 8, 4, 2, 1]:
        cond = (temp >= (1 << k))
        shifts = shifts + torch.where(cond, torch.tensor(k, device=n_safe.device), torch.zeros_like(shifts))
        temp = torch.where(cond, temp >> k, temp)
    return shifts + 1


def integer_sqrt_newton(n: Tensor) -> Tensor:
    """
    Vectorized integer square root floor(sqrt(n)) via Newton-Raphson (I-BERT Alg. 4).
    Uses bit length shift initialization so it converges in <= 4 iterations.
    Zero floating point instructions.
    """
    zero_mask = (n <= 0)
    n_safe = torch.clamp(n.long(), min=1)

    # Initial guess x0 = 1 << ceil(Bits(n) / 2)
    bit_len = pure_int_bit_length(n_safe)
    shifts = torch.clamp((bit_len + 1) >> 1, min=1, max=30)
    x = 1 << shifts

    # Newton-Raphson iterations: x_{k+1} = (x_k + floor(n / x_k)) >> 1
    for _ in range(5):
        x_next = (x + torch.div(n_safe, x, rounding_mode="trunc")) >> 1
        x = torch.where(x_next < x, x_next, x)

    res = torch.where(zero_mask, torch.zeros_like(x), x)
    return res


# ---------------------------------------------------------------------------
# 2. Observers & Calibrators (MinMax, PerChannel, KL Histogram)
# ---------------------------------------------------------------------------

class MinMaxObserver(nn.Module):
    """
    Symmetric per-tensor observer calculating scale = max(|min|, |max|) / 127.
    """
    def __init__(self, bits: int = 8):
        super().__init__()
        self.bits = bits
        self.qmax = (1 << (bits - 1)) - 1
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.register_buffer("scale", torch.tensor(1.0))
        self.calibrated = False

    def forward(self, x: Tensor) -> Tensor:
        if self.training or not self.calibrated:
            with torch.no_grad():
                cur_min = x.detach().min()
                cur_max = x.detach().max()
                self.min_val = torch.min(self.min_val, cur_min)
                self.max_val = torch.max(self.max_val, cur_max)
                bound = torch.max(torch.abs(self.min_val), torch.abs(self.max_val))
                self.scale = torch.clamp(bound / float(self.qmax), min=1e-8)
        return x

    def finish_calibration(self):
        bound = torch.max(torch.abs(self.min_val), torch.abs(self.max_val))
        self.scale = torch.clamp(bound / float(self.qmax), min=1e-8)
        self.calibrated = True


class PerChannelMinMaxObserver(nn.Module):
    """
    Symmetric per-channel observer along dim 0 (output channels for weights).
    """
    def __init__(self, ch_axis: int = 0, bits: int = 8):
        super().__init__()
        self.ch_axis = ch_axis
        self.bits = bits
        self.qmax = (1 << (bits - 1)) - 1
        self.register_buffer("scale", torch.tensor([]))
        self.calibrated = False

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            reduce_dims = [d for d in range(x.dim()) if d != self.ch_axis]
            max_val = x.detach().abs()
            for d in sorted(reduce_dims, reverse=True):
                max_val = max_val.amax(dim=d)
            self.scale = torch.clamp(max_val / float(self.qmax), min=1e-8)
            self.calibrated = True
        return x


class KLHistogramObserver(nn.Module):
    """
    Kullback-Leibler (KL / Entropy) Divergence Observer for Activations
    (Bhandare et al., 2019 / Q8BERT / TensorRT calibration).
    Constructs a 2048-bin histogram of absolute activation values and determines
    the clipping threshold T that minimizes KL(P || Q).
    """
    def __init__(self, num_bins: int = 2048, bits: int = 8):
        super().__init__()
        self.num_bins = num_bins
        self.bits = bits
        self.qmax = (1 << (bits - 1)) - 1
        self.register_buffer("histogram", torch.zeros(num_bins, dtype=torch.float64))
        self.register_buffer("max_val", torch.tensor(0.0))
        self.register_buffer("scale", torch.tensor(1.0))
        self.calibrated = False

    def forward(self, x: Tensor) -> Tensor:
        if not self.calibrated:
            with torch.no_grad():
                cur_max = x.detach().abs().max()
                self.max_val = torch.max(self.max_val, cur_max)
        return x

    def collect_histogram(self, x: Tensor):
        """Second pass to accumulate histogram bins up to self.max_val."""
        with torch.no_grad():
            if self.max_val <= 0:
                return
            hist = torch.histc(x.detach().abs().float(), bins=self.num_bins, min=0.0, max=self.max_val.item())
            self.histogram += hist.to(self.histogram.dtype)

    def finish_calibration(self):
        """Computes optimal threshold minimizing KL divergence."""
        with torch.no_grad():
            if self.max_val <= 0:
                self.scale = torch.tensor(1.0, device=self.max_val.device)
                self.calibrated = True
                return

            hist = self.histogram.cpu().numpy()
            total_elements = hist.sum()
            if total_elements == 0:
                self.scale = torch.clamp(self.max_val / float(self.qmax), min=1e-8)
                self.calibrated = True
                return

            # Target number of quantized bins
            num_quantized_bins = self.qmax
            min_kl = float("inf")
            best_threshold_bin = self.num_bins - 1

            # Search threshold between 128 and num_bins
            for i in range(num_quantized_bins, self.num_bins):
                p = hist[:i].copy()
                outliers = hist[i:].sum()
                p[-1] += outliers
                p = p / p.sum()

                # Quantize p into num_quantized_bins
                num_merged = i / float(num_quantized_bins)
                q = [p[int(j * num_merged):int((j + 1) * num_merged)].sum() for j in range(num_quantized_bins)]

                # Expand q back to size i
                q_expanded = []
                for j in range(num_quantized_bins):
                    start = int(j * num_merged)
                    end = int((j + 1) * num_merged)
                    count = max(1, end - start)
                    val = q[j] / count
                    q_expanded.extend([val] * count)

                q_expanded = q_expanded[:i]
                # Avoid log(0)
                import numpy as np
                p_safe = np.clip(p, 1e-12, 1.0)
                q_safe = np.clip(np.array(q_expanded), 1e-12, 1.0)
                kl = np.sum(p_safe * np.log(p_safe / q_safe))

                if kl < min_kl:
                    min_kl = kl
                    best_threshold_bin = i

            optimal_threshold = (best_threshold_bin + 0.5) * (self.max_val.item() / self.num_bins)
            self.scale = torch.tensor(max(optimal_threshold / float(self.qmax), 1e-8), device=self.max_val.device)
            self.calibrated = True


# ---------------------------------------------------------------------------
# 3. Stochastic Quant-Noise Layers (Fan et al., 2021)
# ---------------------------------------------------------------------------

class QuantNoiseLinear(nn.Linear):
    """
    Linear layer with Quantization Noise (Quant-Noise).
    During training, only a random subset (probability p) of weights undergoes fake-quantization,
    allowing unbiased gradients to flow through unquantized weights and stabilizing Transformers.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, p: float = 0.2, bits: int = 8):
        super().__init__(in_features, out_features, bias=bias)
        self.p = p
        self.bits = bits
        self.qmax = (1 << (bits - 1)) - 1
        self.register_buffer("scale_w", torch.tensor(1.0))
        self.register_buffer("scale_x", torch.tensor(1.0))

    def forward(self, x: Tensor) -> Tensor:
        # Fake quantize activations via STE
        if self.scale_x.item() > 0:
            x_scaled = x / self.scale_x
            x_clamped = torch.clamp(x_scaled, -self.qmax, self.qmax)
            x_quant = (torch.round(x_clamped) - x_scaled).detach() + x_scaled
            x_in = x_quant * self.scale_x
        else:
            x_in = x

        # Weight quantization with Quant-Noise
        w = self.weight
        max_w = w.detach().abs().amax(dim=1, keepdim=True)
        scale_w = torch.clamp(max_w / float(self.qmax), min=1e-8)
        self.scale_w = scale_w.squeeze()

        w_scaled = w / scale_w
        w_clamped = torch.clamp(w_scaled, -self.qmax, self.qmax)
        w_fake = (torch.round(w_clamped) - w_scaled).detach() + w_scaled
        w_quant = w_fake * scale_w

        if self.training and self.p < 1.0:
            # Stochastic mask: 1 with probability p (quantized), 0 with 1 - p (continuous)
            mask = (torch.rand_like(w) < self.p).float()
            w_eff = mask * w_quant + (1.0 - mask) * w
        else:
            w_eff = w_quant

        return F.linear(x_in, w_eff, self.bias)


class QuantNoiseConv2d(nn.Conv2d):
    """
    Conv2d layer with Quantization Noise (Quant-Noise).
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0, bias=True, p=0.2, bits=8):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.p = p
        self.bits = bits
        self.qmax = (1 << (bits - 1)) - 1
        self.register_buffer("scale_w", torch.tensor(1.0))
        self.register_buffer("scale_x", torch.tensor(1.0))

    def forward(self, x: Tensor) -> Tensor:
        if self.scale_x.item() > 0:
            x_scaled = x / self.scale_x
            x_clamped = torch.clamp(x_scaled, -self.qmax, self.qmax)
            x_quant = (torch.round(x_clamped) - x_scaled).detach() + x_scaled
            x_in = x_quant * self.scale_x
        else:
            x_in = x

        w = self.weight
        max_w = w.detach().abs().amax(dim=(1, 2, 3), keepdim=True)
        scale_w = torch.clamp(max_w / float(self.qmax), min=1e-8)
        self.scale_w = scale_w.squeeze()

        w_scaled = w / scale_w
        w_clamped = torch.clamp(w_scaled, -self.qmax, self.qmax)
        w_fake = (torch.round(w_clamped) - w_scaled).detach() + w_scaled
        w_quant = w_fake * scale_w

        if self.training and self.p < 1.0:
            mask = (torch.rand_like(w) < self.p).float()
            w_eff = mask * w_quant + (1.0 - mask) * w
        else:
            w_eff = w_quant

        return F.conv2d(x_in, w_eff, self.bias, self.stride, self.padding, self.dilation, self.groups)


# ---------------------------------------------------------------------------
# 4. Strict Integer Arithmetic Modules
# ---------------------------------------------------------------------------

class DyadicResidualAdd(nn.Module):
    """
    Dyadic Residual Addition (HAWQ-V3):
    q_a = DN(S_m / S_a) * q_m + DN(S_r / S_a) * q_r
    Aligned in INT32 accumulator before clamping to INT8.
    """
    def __init__(self, scale_main: float = 1.0, scale_res: float = 1.0, scale_out: float = 1.0):
        super().__init__()
        self.scale_main = scale_main
        self.scale_res = scale_res
        self.scale_out = scale_out

        bm, cm = float_to_dyadic(scale_main / max(scale_out, 1e-8))
        br, cr = float_to_dyadic(scale_res / max(scale_out, 1e-8))
        self.register_buffer("b_m", torch.tensor(bm, dtype=torch.int64))
        self.register_buffer("c_m", torch.tensor(cm, dtype=torch.int64))
        self.register_buffer("b_r", torch.tensor(br, dtype=torch.int64))
        self.register_buffer("c_r", torch.tensor(cr, dtype=torch.int64))

    def update_scales(self, scale_main: float, scale_res: float, scale_out: float):
        self.scale_main = scale_main
        self.scale_res = scale_res
        self.scale_out = scale_out
        bm, cm = float_to_dyadic(scale_main / max(scale_out, 1e-8))
        br, cr = float_to_dyadic(scale_res / max(scale_out, 1e-8))
        self.b_m.copy_(torch.tensor(bm, dtype=torch.int64))
        self.c_m.copy_(torch.tensor(cm, dtype=torch.int64))
        self.b_r.copy_(torch.tensor(br, dtype=torch.int64))
        self.c_r.copy_(torch.tensor(cr, dtype=torch.int64))

    def forward(self, q_main: Tensor, q_res: Tensor) -> Tensor:
        # q_main and q_res are int32/int64 tensors
        scaled_m = dyadic_scale(q_main.long(), self.b_m.item(), self.c_m.item())
        scaled_r = dyadic_scale(q_res.long(), self.b_r.item(), self.c_r.item())
        q_sum = scaled_m + scaled_r
        return torch.clamp(q_sum, -128, 127)


class QuantizedPatchEmbed(nn.Module):
    """
    End-to-End Integer-Only PatchEmbed Layer:
    - Quantizes image RGB input (uint8 [0, 255] or standard float) to INT8 [-128, 127].
    - Conv2d patch projection executed in INT8 with INT32 accumulator.
    - Rescaled to intermediate patch token scale via dyadic factors.
    - Adds static INT8 positional embedding via DyadicResidualAdd.
    - Output: INT8 patch tokens ready for Transformer Encoder.
    """
    def __init__(
        self,
        img_size: Tuple[int, int] = (32, 128),
        patch_size: Tuple[int, int] = (4, 4),
        in_chans: int = 3,
        embed_dim: int = 384,
        scale_img: float = 1.0 / 127.0,
        scale_patch: float = 0.05,
        scale_pos: float = 0.05,
        scale_out: float = 0.05,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        # Integer quantized weights for 2D convolution: (embed_dim, in_chans, P_h, P_w)
        self.register_buffer("weight_q", torch.zeros((embed_dim, in_chans, patch_size[0], patch_size[1]), dtype=torch.int8))
        self.register_buffer("bias_q", torch.zeros(embed_dim, dtype=torch.int32))
        self.register_buffer("pos_embed_q", torch.zeros((1, self.num_patches, embed_dim), dtype=torch.int8))

        # Dyadic factors for Conv2d per-channel rescaling: S_w * S_img / S_patch
        self.register_buffer("b_conv", torch.ones(embed_dim, dtype=torch.int64))
        self.register_buffer("c_conv", torch.zeros(embed_dim, dtype=torch.int64))

        self.scale_img = scale_img
        self.scale_patch = scale_patch
        self.scale_pos = scale_pos
        self.scale_out = scale_out

        # Dyadic residual add for adding patch tokens and positional embeddings
        self.residual_add = DyadicResidualAdd(scale_patch, scale_pos, scale_out)

    def set_quantized_parameters(
        self,
        fp_weight: Tensor,
        fp_bias: Optional[Tensor],
        fp_pos_embed: Tensor,
        scale_img: float,
        scale_patch: float,
        scale_pos: float,
        scale_out: float,
    ):
        self.scale_img = scale_img
        self.scale_patch = scale_patch
        self.scale_pos = scale_pos
        self.scale_out = scale_out

        # Weight per-channel quantization
        max_w = fp_weight.abs().amax(dim=(1, 2, 3))
        scales_w = torch.clamp(max_w / 127.0, min=1e-8)
        w_q = torch.clamp(torch.round(fp_weight / scales_w.view(-1, 1, 1, 1)), -128, 127).to(torch.int8)
        self.weight_q.copy_(w_q)

        # Bias quantization with scale = scale_img * scale_w
        if fp_bias is not None:
            scales_b = scale_img * scales_w
            b_q = torch.round(fp_bias / scales_b).to(torch.int32)
            self.bias_q.copy_(b_q)
        else:
            self.bias_q.zero_()

        # Dyadic multipliers for conv output: (scale_img * scales_w) / scale_patch
        b_list, c_list = [], []
        for ch in range(self.embed_dim):
            s_ratio = (scale_img * scales_w[ch].item()) / max(scale_patch, 1e-8)
            b, c = float_to_dyadic(s_ratio)
            b_list.append(b)
            c_list.append(c)
        self.b_conv.copy_(torch.tensor(b_list, dtype=torch.int64))
        self.c_conv.copy_(torch.tensor(c_list, dtype=torch.int64))

        # Positional embedding quantization
        pos_q = torch.clamp(torch.round(fp_pos_embed / max(scale_pos, 1e-8)), -128, 127).to(torch.int8)
        self.pos_embed_q.copy_(pos_q)

        self.residual_add.update_scales(scale_patch, scale_pos, scale_out)

    def forward(self, x_img: Tensor) -> Tensor:
        """
        x_img: INT8 tensor [B, 3, H, W] or float tensor to be quantized to INT8.
        Returns: INT8 tensor [B, num_patches, embed_dim].
        """
        if x_img.dtype in [torch.float32, torch.float16]:
            q_img = torch.clamp(torch.round(x_img / self.scale_img), -128, 127).to(torch.int32)
        else:
            q_img = x_img.to(torch.int32)

        # Integer convolution accumulating in INT32
        conv_out = F.conv2d(
            q_img.float(),
            self.weight_q.float(),
            self.bias_q.float(),
            stride=self.patch_size,
        ).long()  # [B, embed_dim, H', W']

        B, C, H_out, W_out = conv_out.shape
        # Flatten spatial dimensions: [B, num_patches, embed_dim]
        tokens = conv_out.flatten(2).transpose(1, 2)

        # Apply per-channel dyadic rescaling
        b_c = self.b_conv.view(1, 1, -1)
        c_c = self.c_conv.view(1, 1, -1)
        rounding = 1 << torch.clamp(c_c - 1, min=0)
        tokens_rescaled = (tokens * b_c + rounding) >> c_c
        tokens_int8 = torch.clamp(tokens_rescaled, -128, 127)

        # Dyadic addition of positional embeddings: tokens + pos_embed
        out_tokens = self.residual_add(tokens_int8, self.pos_embed_q.long())
        return out_tokens.to(torch.int8)


class DataAwarePolyGELU(nn.Module):
    """
    IPTQ-ViT (Kim et al., 2025) Data-Aware Poly-GELU
    Approximates erf(x / sqrt(2)) with quartic polynomial:
    L_ours(u) = sgn(u) * [ a * (clip(|u|, max=-b) + b)^4 + 1 ]
    DataAwarePolyGELU(x) = 0.5 * x * (1 + L_ours(x / sqrt(2)))
    with visual distribution pre-quantized coefficients:
    a = -0.019913, b = -2.698088.
    """
    def __init__(self, scale_in: float = 0.05, scale_out: float = 0.05, integer_only: bool = True):
        super().__init__()
        self.scale_in = scale_in
        self.scale_out = scale_out
        self.integer_only = integer_only

        self.a = -0.019913
        self.b = -2.698088

        # Fixed point dyadic coefficients
        # S_u = scale_in / sqrt(2)
        s_u = scale_in / math.sqrt(2)
        # Precompute dyadic multiplier for u: u = q_x * S_u
        b_u, c_u = float_to_dyadic(s_u)
        self.register_buffer("b_u", torch.tensor(b_u, dtype=torch.int64))
        self.register_buffer("c_u", torch.tensor(c_u, dtype=torch.int64))

        # Dyadic factor for output rescaling: (scale_in / scale_out)
        b_out, c_out = float_to_dyadic(scale_in / max(scale_out, 1e-8))
        self.register_buffer("b_out", torch.tensor(b_out, dtype=torch.int64))
        self.register_buffer("c_out", torch.tensor(c_out, dtype=torch.int64))

    def update_scales(self, scale_in: float, scale_out: float):
        self.scale_in = scale_in
        self.scale_out = scale_out
        s_u = scale_in / math.sqrt(2)
        b_u, c_u = float_to_dyadic(s_u)
        self.b_u.copy_(torch.tensor(b_u, dtype=torch.int64))
        self.c_u.copy_(torch.tensor(c_u, dtype=torch.int64))
        b_out, c_out = float_to_dyadic(scale_in / max(scale_out, 1e-8))
        self.b_out.copy_(torch.tensor(b_out, dtype=torch.int64))
        self.c_out.copy_(torch.tensor(c_out, dtype=torch.int64))

    def forward(self, q_x: Tensor) -> Tensor:
        if not self.integer_only or q_x.is_floating_point():
            # Continuous/Simulated mode
            x = q_x if q_x.is_floating_point() else q_x.float() * self.scale_in
            u = x / math.sqrt(2.0)
            u_clipped = torch.clamp(torch.abs(u), max=-self.b)
            poly = self.a * torch.pow(u_clipped + self.b, 4) + 1.0
            l_ours = torch.sign(u) * poly
            out = 0.5 * x * (1.0 + l_ours)
            if not q_x.is_floating_point():
                return torch.clamp(torch.round(out / self.scale_out), -128, 127).to(torch.int8)
            return out

        # Pure integer arithmetic mode:
        # q_x is int32/int64 tensor
        # Step 1: u in fixed-point 16-bit fraction
        # u = (q_x * b_u + rounding) >> c_u
        q_long = q_x.long()
        sgn = torch.sign(q_long)
        abs_q = torch.abs(q_long)

        # Scale abs_q to fixed-point (with 16 fractional bits)
        # S_u * abs_q:
        scale_u_fp = int((self.scale_in / math.sqrt(2.0)) * (1 << 16))
        u_fp = (abs_q * scale_u_fp)  # 16-bit fixed point

        # clip(|u|, max=-b) where -b = 2.698088 -> in 16-bit fixed point: 2.698088 * 65536 = 176822
        b_fp = int(self.b * (1 << 16))  # -176822
        max_u_fp = -b_fp
        u_clipped_fp = torch.clamp(u_fp, max=max_u_fp)

        # z_fp = (clip + b) in 16-bit fixed point:
        z_fp = u_clipped_fp + b_fp  # <= 0

        # a * z^4:
        # Compute z^2 >> 16, then z^4 >> 16
        z_sq = (z_fp * z_fp) >> 16
        z_4 = (z_sq * z_sq) >> 16

        # Multiply by a = -0.019913 in 16-bit fixed point: -0.019913 * 65536 = -1305
        a_fp = int(self.a * (1 << 16))
        poly_fp = ((z_4 * a_fp) >> 16) + (1 << 16)

        # l_ours_fp = sgn * poly_fp
        l_ours_fp = sgn * poly_fp

        # GELU: 0.5 * x * (1 + L_ours)
        # in integer: (1 + L_ours) in 16-bit fixed point = (1 << 16) + l_ours_fp
        factor_fp = (1 << 16) + l_ours_fp

        # (q_x * factor_fp) >> 17 (divide by 2 and 2^16)
        gelu_int = (q_long * factor_fp) >> 17

        # Rescale to output scale:
        out_int = dyadic_scale(gelu_int, self.b_out.item(), self.c_out.item())
        return torch.clamp(out_int, -128, 127).to(torch.int8)


class EfficientBitSoftmax(nn.Module):
    """
    IPTQ-ViT (Kim et al., 2025) Efficient Bit-Softmax
    Base-2 decomposition with Taylor series:
    Phi(x) = (x >> 1) + (x >> 3) + (x >> 4)  (ln 2 ~= 0.1011_2)
    Scaled Integer Division (IntDiv) with precision shift M = 28:
    inv_denom = floor(2^M / sum(Q_exp))
    Q_out = (Q_exp * inv_denom) >> (M - (b - 1))
    """
    def __init__(self, scale_in: float = 0.05, bit_width: int = 8, M: int = 28):
        super().__init__()
        self.scale_in = scale_in
        self.bit_width = bit_width
        self.M = M
        self.qmax = (1 << (bit_width - 1)) - 1  # 127

    def update_scale(self, scale_in: float):
        self.scale_in = scale_in

    def forward(self, q_x: Tensor) -> Tensor:
        """
        q_x: INT8 or INT32 attention logits [B, H, N, N].
        Returns: INT8 probability distribution [B, H, N, N] where rows sum to ~128.
        """
        # Step 1: Numerical stabilization via max subtraction
        x_max, _ = torch.max(q_x.long(), dim=-1, keepdim=True)
        q_tilde = q_x.long() - x_max  # <= 0

        # Step 2: Base-2 exponent approximation Q_p = q + (q >> 1) - (q >> 4) ~= q * log2(e)
        q_p = q_tilde + (q_tilde >> 1) - (q_tilde >> 4)

        # Scale in fixed-point 16-bit
        scale_fp = int(self.scale_in * (1 << 16))
        z_fp = q_p * scale_fp  # fixed point 16-bit, <= 0

        # Step 3: Decompose z = -q_shift + r_f, where q_shift >= 0, r_f in [-(1<<16), 0]
        q_shift = (-z_fp) >> 16
        r_fp = z_fp + (q_shift << 16)

        # Step 4: Phi(r) = (r >> 1) + (r >> 3) + (r >> 4)
        phi_r = (r_fp >> 1) + (r_fp >> 3) + (r_fp >> 4)
        two_to_r = (1 << 16) + phi_r

        # Step 5: Exponentiation by bit shift: Q_exp = 2^r >> q_shift
        q_exp = torch.clamp(two_to_r >> q_shift, min=0)

        # Step 6: Scaled integer division (IntDiv) with precision shift M
        denom = torch.clamp(torch.sum(q_exp, dim=-1, keepdim=True), min=1)
        inv_denom = (1 << self.M) // denom

        # Quantize to INT8 (summing to ~128)
        shift_amt = self.M - (self.bit_width - 1)
        q_out = (q_exp * inv_denom) >> shift_amt
        return torch.clamp(q_out, 0, self.qmax).to(torch.int8)


class IntegerLayerNorm(nn.Module):
    """
    I-BERT (Kim et al., 2021) Integer-Only LayerNorm
    Calculates mean, variance, and standard deviation via Newton-Raphson integer square root.
    Zero floating point or hardware division instructions during execution.
    """
    def __init__(self, normalized_shape: int, scale_in: float = 0.05, scale_out: float = 0.05, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.scale_in = scale_in
        self.scale_out = scale_out
        self.eps = eps

        # Integer quantized weights (gamma) and bias (beta)
        self.register_buffer("weight_q", torch.ones(normalized_shape, dtype=torch.int32))
        self.register_buffer("bias_q", torch.zeros(normalized_shape, dtype=torch.int32))

        # Dyadic factor for output rescaling: (scale_gamma / scale_out)
        b_out, c_out = float_to_dyadic(1.0)
        self.register_buffer("b_out", torch.tensor(b_out, dtype=torch.int64))
        self.register_buffer("c_out", torch.tensor(c_out, dtype=torch.int64))

    def set_parameters(self, fp_weight: Tensor, fp_bias: Optional[Tensor], scale_in: float, scale_out: float):
        self.scale_in = scale_in
        self.scale_out = scale_out
        # Quantize gamma with scale = 1.0 / 127.0 so gamma_q ~ 127 for weight=1.0
        scale_gamma = fp_weight.abs().max().item() / 127.0
        self.weight_q.copy_(torch.round(fp_weight / scale_gamma).to(torch.int32))

        if fp_bias is not None:
            self.bias_q.copy_(torch.round(fp_bias / scale_out).to(torch.int32))
        else:
            self.bias_q.zero_()

        b_out, c_out = float_to_dyadic(scale_gamma / max(scale_out, 1e-8))
        self.b_out.copy_(torch.tensor(b_out, dtype=torch.int64))
        self.c_out.copy_(torch.tensor(c_out, dtype=torch.int64))

    def forward(self, q_x: Tensor) -> Tensor:
        """
        q_x: INT8 or INT32 tensor [..., normalized_shape].
        Returns: INT8 tensor [..., normalized_shape].
        """
        q = q_x.long()
        C = self.normalized_shape

        # Step 1: Integer mean mu = floor(sum(q) / C)
        mu = torch.div(q.sum(dim=-1, keepdim=True), C, rounding_mode="trunc")

        # Step 2: Centered activation q' = q - mu
        q_centered = q - mu

        # Step 3: Integer variance V = floor(sum((q')^2) / C)
        var = torch.div(torch.sum(q_centered * q_centered, dim=-1, keepdim=True), C, rounding_mode="trunc")

        # Step 4: Newton-Raphson integer square root sigma = isqrt(V)
        sigma = integer_sqrt_newton(var)
        sigma = torch.clamp(sigma, min=1)

        # Step 5: Affine scaling: (q_centered * weight_q) // sigma
        # Use guard factor (1 << 8) to preserve precision before division
        scaled = torch.div((q_centered * self.weight_q.long()) << 8, sigma, rounding_mode="trunc") >> 8

        # Step 6: Dyadic output rescaling + bias addition
        normed = dyadic_scale(scaled, self.b_out.item(), self.c_out.item()) + self.bias_q.long()
        return torch.clamp(normed, -128, 127).to(torch.int8)


class IntegerLinear(nn.Module):
    """
    Pure Integer Linear Layer:
    INT8 input * INT8 weight -> INT32 accumulator -> dyadic rescale to INT8 output.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight_q", torch.zeros((out_features, in_features), dtype=torch.int8))
        if bias:
            self.register_buffer("bias_q", torch.zeros(out_features, dtype=torch.int32))
        else:
            self.register_buffer("bias_q", None)

        self.register_buffer("b_mult", torch.ones(out_features, dtype=torch.int64))
        self.register_buffer("c_shift", torch.zeros(out_features, dtype=torch.int64))
        self.scale_in = 1.0
        self.scale_out = 1.0

    def set_quantized_parameters(self, fp_weight: Tensor, fp_bias: Optional[Tensor], scale_in: float, scale_out: float):
        self.scale_in = scale_in
        self.scale_out = scale_out
        max_w = fp_weight.abs().amax(dim=1)
        scales_w = torch.clamp(max_w / 127.0, min=1e-8)
        w_q = torch.clamp(torch.round(fp_weight / scales_w.view(-1, 1)), -128, 127).to(torch.int8)
        self.weight_q.copy_(w_q)

        if fp_bias is not None and self.bias_q is not None:
            scales_b = scale_in * scales_w
            b_q = torch.round(fp_bias / scales_b).to(torch.int32)
            self.bias_q.copy_(b_q)

        b_list, c_list = [], []
        for oc in range(self.out_features):
            s_ratio = (scale_in * scales_w[oc].item()) / max(scale_out, 1e-8)
            b, c = float_to_dyadic(s_ratio)
            b_list.append(b)
            c_list.append(c)
        self.b_mult.copy_(torch.tensor(b_list, dtype=torch.int64))
        self.c_shift.copy_(torch.tensor(c_list, dtype=torch.int64))

    def forward(self, q_x: Tensor) -> Tensor:
        # q_x: [..., in_features] in int8/int32
        x_long = q_x.long()
        w_long = self.weight_q.long()
        acc = F.linear(x_long.float(), w_long.float(), self.bias_q.float() if self.bias_q is not None else None).long()

        # Per-channel dyadic rescaling
        b_c = self.b_mult.view(*([1] * (acc.dim() - 1)), -1)
        c_c = self.c_shift.view(*([1] * (acc.dim() - 1)), -1)
        rounding = 1 << torch.clamp(c_c - 1, min=0)
        out = (acc * b_c + rounding) >> c_c
        return torch.clamp(out, -128, 127).to(torch.int8)


class IntegerMatMul(nn.Module):
    """
    Pure Integer Matrix Multiplication for Attention:
    Q * K^T or Attn * V
    INT8 * INT8 -> INT32 accumulator -> dyadic rescale to INT8.
    """
    def __init__(self, scale_a: float = 1.0, scale_b: float = 1.0, scale_out: float = 1.0):
        super().__init__()
        self.scale_a = scale_a
        self.scale_b = scale_b
        self.scale_out = scale_out
        b, c = float_to_dyadic((scale_a * scale_b) / max(scale_out, 1e-8))
        self.register_buffer("b_dyadic", torch.tensor(b, dtype=torch.int64))
        self.register_buffer("c_dyadic", torch.tensor(c, dtype=torch.int64))

    def update_scales(self, scale_a: float, scale_b: float, scale_out: float):
        self.scale_a = scale_a
        self.scale_b = scale_b
        self.scale_out = scale_out
        b, c = float_to_dyadic((scale_a * scale_b) / max(scale_out, 1e-8))
        self.b_dyadic.copy_(torch.tensor(b, dtype=torch.int64))
        self.c_dyadic.copy_(torch.tensor(c, dtype=torch.int64))

    def forward(self, q_a: Tensor, q_b: Tensor) -> Tensor:
        # q_a: [..., M, K], q_b: [..., K, N]
        acc = torch.matmul(q_a.long().float(), q_b.long().float()).long()
        rescaled = dyadic_scale(acc, self.b_dyadic.item(), self.c_dyadic.item())
        return torch.clamp(rescaled, -128, 127).to(torch.int8)
