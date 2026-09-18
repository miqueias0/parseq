import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
import ctypes
import os

from strhub.quant.int_flashattention import INTFlashAttention
from strhub.quant.integer_layernorm import IBERTLayerNorm
from strhub.quant.integer_gelu import IBERTGELU
from strhub.quant.integer_softmax import IBERTSoftmax
from strhub.quant.plugins.trt_plugins import get_plugin_dll

dll = get_plugin_dll()

# 1. Test INT-FlashAttention Parity
torch.manual_seed(42)
B, H, N, S, D = 1, 6, 217, 217, 64
scale = 1.0 / (D ** 0.5)

q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32)
v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32)

# PyTorch Reference
ref_fa = INTFlashAttention(embed_dim=H*D, num_heads=H, block_r=16, block_c=16, bits=8).cuda()
with torch.no_grad():
    ref_out = ref_fa(q, k, v)

# CUDA Kernel
cuda_out = torch.empty_like(q)
stream = torch.cuda.current_stream()
dll.run_int_flash_attention(
    ctypes.c_void_p(cuda_out.data_ptr()),
    ctypes.c_void_p(q.data_ptr()),
    ctypes.c_void_p(k.data_ptr()),
    ctypes.c_void_p(v.data_ptr()),
    ctypes.c_int(B), ctypes.c_int(H), ctypes.c_int(N), ctypes.c_int(S), ctypes.c_int(D),
    ctypes.c_float(scale),
    ctypes.c_void_p(stream.cuda_stream)
)
torch.cuda.synchronize()

cos_sim_fa = F.cosine_similarity(ref_out.flatten(), cuda_out.flatten(), dim=0).item()
mae_fa = torch.mean(torch.abs(ref_out - cuda_out)).item()
print(f"INT-FlashAttention Parity: CosSim = {cos_sim_fa:.4f}, MAE = {mae_fa:.6f}")

# 2. Test Integer LayerNorm Parity
x = torch.randn(217, 384, device="cuda", dtype=torch.float32)
ref_ln = IBERTLayerNorm(384).cuda().eval()
with torch.no_grad():
    ref_ln_out = ref_ln(x)

cuda_ln_out = torch.empty_like(x)
dll.run_integer_layernorm(
    ctypes.c_void_p(cuda_ln_out.data_ptr()),
    ctypes.c_void_p(x.data_ptr()),
    ctypes.c_void_p(ref_ln.weight.data.data_ptr()),
    ctypes.c_void_p(ref_ln.bias.data.data_ptr()),
    ctypes.c_int(217), ctypes.c_int(384), ctypes.c_float(1e-5),
    ctypes.c_void_p(stream.cuda_stream)
)
torch.cuda.synchronize()
cos_sim_ln = F.cosine_similarity(ref_ln_out.flatten(), cuda_ln_out.flatten(), dim=0).item()
mae_ln = torch.mean(torch.abs(ref_ln_out - cuda_ln_out)).item()
print(f"Integer LayerNorm Parity: CosSim = {cos_sim_ln:.4f}, MAE = {mae_ln:.6f}")

# 3. Test Integer GELU Parity
x_gelu = torch.randn(217, 384, device="cuda", dtype=torch.float32)
ref_gelu = IBERTGELU().cuda().eval()
with torch.no_grad():
    ref_gelu_out = ref_gelu(x_gelu)

cuda_gelu_out = torch.empty_like(x_gelu)
dll.run_integer_gelu(
    ctypes.c_void_p(cuda_gelu_out.data_ptr()),
    ctypes.c_void_p(x_gelu.data_ptr()),
    ctypes.c_int(x_gelu.numel()),
    ctypes.c_void_p(stream.cuda_stream)
)
torch.cuda.synchronize()
cos_sim_gelu = F.cosine_similarity(ref_gelu_out.flatten(), cuda_gelu_out.flatten(), dim=0).item()
mae_gelu = torch.mean(torch.abs(ref_gelu_out - cuda_gelu_out)).item()
print(f"Integer GELU Parity: CosSim = {cos_sim_gelu:.4f}, MAE = {mae_gelu:.6f}")

# 4. Test Integer Softmax Parity
x_sm = torch.randn(6, 217, 217, device="cuda", dtype=torch.float32)
ref_sm = IBERTSoftmax(dim=-1).cuda().eval()
with torch.no_grad():
    ref_sm_out = ref_sm(x_sm)

cuda_sm_out = torch.empty_like(x_sm)
dll.run_integer_softmax(
    ctypes.c_void_p(cuda_sm_out.data_ptr()),
    ctypes.c_void_p(x_sm.data_ptr()),
    ctypes.c_int(6 * 217), ctypes.c_int(217),
    ctypes.c_void_p(stream.cuda_stream)
)
torch.cuda.synchronize()
cos_sim_sm = F.cosine_similarity(ref_sm_out.flatten(), cuda_sm_out.flatten(), dim=0).item()
mae_sm = torch.mean(torch.abs(ref_sm_out - cuda_sm_out)).item()
print(f"Integer Softmax Parity: CosSim = {cos_sim_sm:.4f}, MAE = {mae_sm:.6f}")
