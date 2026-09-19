import torch

# Test mathematical bit-exactness of float32 GEMM for int8 values
a = torch.randint(-128, 127, (64, 64), dtype=torch.int8, device='cuda')
b = torch.randint(-128, 127, (64, 64), dtype=torch.int8, device='cuda')

# Exact integer via CPU loops or cpu int32
a_cpu_int32 = a.cpu().to(torch.int64)
b_cpu_int32 = b.cpu().to(torch.int64)
exact_int32 = torch.matmul(a_cpu_int32, b_cpu_int32).to(torch.int32).cuda()

# Float32 matmul on CUDA
float_gemm_int32 = torch.matmul(a.float(), b.float()).to(torch.int32)

diff = (exact_int32 - float_gemm_int32).abs().max().item()
print("Max diff between Float32 matmul and exact int32:", diff)
assert diff == 0, "Float32 GEMM was not bit-exact!"
print("SUCCESS: Float32 GEMM is 100% bit-exact for int8 x int8 -> int32!")
