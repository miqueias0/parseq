import torch

print("PyTorch:", torch.__version__)
a = torch.randint(-128, 127, (32, 32), dtype=torch.int8)
b = torch.randint(-128, 127, (32, 32), dtype=torch.int8)

try:
    c_cpu = torch._int_mm(a, b)
    print("torch._int_mm CPU works! shape:", c_cpu.shape, c_cpu.dtype)
except Exception as e:
    print("torch._int_mm CPU failed:", type(e).__name__, e)

if torch.cuda.is_available():
    a_cuda = a.cuda()
    b_cuda = b.cuda()
    try:
        c_cuda = torch._int_mm(a_cuda, b_cuda)
        print("torch._int_mm CUDA works! shape:", c_cuda.shape)
    except Exception as e:
        print("torch._int_mm CUDA failed:", type(e).__name__, e)
