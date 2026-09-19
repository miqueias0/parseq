import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))
    a = torch.randint(-128, 127, (16, 32), dtype=torch.int8, device='cuda')
    b = torch.randint(-128, 127, (32, 16), dtype=torch.int8, device='cuda')
    try:
        c = torch._int_mm(a, b)
        print("torch._int_mm works on CUDA! Result shape:", c.shape)
    except Exception as e:
        print("torch._int_mm on CUDA failed:", type(e).__name__, e)
    
    try:
        ca = torch.matmul(a.to(torch.int32), b.to(torch.int32))
        print("int32 matmul fallback works! Result shape:", ca.shape)
    except Exception as e:
        print("int32 matmul fallback failed:", type(e).__name__, e)
