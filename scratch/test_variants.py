import torch
from strhub.models.utils import load_from_checkpoint
from strhub.models.parseq.quantized_parseq import create_model_variant

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
system = load_from_checkpoint("pretrained/parseq_alpr_98.5.ckpt").eval().to(device)
print("Base system loaded on", device)

for v in ["m0", "m1", "m2", "m3", "m4", "m5", "m6"]:
    try:
        model_v = create_model_variant(v, system.model).eval().to(device)
        dummy_img = torch.randn(1, 3, 32, 128, device=device)
        if v == "m2":
            dummy_img = dummy_img.half()
        out = model_v(system.tokenizer, dummy_img)
        print(f"Variant {v}: SUCCESS - output shape: {out.shape}")
    except Exception as e:
        print(f"Variant {v}: FAILED - {type(e).__name__}: {e}")
