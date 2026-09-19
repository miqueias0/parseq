from timm.models.vision_transformer import Attention
import inspect

attn = Attention(dim=384, num_heads=6)
print("Attention attributes:", [x for x in dir(attn) if not x.startswith("_")])
print("Source of forward:")
print(inspect.getsource(attn.forward))
