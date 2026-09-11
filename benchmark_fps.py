#!/usr/bin/env python3
import argparse
import os
import time
from pathlib import Path
from typing import Optional, Sequence

import torch
from hydra import compose, initialize_config_dir
import hydra.utils

from strhub.models.parseq.system import PARSeq
from strhub.models.utils import load_from_checkpoint


def load_model(
    model_or_ckpt: str,
    device: str = 'cpu',
    img_size: Optional[Sequence[int]] = None,
    enc_depth: Optional[int] = None,
    cnn_depth: Optional[int] = None,
    embed_dim: Optional[int] = None,
    backbone: Optional[str] = None,
    block_type: Optional[str] = None,
):
    """Loads a model from a checkpoint (.ckpt), pretrained ID, or Hydra model config."""
    # 1. Pretrained tag (e.g. pretrained=parseq)
    if model_or_ckpt.startswith('pretrained='):
        print(f"\nCarregando pesos pré-treinados: {model_or_ckpt}")
        model = load_from_checkpoint(model_or_ckpt)
        return model.eval().to(device)

    # 2. Check if it's an existing file on disk (.ckpt or .pt)
    if os.path.isfile(model_or_ckpt):
        print(f"\nCarregando checkpoint de arquivo: {model_or_ckpt}")
        try:
            model = load_from_checkpoint(model_or_ckpt)
        except Exception:
            # Fallback direct load with PARSeq
            model = PARSeq.load_from_checkpoint(model_or_ckpt)
        return model.eval().to(device)

    # 3. Model config name (e.g. parseq_cnn_repvit, parseq_alpr_fast, etc.)
    model_name = model_or_ckpt
    if model_name.startswith('model='):
        model_name = model_name.split('=', 1)[1]
    if model_name.endswith('.yaml'):
        model_name = Path(model_name).stem

    print(f"\nInstanciando modelo a partir da configuração: configs/model/{model_name}.yaml")
    config_dir = str(Path(__file__).parent.joinpath('configs').resolve())
    overrides = [f'model={model_name}']

    if img_size is not None:
        overrides.append(f'++model.img_size=[{img_size[0]},{img_size[1]}]')
    if enc_depth is not None:
        overrides.append(f'++model.enc_depth={enc_depth}')
    if cnn_depth is not None:
        overrides.append(f'++model.cnn_depth={cnn_depth}')
    if embed_dim is not None:
        overrides.append(f'++model.embed_dim={embed_dim}')
    if backbone is not None:
        overrides.append(f'++model.backbone={backbone}')
    if block_type is not None:
        overrides.append(f'++model.block_type={block_type}')

    with initialize_config_dir(config_dir=config_dir, version_base='1.2'):
        cfg = compose(config_name='main', overrides=overrides)
        model = hydra.utils.instantiate(cfg.model)

    return model.eval().to(device)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Benchmark puro de FPS e Latência para PARSeq (ViT, CNN e Híbrido)")
    parser.add_argument('checkpoint', help="Model checkpoint (.ckpt), nome de config (ex: parseq_cnn_repvit) ou 'pretrained=<id>'")
    parser.add_argument('--batch_size', type=int, default=1, help="Tamanho do lote (Padrão: 1 para simular tempo real)")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help="Dispositivo (cuda ou cpu)")
    parser.add_argument('--iterations', type=int, default=1000, help="Número de iterações do benchmark (Padrão: 1000)")
    parser.add_argument('--warmup', type=int, default=100, help="Número de iterações de aquecimento (Padrão: 100)")
    parser.add_argument('--img_size', type=int, nargs=2, default=None, metavar=('H', 'W'), help="Sobrescrever tamanho da imagem [H W]")
    parser.add_argument('--enc_depth', type=int, default=None, help="Sobrescrever enc_depth (0=100% CNN, >0=Híbrido)")
    parser.add_argument('--cnn_depth', type=int, default=None, help="Sobrescrever cnn_depth (número de blocos CNN residuais)")
    parser.add_argument('--embed_dim', type=int, default=None, help="Sobrescrever embed_dim")
    parser.add_argument('--backbone', type=str, default=None, help="Sobrescrever backbone CNN (ex: repvit_m0_9, mobilenetv3_small_050, resnet18, conv_stem)")
    parser.add_argument('--block_type', choices=['transformer', 'hybrid'], default=None, help="Tipo de bloco encoder (transformer ou hybrid)")
    args = parser.parse_args()

    # Se CUDA foi requisitado mas não está disponível, avisa e usa CPU
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Aviso: CUDA não está disponível. Executando benchmark na CPU.")
        args.device = 'cpu'

    model = load_model(
        args.checkpoint,
        device=args.device,
        img_size=args.img_size,
        enc_depth=args.enc_depth,
        cnn_depth=args.cnn_depth,
        embed_dim=args.embed_dim,
        backbone=args.backbone,
        block_type=args.block_type,
    )

    if args.device == 'cuda':
        torch.backends.cudnn.benchmark = True

    # Informações da arquitetura do modelo
    hparams = getattr(model, 'hparams', {})
    if args.img_size is not None:
        img_h, img_w = args.img_size
    else:
        img_size_hp = getattr(hparams, 'img_size', (32, 128))
        img_h, img_w = img_size_hp[0], img_size_hp[1]

    backbone_name = getattr(hparams, 'backbone', None)
    enc_depth = getattr(hparams, 'enc_depth', None)
    cnn_depth_val = getattr(hparams, 'cnn_depth', 0)
    embed_dim = getattr(hparams, 'embed_dim', None)
    dec_depth = getattr(hparams, 'dec_depth', 1)
    dec_heads = getattr(hparams, 'dec_num_heads', None)
    block_type = getattr(hparams, 'block_type', 'transformer')

    if backbone_name or cnn_depth_val > 0:
        b_name = backbone_name or 'conv_stem'
        if enc_depth == 0:
            arch_type = f"Pure CNN (Backbone: {b_name}, cnn_depth={cnn_depth_val}, enc_depth=0)"
        else:
            arch_type = f"Hybrid CNN + {block_type.capitalize()} (Backbone: {b_name}, cnn_depth={cnn_depth_val}, enc_depth={enc_depth})"
    else:
        arch_type = f"Pure ViT Transformer (enc_depth={enc_depth})"

    total_params = sum(p.numel() for p in model.parameters())

    print(f"Criando tensores dummy em {args.device} (Batch: {args.batch_size}, Shape: 3x{img_h}x{img_w})...")
    dummy_imgs = torch.rand(args.batch_size, 3, img_h, img_w, device=args.device)

    print(f"Realizando Warm-up ({args.warmup} iterações)...")
    for _ in range(args.warmup):
        _ = model(dummy_imgs)
        if args.device == 'cuda':
            torch.cuda.synchronize()

    print(f"Iniciando Benchmark Oficial ({args.iterations} iterações)...")
    if args.device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()

    for _ in range(args.iterations):
        _ = model(dummy_imgs)

    if args.device == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()

    total_time = t1 - t0
    total_images = args.batch_size * args.iterations
    fps = total_images / total_time
    ms_per_batch = (total_time / args.iterations) * 1000
    ms_per_image = ms_per_batch / args.batch_size

    device_name = torch.cuda.get_device_name(0) if args.device == 'cuda' else 'CPU'

    print("\n" + "=" * 60)
    print(" 🚀 Resultado do Benchmark de FPS & Latência (PARSeq):")
    print("=" * 60)
    print(f"Dispositivo:       {device_name}")
    print(f"Tipo de Encoder:   {arch_type}")
    print(f"Dimensão Embed:    {embed_dim}")
    print(f"Decoder:           {dec_depth} camada(s), {dec_heads} cabeças")
    print(f"Parâmetros:        {total_params:,} ({total_params / 1e6:.2f} M)")
    print(f"Resolução Entrada: 3x{img_h}x{img_w}")
    print(f"Batch Size:        {args.batch_size}")
    print(f"Iterações:         {args.iterations}")
    print(f"Tempo Total:       {total_time:.4f} s")
    print(f"Imagens Proc.:     {total_images}")
    print(f"Latência/Imagem:   {ms_per_image:.2f} ms")
    print(f"Latência/Batch:    {ms_per_batch:.2f} ms")
    print(f"FPS Oficial:       {fps:.2f} FPS")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
