#!/usr/bin/env python3
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import argparse
import time
import torch
from strhub.models.utils import load_from_checkpoint

@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Benchmark puro de FPS (Sem Dataloader)")
    parser.add_argument('checkpoint', help="Model checkpoint (ou 'pretrained=<model_id>')")
    parser.add_argument('--batch_size', type=int, default=1, help="Tamanho do lote (Use 1 para simular tempo real)")
    parser.add_argument('--device', default='cuda', help="Dispositivo para rodar o benchmark")
    parser.add_argument('--iterations', type=int, default=1000, help="Número de iterações do benchmark")
    parser.add_argument('--warmup', type=int, default=100, help="Número de iterações de aquecimento")
    args = parser.parse_args()

    print(f"\nCarregando modelo: {args.checkpoint}")
    model = load_from_checkpoint(args.checkpoint).eval().to(args.device)
    
    # Otimização do CuDNN para o tamanho de input fixo
    torch.backends.cudnn.benchmark = True
    
    # Pega o tamanho da imagem diretamente da configuração do modelo
    img_size = getattr(model.hparams, 'img_size', (32, 128))
    
    print(f"Criando tensores dummy em {args.device} (Batch: {args.batch_size}, Shape: 3x{img_size[0]}x{img_size[1]})...")
    # Tensor aleatório direto na VRAM, evitando totalmente a CPU e o disco rígido
    dummy_imgs = torch.rand(args.batch_size, 3, img_size[0], img_size[1], device=args.device)
    
    print(f"Realizando Warm-up ({args.warmup} iterações)...")
    for _ in range(args.warmup):
        _ = model(dummy_imgs)
        if args.device == 'cuda':
            torch.cuda.synchronize()
            
    print(f"Iniciando Benchmark Oficial ({args.iterations} iterações)...")
    # Sincroniza antes do cronômetro começar
    if args.device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    
    for _ in range(args.iterations):
        _ = model(dummy_imgs)
        
    # Sincroniza depois de todas as requisições, garantindo que o cronômetro só 
    # pause quando a placa de vídeo de fato concluir todo o processamento
    if args.device == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()
    
    total_time = t1 - t0
    total_images = args.batch_size * args.iterations
    fps = total_images / total_time
    ms_per_batch = (total_time / args.iterations) * 1000
    
    device_name = torch.cuda.get_device_name(0) if args.device == 'cuda' and torch.cuda.is_available() else 'CPU'
    
    print("\n" + "="*50)
    print(" 🚀 Resultado do Benchmark Puro (Padrão Acadêmico):")
    print("="*50)
    print(f"GPU:            {device_name}")
    print(f"Batch Size:     {args.batch_size}")
    print(f"Tempo Total:    {total_time:.4f} s")
    print(f"Imagens Proc.:  {total_images}")
    print(f"Latência/Batch: {ms_per_batch:.2f} ms")
    print(f"FPS Oficial:    {fps:.2f} FPS")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()
