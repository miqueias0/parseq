# Guia Completo: Modelos Híbridos (CNN + Transformer), Pure CNN e ALPR no PARSeq

Este documento apresenta a arquitetura, catálogo de modelos, instruções de benchmark e guia de parametrização via Linha de Comando (CLI) para os encoders convolucionais e híbridos implementados no PARSeq.

---

## 1. Visão Geral da Arquitetura

O PARSeq padrão utiliza um Vision Transformer (ViT) puro de 12 camadas. Esta extensão adiciona suporte total a:
1. **Pure CNN (100% Convolucional)**: Todo o encoder visual é executado via CNN, sem blocos Transformer (`enc_depth: 0`).
2. **Híbrido Clássico (CNN + ViT)**: Um backbone convolucional extrai traços locais e texturas primitivas, seguido por $N$ camadas Transformer (`enc_depth: 2, 4, 6`) para atenção global entre caracteres.
3. **Híbrido com `HybridBlock`**: Cada camada do encoder combina internamente convolução Depthwise ($3 \times 3$) para mistura local com Multi-Head Self-Attention (MHSA) para dependências globais e FFN.

### Fluxo Arquitetural (Pipeline do Encoder)

```text
Entrada (Ex: 32x128, 48x96, 64x96)
         │
         ▼
┌───────────────────────────────────────────────┐
│ CNN Backbone Truncado (Redução 4x)            │
│ (RepViT-M0.9 / MobileNetV3 / RepVGG / ResNet) │
│ Saída: [H/4, W/4, C_in] (Ex: 8x32, 12x24)     │
└───────────────────────────────────────────────┘
         │
         ▼
┌───────────────────────────────────────────────┐
│ DWConv 3x3 Anisotrópico (stride=[1, 2])       │
│ Preserva a altura H/4 e reduz a largura W/8   │
│ Saída: [H/4, W/8, C_in] (Ex: 8x16, 12x12)     │
└───────────────────────────────────────────────┘
         │
         ▼
┌───────────────────────────────────────────────┐
│ Projeção Convolucional 1x1                    │
│ [H/4, W/8, C_in] ──► [H/4, W/8, embed_dim]    │
└───────────────────────────────────────────────┘
         │
         ▼
┌───────────────────────────────────────────────┐
│ Flatten Espacial + Positional Embedding 2D    │
│ Grade [H/4, W/8] ──► Sequência de N Tokens    │
│ N = (H/4) * (W/8) (Ex: 8x16 = 128 tokens)     │
└───────────────────────────────────────────────┘
         │
         ▼
┌───────────────────────────────────────────────┐
│ Blocos Encoder (Opcionais se enc_depth > 0)   │
│ - Se enc_depth == 0: Vai direto ao Decoder    │
│ - Se block_type == 'transformer': ViT Blocks  │
│ - Se block_type == 'hybrid': HybridBlocks     │
└───────────────────────────────────────────────┘
         │
         ▼
┌───────────────────────────────────────────────┐
│ PARSeq Decoder (Cross-Attention com Memória)  │
│ 1 ou 2 camadas autorregressivas / paralelas   │
└───────────────────────────────────────────────┘
```

> **Por que Downsampling Anisotrópico (`stride=[1, 2]`)?**  
> Em reconhecimento de placas e textos, os caracteres são naturalmente estreitos e altos. Strides simétricos ($2 \times 2$) destroem a resolução vertical rapidamente. O stride $(1, 2)$ reduz a dimensão horizontal sem esmagar a altura dos caracteres.

---

## 2. Catálogo de Modelos Criados

Todos os modelos estão disponíveis como arquivos YAML prontos em `configs/model/`.

### A. Modelos Especializados para ALPR (Ultra FPS e Edge)

Configurações afinadas para leitura de placas (7 caracteres, inferência paralela não-autorregressiva `decode_ar: false`):

| Configuração | Backbone | Embed Dim | Enc Depth | Decodificador | Parâmetros | FPS CPU (Batch 1) | Latência CPU |
|---|---|---|---|---|---|---|---|
| `parseq_cnn_alpr_nano.yaml` | RepViT M0.9 | 192 | 0 (Pure CNN) | 1 layer, 3 heads, 0 refine | **0.53 M** | **~110 FPS** | ~9.1 ms |
| `parseq_cnn_alpr_fast.yaml` | RepViT M0.9 | 256 | 0 (Pure CNN) | 1 layer, 4 heads, 1 refine | **1.03 M** | **~112 FPS** | ~8.9 ms |
| `parseq_cnn_alpr_small.yaml` | RepViT M0.9 | 256 | 2 (Hybrid) | 1 layer, 4 heads, 1 refine | **1.82 M** | **~70 FPS** | ~14.5 ms |
| `parseq_cnn_mobilenetv3_alpr_fast.yaml` | MobileNetV3 Small | 256 | 0 (Pure CNN) | 1 layer, 4 heads, 1 refine | **1.10 M** | **~102 FPS** | ~9.8 ms |
| `parseq_cnn_repvgg_alpr_fast.yaml` | RepVGG A0 | 256 | 0 (Pure CNN) | 1 layer, 4 heads, 1 refine | **1.35 M** | **~90 FPS** | ~11.2 ms |

---

### B. Matriz por Família de Backbone

#### 1. RepViT (`repvit_m0_9`)
- `parseq_cnn_repvit.yaml`: Pure CNN baseline (`embed_dim: 384`, `enc_depth: 0`).
- `parseq_cnn_repvit_hybrid.yaml`: Híbrido com 2 camadas Transformer (`enc_depth: 2`).
- `parseq_cnn_repvit_hybrid_4v.yaml`: Híbrido com 4 camadas Transformer (`enc_depth: 4`).
- `parseq_cnn_repvit_hybrid_6v.yaml`: Híbrido com 6 camadas Transformer (`enc_depth: 6`).
- `parseq_cnn_repvit_hybrid_block.yaml`: 4 camadas de `HybridBlock` (DWConv + Atenção Integrada).

#### 2. MobileNetV3
- `parseq_cnn_mobilenetv3_small.yaml`: Pure CNN com MobileNetV3-Small (`embed_dim: 384`).
- `parseq_cnn_mobilenetv3_large.yaml`: Pure CNN com MobileNetV3-Large (`embed_dim: 384`).
- `parseq_cnn_mobilenetv3_hybrid.yaml`: MobileNetV3-Small + 4 camadas Transformer.

#### 3. RepVGG (`repvgg_a0`)
- `parseq_cnn_repvgg_a0.yaml`: Pure CNN com RepVGG A0 reparametrizável.
- `parseq_cnn_repvgg_hybrid.yaml`: RepVGG A0 + 4 camadas Transformer.

#### 4. ResNet-18 & GhostNet
- `parseq_cnn_resnet18.yaml` e `parseq_cnn_resnet18_hybrid.yaml`: Baseline clássico de STR.
- `parseq_cnn_ghostnet.yaml` e `parseq_cnn_ghostnet_hybrid.yaml`: Backbone com operações convolucionais baratas.

---

### C. Matriz de Ablações Sistemáticas

#### Variação de `embed_dim` (Largura de Canal)
- **192**:
  - `parseq_cnn_repvit_embed192.yaml` (`enc_depth: 0`)
  - `parseq_cnn_repvit_embed192_enc_depth2.yaml` (`enc_depth: 2`)
  - `parseq_cnn_repvit_embed192_enc_depth4.yaml` (`enc_depth: 4`)
  - `parseq_cnn_repvit_embed192_enc_depth6.yaml` (`enc_depth: 6`)
  - `parseq_cnn_mobilenetv3_embed192.yaml`, `parseq_cnn_repvgg_embed192.yaml`
- **256**:
  - `parseq_cnn_repvit_embed256.yaml` (`enc_depth: 0`)
  - `parseq_cnn_repvit_embed256_enc_depth2.yaml` (`enc_depth: 2`)
  - `parseq_cnn_repvit_embed256_enc_depth4.yaml` (`enc_depth: 4`)
  - `parseq_cnn_repvit_embed256_enc_depth6.yaml` (`enc_depth: 6`)
  - `parseq_cnn_mobilenetv3_embed256.yaml`, `parseq_cnn_repvgg_embed256.yaml`

#### Variação de Cabeças de Atenção, Razão MLP e Decoder
- **Cabeças**: `parseq_cnn_repvit_heads3_3.yaml` (3 heads), `parseq_cnn_repvit_heads6_6.yaml` (6 heads).
- **MLP Expansion**: `parseq_cnn_repvit_mlp2.yaml` (ratio 2), `parseq_cnn_repvit_mlp3.yaml` (ratio 3).
- **Profundidade do Decoder**: `parseq_cnn_repvit_dec_depth2.yaml` (2 camadas decoder), `parseq_cnn_repvit_hybrid_dec_depth2.yaml` (Híbrido com 2 camadas decoder).

---

## 3. Como Executar os Benchmarks de FPS e Latência

O script `benchmark_fps.py` permite avaliar qualquer modelo instanciado via configuração, pesos pré-treinados ou arquivo de checkpoint `.ckpt`.

### Sintaxe Básica:
```bash
python benchmark_fps.py <MODELO_OU_CHECKPOINT> [OPÇÕES]
```

### Argumentos Principais:
- `<checkpoint>`: Nome do arquivo YAML em `configs/model/` (ex: `parseq_cnn_alpr_fast`), caminho para `.ckpt` ou tag `pretrained=<nome>`.
- `--device`: Dispositivo de execução (`cuda` ou `cpu`). Padrão: automático.
- `--batch_size`: Tamanho do lote (padrão: `1` para simular latência de tempo real).
- `--iterations`: Número de passagens para média (padrão: `1000`).
- `--warmup`: Número de iterações de aquecimento descartadas (padrão: `100`).
- `--img_size H W`: Sobrescreve a resolução de entrada (ex: `--img_size 48 96` ou `--img_size 64 96`).

### Exemplos Práticos de Execução:

```bash
# 1. Benchmark do ALPR Nano em CPU (50 iterações, batch 1)
python benchmark_fps.py parseq_cnn_alpr_nano --device cpu --iterations 100 --warmup 20

# 2. Benchmark do ALPR Fast na resolução 48x96 em GPU
python benchmark_fps.py parseq_cnn_alpr_fast --img_size 48 96 --device cuda

# 3. Benchmark do Híbrido RepViT + 4 ViT na resolução 64x96 (motos de 2 linhas)
python benchmark_fps.py parseq_cnn_repvit_hybrid_4v --img_size 64 96 --device cuda

# 4. Benchmark do HybridBlock integrado (DWConv + MHSA)
python benchmark_fps.py parseq_cnn_repvit_hybrid_block --device cuda

# 5. Benchmark de um checkpoint treinado salvo em disco
python benchmark_fps.py outputs/parseq-cnn-repvit/checkpoints/best.ckpt --device cuda
```

---

## 4. Como Modificar as Estruturas dos Modelos via CLI

Você pode modificar a arquitetura do modelo **dinamicamente via linha de comando**, sem precisar criar novos arquivos YAML!

### A. Modificações Estruturais no `benchmark_fps.py`

O script expõe flags dedicadas para alterar a arquitetura no momento da execução:

| Flag CLI | Descrição | Exemplo de Uso |
|---|---|---|
| `--backbone` | Troca o backbone CNN dinamicamente | `--backbone mobilenetv3_large_100` ou `--backbone conv_stem` |
| `--cnn_depth` | Define o número de blocos CNN residuais dedicados | `--cnn_depth 4` ou `--cnn_depth 6` |
| `--enc_depth` | Altera profundidade do encoder Transformer (0 = Pure CNN, >0 = Híbrido) | `--enc_depth 2` ou `--enc_depth 6` |
| `--embed_dim` | Altera a dimensão latente de canais | `--embed_dim 192` ou `--embed_dim 256` |
| `--block_type` | Alterna entre bloco ViT padrão e HybridBlock | `--block_type hybrid` |
| `--img_size` | Altera dimensões espaciais de entrada | `--img_size 64 96` |

#### Exemplos no Benchmark:
```bash
# 1. Definir exatamente 4 blocos CNN + 4 blocos Transformer (4 CNN + 4 ViT):
python benchmark_fps.py parseq_cnn_repvit --cnn_depth 4 --enc_depth 4 --device cuda

# 2. Modelo 100% Convolucional com ConvStem nativo e 6 blocos CNN (sem Transformer):
python benchmark_fps.py parseq_cnn_repvit --backbone conv_stem --cnn_depth 6 --enc_depth 0 --device cuda

# 3. Testar RepViT com 6 camadas ViT e embed_dim=256 sem criar YAML novo:
python benchmark_fps.py parseq_cnn_repvit --enc_depth 6 --embed_dim 256 --device cuda

# 4. Testar MobileNetV3-Large Pure CNN em resolução 48x96:
python benchmark_fps.py parseq_cnn_repvit --backbone mobilenetv3_large_100 --enc_depth 0 --img_size 48 96

# 5. Testar bloco híbrido (DWConv + MHSA) com 4 camadas:
python benchmark_fps.py parseq_cnn_repvit --block_type hybrid --enc_depth 4 --device cuda
```

---

### B. Modificações Estruturais no Treinamento (`train.py` via Hydra)

Como o projeto utiliza Hydra, qualquer hiperparâmetro ou nó do modelo pode ser sobrescrito com a sintaxe `model.chave=valor` ou `+experiment=nome`.

#### 1. Configurando a Quantidade de Blocos CNNs (`cnn_depth`)
Para controlar o número exato de blocos convolucionais locais (DWConv $3\times3$ + PWConv $1\times1$ residual) antes da atenção ou como Pure CNN:

```bash
# Exemplo 1: Híbrido equilibrado 4 CNN + 4 ViT
python train.py +experiment=exp_48x96_parseq_cnn_repvit \
    model.cnn_depth=4 \
    model.enc_depth=4

# Exemplo 2: Híbrido 6 CNN + 2 ViT
python train.py +experiment=exp_48x96_parseq_cnn_repvit \
    model.cnn_depth=6 \
    model.enc_depth=2

# Exemplo 3: Pure CNN independente (ConvStem + 6 blocos CNN e 0 ViT)
python train.py +experiment=exp_48x96_parseq_cnn_repvit \
    model.backbone=conv_stem \
    model.cnn_depth=6 \
    model.enc_depth=0
```

#### 2. Sobrescrevendo o Backbone e Profundidade do Encoder
```bash
# Treinar com MobileNetV3-Small como Pure CNN (enc_depth=0)
python train.py +experiment=exp_48x96_parseq_cnn_repvit \
    model.backbone=mobilenetv3_small_050 \
    model.enc_depth=0

# Treinar Híbrido RepViT com 6 blocos Transformer
python train.py +experiment=exp_48x96_parseq_cnn_repvit \
    model.enc_depth=6
```

#### 2. Sobrescrevendo Dimensão Latente e Cabeças de Atenção
> **Atenção**: `embed_dim` deve ser divisível pelo número de cabeças (`enc_num_heads` e `dec_num_heads`).
```bash
# Reduzir para embed_dim=192 com 3 cabeças no encoder e 3 no decoder
python train.py +experiment=exp_48x96_parseq_cnn_repvit \
    model.embed_dim=192 \
    model.enc_num_heads=3 \
    model.dec_num_heads=3 \
    model.enc_mlp_ratio=2 \
    model.dec_mlp_ratio=2
```

#### 3. Ativando o Bloco Híbrido (`HybridBlock`)
```bash
# Treinar com HybridBlock (DWConv 3x3 local + Atenção global no mesmo bloco)
python train.py +experiment=exp_48x96_parseq_cnn_repvit \
    model.block_type=hybrid \
    model.enc_depth=4
```

#### 4. Otimizações de Inferência ALPR (Não-Autorregressivo & Sem Refinamento)
```bash
# Treinar com modo não-autorregressivo (disparo direto de 1 passo)
python train.py +experiment=exp_48x96_parseq_cnn_repvit \
    model.decode_ar=false \
    model.refine_iters=0 \
    model.max_label_length=8
```

#### 5. Sobrescrevendo Resolução e Parâmetros de Treinador
```bash
# Ajustar resolução para 64x96, batch_size para 128 e taxa de aprendizado:
python train.py +experiment=exp_64x96_parseq_cnn_alpr_fast \
    model.batch_size=128 \
    model.lr=1e-3 \
    trainer.max_epochs=100
```

---

## 5. Matriz de Experimentos Prontos (`configs/experiment/`)

Foram gerados 67 arquivos de experimentos organizados por resolução de imagem e tipo de veículo/placa:

### Resolução 48×96 (Unificado VeSV - Carros e Motos)
- `exp_48x96_parseq_cnn_alpr_nano.yaml`
- `exp_48x96_parseq_cnn_alpr_fast.yaml`
- `exp_48x96_parseq_cnn_alpr_small.yaml`
- `exp_48x96_parseq_cnn_repvit.yaml`
- `exp_48x96_parseq_cnn_repvit_hybrid_4v.yaml`
- `exp_48x96_parseq_cnn_repvit_hybrid_6v.yaml`
- `exp_48x96_parseq_cnn_repvit_hybrid_block.yaml`
- `exp_48x96_parseq_cnn_repvit_embed192.yaml`
- `exp_48x96_parseq_cnn_repvit_embed256.yaml`
- `exp_48x96_parseq_cnn_mobilenetv3_small.yaml`
- `exp_48x96_parseq_cnn_mobilenetv3_large.yaml`
- `exp_48x96_parseq_cnn_mobilenetv3_hybrid.yaml`
- `exp_48x96_parseq_cnn_repvgg_a0.yaml`
- `exp_48x96_parseq_cnn_repvgg_hybrid.yaml`
- `exp_48x96_parseq_cnn_resnet18.yaml`
- `exp_48x96_parseq_cnn_resnet18_hybrid.yaml`
- `exp_48x96_parseq_cnn_ghostnet.yaml`
- `exp_48x96_parseq_cnn_ghostnet_hybrid.yaml`
- *(Ablações de heads, mlp, dec_depth e embed 192/256 com enc_depth 2, 4, 6)*

### Resolução 64×96 (Especializado para Motos com 2 Linhas de Caracteres)
- `exp_64x96_parseq_cnn_alpr_nano.yaml`
- `exp_64x96_parseq_cnn_alpr_fast.yaml`
- `exp_64x96_parseq_cnn_alpr_small.yaml`
- `exp_64x96_parseq_cnn_repvit.yaml`
- `exp_64x96_parseq_cnn_repvit_hybrid_4v.yaml`
- `exp_64x96_parseq_cnn_repvit_hybrid_6v.yaml`
- `exp_64x96_parseq_cnn_repvit_hybrid_block.yaml`
- `exp_64x96_parseq_cnn_repvit_embed192.yaml`
- `exp_64x96_parseq_cnn_repvit_embed256.yaml`
- `exp_64x96_parseq_cnn_mobilenetv3_small.yaml`
- `exp_64x96_parseq_cnn_repvgg_a0.yaml`
- `exp_64x96_parseq_cnn_resnet18.yaml`
- `exp_64x96_parseq_cnn_ghostnet.yaml`
- *(Ablações correspondentes de heads, mlp e profundidade)*

---

## 6. Resumo e Próximos Passos Sugeridos

1. **Benchmark Preliminar na sua GPU**:
   Rode os modelos de ALPR na GPU alvo para aferir a curva de FPS com aceleração por hardware:
   ```bash
   python benchmark_fps.py parseq_cnn_alpr_nano --device cuda
   python benchmark_fps.py parseq_cnn_alpr_fast --device cuda
   python benchmark_fps.py parseq_cnn_alpr_small --device cuda
   ```
2. **Treinamento Comparativo Inicial**:
   Inicie treinando os três principais candidatos em `48x96`:
   - Baseline Pure CNN: `python train.py +experiment=exp_48x96_parseq_cnn_alpr_fast`
   - Híbrido Leve (2 ViT): `python train.py +experiment=exp_48x96_parseq_cnn_alpr_small`
   - Híbrido Padrão (4 ViT): `python train.py +experiment=exp_48x96_parseq_cnn_repvit_hybrid_4v`
