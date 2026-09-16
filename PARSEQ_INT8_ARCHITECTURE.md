# PARSeq End-to-End INT8 Quantization: System Architecture Document
**Unified Architecture: Jetfire Data Flow + INT-FlashAttention + I-BERT Integer Arithmetic**

---

## 1. Visão Geral e Fundamentação Teórica

Este documento estabelece o projeto arquitetural, o grafo computacional quantizado, o mapeamento de tipos de dados, o fluxo de tensores e as estratégias de execução em hardware (CUDA sm_75 / Turing RTX 2080 Ti e CPU x86_64) para o modelo **PARSeq** (`strhub.models.parseq`), unificando e adaptando as três referências fundamentais da literatura:

1. **Jetfire (arXiv:2403.12422 / ICML 2024):**
   * *INT8 Data Flow* de ponta a ponta eliminando a barreira de latência do paradigma *Quantize-Compute-Dequantize (QCD)*.
   * Quantização em blocos 2D (*Per-Block Quantization*, blocos $32 \times 32$ ou $64 \times 64$) que confina *outliers* de canais e tokens a ladrilhos locais, impedindo que um pico contamine o restante da sequência.
   * Treinamento Totalmente Quantizado (*Fully Quantized Training - FQT*): propagação de gradientes de ativação $\nabla X$ e pesos $\nabla W$ calculados em 8 bits durante o backward pass, com atualização dos pesos mestres em FP32.
   * Fusão de operadores não-lineares (GELU, LayerNorm, Residual Add) limitados por largura de banda de memória (HBM/VRAM).

2. **INT-FlashAttention (arXiv:2409.16997):**
   * Mecanismo de atenção executado inteiramente em INT8 com acumulador INT32.
   * Escalonamento vetorial por token para queries e keys ($S_Q \in \mathbb{R}^N, S_K \in \mathbb{R}^M$) e escalonamento constante/por bloco para values ($S_V$).
   * *Online Softmax* quantizado calculado diretamente nos registradores/SRAM, sem materializar a matriz de atenção $N \times M$ em VRAM:
     $$S_i^{(j)} = \text{diag}(S_{Q_i}) (Q_i K_j^T) \text{diag}(S_{K_j}) \cdot \frac{1}{\sqrt{d}}$$
     $$P_i^{(j)} = \lfloor R \cdot \exp(S_i^{(j)} - m_i^{(j)}) \rceil \in \mathbb{I}_8$$
     $$O_i^{(j)} = \text{diag}(e^{m_i^{(j-1)} - m_i^{(j)}}) O_i^{(j-1)} + P_i^{(j)} V_j$$
     $$O_i = \text{diag}(l_i)^{-1} O_i \cdot S_V$$

3. **I-BERT (ICML 2021 / arXiv:2101.01321):**
   * Aritmética puramente inteira (*integer-only arithmetic*) com decomposição de base e *bit-shifting* ($>> z$), eliminando a necessidade de unidades de ponto flutuante:
     * `i-GELU`: aproximação polinomial de 2º grau:
       $$i\text{-}GELU(x) = x \cdot \frac{1}{2}\left[1 + \text{sgn}(x)\left(a(\text{clip}(|x/\sqrt{2}|, \max=-b) + b)^2 + 1\right)\right]$$
       onde $a = -0.2888$ e $b = -1.769$ (erro máximo $< 0.019$).
     * `i-Softmax`: decomposição $z = \lfloor -\tilde{x} / \ln 2 \rfloor$ e $p = \tilde{x} + z\ln 2$ com $L(p) = 0.3585(p + 1.353)^2 + 0.344$, onde $\exp(x) \approx L(p) \gg z$.
     * `i-LayerNorm`: raiz quadrada inteira de $\sigma$ via iteração de Newton-Raphson convergente em no máximo 4 iterações para INT32:
       $$x_{k+1} = \lfloor (x_k + \lfloor n / x_k \rfloor) / 2 \rfloor$$

---

## 2. Grafo Computacional Quantizado do PARSeq

O PARSeq combina um **Encoder ViT** e um **Decodificador Autoregressivo Permutado** com modelagem de linguagem por permutação (PLM) e cabeçote de projeção de vocabulário.

```
[Imagem de Entrada (B, 3, H, W)]
            │
            ▼
┌───────────────────────────────────────────────┐
│ ENCODER (Vision Transformer ViT)              │
│  ├─ PatchEmbed: Linear/Conv INT8              │
│  ├─ Blocos Transformer (x12):                 │
│  │   ├─ JetfireFusedLayerNorm / ILayerNorm    │
│  │   ├─ INT-FlashAttention (Self-Attention)   │
│  │   │   └─ Q, K, V em INT8, GEMM INT32,      │
│  │   │      Online Softmax na SRAM            │
│  │   ├─ Fused Residual Add                    │
│  │   ├─ JetfireFusedLayerNorm / ILayerNorm    │
│  │   ├─ MLP (Linear1 INT8 -> GELU -> Linear2) │
│  │   │   └─ JetfireFusedGELU / IGELU          │
│  │   └─ Fused Residual Add                    │
│  └─ Norm Final: ILayerNorm / FusedLayerNorm   │
└───────────────────────────────────────────────┘
            │
            ▼ Memory [B, N_patches, d]
┌───────────────────────────────────────────────┐
│ DECODER (Two-Stream Permuted Transformer)     │
│  ├─ Token Embedding & Pos Queries             │
│  ├─ Camadas Decoder (x1 a x6):                │
│  │   ├─ Two-Stream Self-Attention INT8:       │
│  │   │   ├─ Query Stream (com máscara PLM/AR) │
│  │   │   └─ Content Stream (com máscara PLM)  │
│  │   ├─ Cross-Attention INT8 (atende Memory)  │
│  │   ├─ FeedForward (Linear1 INT8 -> Linear2) │
│  │   └─ Norms: FusedLayerNorm / ILayerNorm    │
│  └─ Norm Final Decoder                        │
└───────────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────────┐
│ CHAR HEAD                                     │
│  └─ Linear INT8: (embed_dim -> num_tokens - 2)│
│     (cuBLASLt INT8 Tensor Cores sm_75 / DP4A) │
└───────────────────────────────────────────────┘
            │
            ▼ Logits [B, Max_Len, Vocab]
```

---

## 3. Matriz de Tipos de Dados por Nó

| Componente / Nó | Formato de Dados Entrada | Formato de Dados Operação | Acumulador / Registrador | Formato Saída | Fator de Escala |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Patch Embedding** | FP32 / INT8 | INT8 GEMM / Conv | INT32 | INT8 | $S_{patch} \in \mathbb{R}$ |
| **Linear Layers (QKV, Proj, MLP)** | INT8 | INT8 WMMA / `_int_mm` | INT32 $\to$ FP32 | INT8 | $S_X \in \mathbb{R}^{M \times 1}, S_W \in \mathbb{R}^{1 \times N}$ |
| **Linear Jetfire (Blocos $32 \times 32$)** | INT8 ($32 \times 32$) | INT8 WMMA | INT32 local | INT8 | $S_{ij} = \max(\|X_{ij}\|)/127$ |
| **Attention Scores $Q K^T$** | $Q \in \mathbb{I}_8, K \in \mathbb{I}_8$ | INT8 GEMM | INT32 | FP32 (SRAM) | $S_{Q_i} \in \mathbb{R}^{B_r}, S_{K_j} \in \mathbb{R}^{B_c}$ |
| **Online Softmax / i-Softmax** | FP32 / Fixed-Point | $\exp$ ou $L(p) \gg z$ | FP32 / INT32 | $P \in \mathbb{I}_8$ | $S_P = 1/127$ |
| **Attention Output $P V$** | $P \in \mathbb{I}_8, V \in \mathbb{I}_8$ | INT8 GEMM | INT32 $\to$ FP32 | FP32 / INT8 | $S_V \in \mathbb{R}$ |
| **LayerNorm / i-LayerNorm** | INT8 | Integer / FP32 local | INT32 (Newton sqrt) | INT8 | $S_{out} = \max(\|Y\|)/127$ |
| **GELU / i-GELU** | INT8 | Polinomial 2º grau | INT32 / FP32 local | INT8 | $S_{out} = \max(\|Y\|)/127$ |
| **Residual Add** | INT8 + INT8 | Adição em SRAM | INT32 / FP32 | INT8 | Re-escalonado por $S_Y$ |
| **Backward $\nabla X$ (FQT)** | $\nabla Y \in \mathbb{I}_8, W \in \mathbb{I}_8$ | INT8 GEMM | INT32 $\to$ FP32 | FP32 | $S_{\nabla Y} \times S_W$ |
| **Backward $\nabla W$ (FQT)** | $\nabla Y \in \mathbb{I}_8, X \in \mathbb{I}_8$ | INT8 GEMM | INT32 $\to$ FP32 | FP32 | $S_{\nabla Y} \times S_X$ |

---

## 4. Hardware Backend & Estratégias de Despacho

### 4.1. NVIDIA Turing (GeForce RTX 2080 Ti / sm_75)
* **Tensor Cores INT8:** Suporta instruções WMMA (`mma.sync.aligned.m8n8k32.row.col`) e `dp4a` (dot product de 4 elementos INT8 com acumulador INT32).
* **Alinhamento:** Matrizes são alinhadas para dimensões múltiplas de 8 ou 16 para máxima taxa de ocupação dos Tensor Cores via `torch._int_mm`.
* **Kernels Triton:** Fused LayerNorm, Fused GELU e INT-FlashAttention rodam nativamente com compilação `@triton.jit` otimizada para Turing.
* **Memória SRAM:** Manutenção de $Q_i$, $K_j$, $V_j$ e acumuladores $m_i$, $l_i$, $O_i$ dentro da memória compartilhada de 64KB por SM da RTX 2080 Ti.

### 4.2. CPU (x86_64, AVX2, AVX-512 VNNI, oneDNN)
* **oneDNN / PyTorch Dynamic Quant:** Módulos `RealHardwareInt8Linear` e `dynamic` despacham para o backend nativo oneDNN, explorando instruções VNNI (`vpdpbusd`).
* **Integer-Only Dyadic:** No modo `ibert`, a CPU executa raiz quadrada inteira de Newton-Raphson e avaliações polinomiais sem conversão para ponto flutuante.

### 4.3. ONNX Runtime
* Modelos exportados via `PARSeqQuantizer.export_onnx` e quantizados via `PARSeqQuantizer.export_onnx_int8` utilizam nós `MatMulInteger` / `QLinearMatMul` nativos.
* Compatível com `CUDAExecutionProvider` e `CPUExecutionProvider`.

---

## 5. Guia de Execução no Servidor com GPU (Dual RTX 2080 Ti)

Para executar o treinamento FQT, os testes e o benchmark completo no servidor `thanos`, execute os comandos abaixo diretamente no terminal do servidor:

### 5.1. Benchmark Comparativo Completo (CUDA)
```bash
python3 quantize_parseq.py pretrained=parseq --compare_all --device cuda --iterations 50 --batch_size 1
```

### 5.2. Treinamento FQT INT8 de Ponta a Ponta (Jetfire)
```bash
python3 quantize_parseq.py pretrained=parseq --method jetfire_fqt --block_size 32 \
    --finetune --epochs 5 --max_steps 1000 --lr 1e-4 --device cuda \
    --save_path outputs/parseq_jetfire_fqt.pt
```

### 5.3. Avaliação de Acurácia nos Datasets Padrão de STR
```bash
# Avaliação Baseline FP32
python3 test.py pretrained=parseq --device cuda

# Avaliação com Unified INT8 (Jetfire + INT-FlashAttention)
python3 test.py pretrained=parseq --quant_method unified_int8 --block_size 32 --device cuda

# Avaliação com INT-FlashAttention
python3 test.py pretrained=parseq --quant_method int_flashattn --block_size 32 --device cuda

# Avaliação com I-BERT Integer-Only
python3 test.py pretrained=parseq --quant_method ibert --device cuda
```

### 5.4. Exportação e Benchmark ONNX INT8
```bash
python3 -c "
from strhub.models.quantization import PARSeqQuantizer
PARSeqQuantizer.export_onnx('pretrained=parseq', 'outputs/parseq_nar.onnx', mode='nar')
PARSeqQuantizer.export_onnx_int8('outputs/parseq_nar.onnx', 'outputs/parseq_nar_int8.onnx')
print('ONNX INT8 exportado com sucesso!')
"
```
