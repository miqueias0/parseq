# Guia de Referência de Executáveis Python (.py)

Este documento cataloga de forma exaustiva todos os scripts Python executáveis (`*.py`) presentes no repositório **i_parseq**, detalhando a função de cada script, seus argumentos e flags de linha de comando, tipo de dado, valores padrão e obrigatoriedade.

---

## Sumário dos Executáveis

| Script | Categoria | Tipo de CLI | Finalidade Principal |
|---|---|---|---|
| [`read.py`](file:///c:/Users/administrador/Desktop/i_parseq/read.py) | Inferência | `argparse` | Leitura de placas/textos em imagens individuais via checkpoint PyTorch. |
| [`benchmark_fps.py`](file:///c:/Users/administrador/Desktop/i_parseq/benchmark_fps.py) | Benchmark | `argparse` | Medição de FPS e latência pura em GPU sem overhead de I/O de disco. |
| [`bench.py`](file:///c:/Users/administrador/Desktop/i_parseq/bench.py) | Benchmark | `Hydra` | Perfilamento de FLOPs, ativações e tempo de inferência via PyTorch Benchmark. |
| [`test.py`](file:///c:/Users/administrador/Desktop/i_parseq/test.py) | Avaliação | `argparse` | Avaliação de acurácia, NED e confiança em modelos `.ckpt`, `.onnx` ou `.engine`. |
| [`train.py`](file:///c:/Users/administrador/Desktop/i_parseq/train.py) | Treinamento | `Hydra` | Treinamento do modelo PARSeq via PyTorch Lightning. |
| [`tune.py`](file:///c:/Users/administrador/Desktop/i_parseq/tune.py) | Otimização | `Hydra` | Ajuste de hiperparâmetros (Hyperparameter Tuning) com Ray Tune. |
| [`run_experiment.py`](file:///c:/Users/administrador/Desktop/i_parseq/run_experiment.py) | Orquestração | `argparse` | Execução completa da campanha experimental científica (PyTorch -> TRT -> Figuras). |
| [`tools/run_full_matrix.py`](file:///c:/Users/administrador/Desktop/i_parseq/tools/run_full_matrix.py) | Orquestração | `argparse` | Pipeline completo da matriz de variantes (M0 a M6), ONNX, TensorRT, testes e benchmarks. |
| [`tools/export_onnx.py`](file:///c:/Users/administrador/Desktop/i_parseq/tools/export_onnx.py) | Exportação | `argparse` | Exportador para ONNX (M0..M6) com fusão de kernels e INT-FlashAttention. |
| [`tools/build_tensorrt.py`](file:///c:/Users/administrador/Desktop/i_parseq/tools/build_tensorrt.py) | Compilação | `argparse` | Compilador de engines TensorRT (FP32, FP16, INT8, INT8_IO). |
| [`tools/calibrate.py`](file:///c:/Users/administrador/Desktop/i_parseq/tools/calibrate.py) | Quantização | `argparse` | Calibração pós-treinamento (PTQ), coleta de escalas e cálculo de SQNR. |
| [`tools/finetune_qat.py`](file:///c:/Users/administrador/Desktop/i_parseq/tools/finetune_qat.py) | Quantização | `argparse` | Fine-tuning com Quantization-Aware Training (QAT) para variante M6. |
| [`tools/evaluate_alpr.py`](file:///c:/Users/administrador/Desktop/i_parseq/tools/evaluate_alpr.py) | Avaliação | `argparse` | Avaliação aprofundada de acurácia ALPR, intervalos de confiança 95%, CER e NED. |
| [`tools/benchmark_tensorrt.py`](file:///c:/Users/administrador/Desktop/i_parseq/tools/benchmark_tensorrt.py) | Benchmark | `argparse` | Benchmark de latência, percentis (p50/p95/p99) e FPS de engines TensorRT. |
| [`tools/benchmark_pytorch.py`](file:///c:/Users/administrador/Desktop/i_parseq/tools/benchmark_pytorch.py) | Benchmark | `argparse` | Benchmark PyTorch com decomposição por módulo (H2D, Encoder, Decoder, Head, D2H). |
| [`tools/benchmark_video.py`](file:///c:/Users/administrador/Desktop/i_parseq/tools/benchmark_video.py) | Benchmark | `argparse` | Benchmark em stream de vídeo contínuo a 30/60 FPS e análise de dropped frames. |
| [`tools/collect_scientific_results.py`](file:///c:/Users/administrador/Desktop/i_parseq/tools/collect_scientific_results.py) | Orquestração | Script Direto | Consolidação de resultados em `metrics.json` e geração de relatórios. |
| [`tools/generate_figures_tables.py`](file:///c:/Users/administrador/Desktop/i_parseq/tools/generate_figures_tables.py) | Relatórios | Script Direto | Geração de 15 figuras de alta resolução e 12 tabelas LaTeX acadêmicas. |
| [`tools/validate_int_flashattention.py`](file:///c:/Users/administrador/Desktop/i_parseq/tools/validate_int_flashattention.py) | Validação | Script Direto | Validação matemática e numérica do INT-FlashAttention (arXiv:2409.16997v2). |
| [`tools/validate_sage_attention.py`](file:///c:/Users/administrador/Desktop/i_parseq/tools/validate_sage_attention.py) | Validação | Script Direto | Validação matemática, de suavização de chaves e numérica do SageAttention (arXiv:2410.02367v9). |
| [`tools/validate_trt_plugins.py`](file:///c:/Users/administrador/Desktop/i_parseq/tools/validate_trt_plugins.py) | Validação | Script Direto | Validação dos plugins customizados TensorRT (`IPluginV2DynamicExt`). |
| [`tools/validate_onnx.py`](file:///c:/Users/administrador/Desktop/i_parseq/tools/validate_onnx.py) | Validação | `argparse` | Auditoria de concordância e fidelidade numérica entre PyTorch e ONNX. |
| [`tools/validate_tensorrt.py`](file:///c:/Users/administrador/Desktop/i_parseq/tools/validate_tensorrt.py) | Validação | `argparse` | Validação de concordância numérica e semântica entre PyTorch e TensorRT. |
| [`tools/layer_sensitivity.py`](file:///c:/Users/administrador/Desktop/i_parseq/tools/layer_sensitivity.py) | Análise | `argparse` | Análise de sensibilidade por camada e seleção de aproximações polinomiais. |
| [`tools/create_lmdb_dataset.py`](file:///c:/Users/administrador/Desktop/i_parseq/tools/create_lmdb_dataset.py) | Dataset | `Fire` | Criação de datasets no formato binário LMDB a partir de pastas de imagens. |
| [`tools/filter_lmdb.py`](file:///c:/Users/administrador/Desktop/i_parseq/tools/filter_lmdb.py) | Dataset | `argparse` | Filtragem de imagens corrompidas ou com dimensões inferiores ao limiar. |
| [`tools/coco_2_converter.py`](file:///c:/Users/administrador/Desktop/i_parseq/tools/coco_2_converter.py) | Dataset | `argparse` | Conversão e recorte de caixas de anotação para o dataset COCO Text / TextOCR. |
| [`tools/lsvt_converter.py`](file:///c:/Users/administrador/Desktop/i_parseq/tools/lsvt_converter.py) | Dataset | `argparse` | Conversão e recorte de anotações do dataset LSVT. |
| [`tools/textocr_converter.py`](file:///c:/Users/administrador/Desktop/i_parseq/tools/textocr_converter.py) | Dataset | `argparse` | Conversão do dataset TextOCR com retificação heurística de rotação de texto. |
| [`tools/openvino_converter.py`](file:///c:/Users/administrador/Desktop/i_parseq/tools/openvino_converter.py) | Dataset | `argparse` | Conversão de anotações OpenVINO / Open Images. |
| [`tools/test_abinet_lm_acc.py`](file:///c:/Users/administrador/Desktop/i_parseq/tools/test_abinet_lm_acc.py) | Avaliação | `argparse` | Avaliação da acurácia isolada do módulo de linguagem (LM) do ABINet. |
| [`tools/case_sensitive_str_datasets_converter.py`](file:///c:/Users/administrador/Desktop/i_parseq/tools/case_sensitive_str_datasets_converter.py) | Dataset | Positional | Conversão de datasets case-sensitive estruturados em `label/` e `IMG/`. |
| [`tools/mlt19_converter.py`](file:///c:/Users/administrador/Desktop/i_parseq/tools/mlt19_converter.py) | Dataset | Positional | Conversão do dataset MLT19 filtrando scripts em alfabeto latino. |
| [`tools/art_converter.py`](file:///c:/Users/administrador/Desktop/i_parseq/tools/art_converter.py) | Dataset | Sem args | Conversão do dataset ART (Arbitrary-shaped Text) com caminhos locais. |
| [`tools/coco_text_converter.py`](file:///c:/Users/administrador/Desktop/i_parseq/tools/coco_text_converter.py) | Dataset | Sem args | Conversão simples de anotações do COCO-Text para formato `lmdb.txt`. |

---

## 1. Scripts Principais da Raiz do Projeto

### 1.1 `read.py`
Carrega um modelo pré-treinado ou checkpoint customizado e executa inferência em uma ou mais imagens informadas pelo usuário.

```bash
python read.py <checkpoint> --images <img1.jpg> <img2.jpg> [--device cuda]
```

| Argumento / Flag | Tipo | Status | Padrão | Descrição |
|---|---|---|---|---|
| `checkpoint` | `str` (Positional) | **Obrigatório** | — | Caminho para o checkpoint (`.ckpt`) ou identificador de modelo pré-treinado (`pretrained=<id>`). |
| `--images` | `str` (Lista) | **Obrigatório\*** | `None` | Uma ou mais imagens a serem processadas (`nargs='+'`). |
| `--device` | `str` | Opcional | `'cuda'` | Dispositivo de computação (`'cuda'` ou `'cpu'`). |
| *Argumentos extras* | `kwargs` | Opcional | — | Parâmetros dinâmicos repassados para inicialização do modelo (ex: `--decode_ar False`). |

*\*Observação: Embora tecnicamente `argparse` declare `--images` como opção, na prática o script itera sobre a lista; sem imagens, nenhuma inferência é realizada.*

---

### 1.2 `benchmark_fps.py`
Realiza benchmark de taxa de quadros (FPS) e latência em GPU pura. Aloca tensores dummy diretamente na VRAM para eliminar ruídos de leitura de disco ou CPU.

```bash
python benchmark_fps.py <checkpoint> [--batch_size 1] [--device cuda] [--iterations 1000] [--warmup 100]
```

| Argumento / Flag | Tipo | Status | Padrão | Descrição |
|---|---|---|---|---|
| `checkpoint` | `str` (Positional) | **Obrigatório** | — | Caminho para o checkpoint (`.ckpt`) ou modelo pré-treinado. |
| `--batch_size` | `int` | Opcional | `1` | Tamanho do lote de inferência (usar 1 para simular tempo real de câmera). |
| `--device` | `str` | Opcional | `'cuda'` | Dispositivo para execução do benchmark (`'cuda'` ou `'cpu'`). |
| `--iterations` | `int` | Opcional | `1000` | Número de iterações medidas com cronômetro sincronizado via CUDA. |
| `--warmup` | `int` | Opcional | `100` | Número de iterações de aquecimento prévio da GPU para estabilizar clocks. |

---

### 1.3 `bench.py`
Perfilador estruturado com Hydra que calcula contagem exata de FLOPs, ativações e velocidade de inferência usando `torch.utils.benchmark.Timer`.

```bash
python bench.py [model=<config>] [device=cuda] [range=True]
```

| Argumento / Override | Tipo | Status | Padrão | Descrição |
|---|---|---|---|---|
| `config_name` | Hydra Config | Opcional | `bench` | Arquivo base em `configs/bench.yaml`. |
| `device` | `str` | Opcional | `'cuda'` | Dispositivo de teste (`'cuda'` ou `'cpu'`). |
| `range` | `bool` | Opcional | `False` | Se `True`, avalia latência variando comprimentos de decodificação de 1 a 25. |
| `model` | Config Group | Opcional | `parseq` | Configuração da arquitetura do modelo (instanciada via Hydra). |
| `data.img_size` | `[int, int]` | Opcional | `[32, 128]` | Dimensão da imagem dummy de entrada `[H, W]`. |

---

### 1.4 `test.py`
Utilitário completo de avaliação que unifica testes de acurácia de leitura de placas e textos para checkpoints PyTorch (`.ckpt`, `.pt`, `.pth`), grafos ONNX (`.onnx`) e engines compiladas TensorRT (`.engine`).

```bash
python test.py <checkpoint> [--dataset VeSV_pad] [--batch_size 64] [--variant m1] [--device cuda]
```

| Argumento / Flag | Tipo | Status | Padrão | Descrição |
|---|---|---|---|---|
| `checkpoint` | `str` (Positional) | **Obrigatório** | — | Caminho do arquivo a testar (`.ckpt`, `.onnx` ou `.engine`). |
| `--base_checkpoint` | `str` | Opcional | `pretrained/parseq_alpr_98.5.ckpt` | Checkpoint PyTorch base fornecendo tokenizer, dicionário de caracteres e hparams ao testar arquivos `.onnx` ou `.engine`. |
| `--dataset`, `--datasets` | `str` (Lista) | Opcional | `None` | Datasets de teste a avaliar (ex: `VeSV_pad`, `RodoSol_pad`, `UFPR_ALPR_pad`). |
| `--data_root` | `str` | Opcional | `'data'` | Diretório raiz dos datasets LMDB. |
| `--batch_size` | `int` | Opcional | `64` | Tamanho do batch durante a avaliação. |
| `--num_workers` | `int` | Opcional | `0` | Número de subprocessos no DataLoader. |
| `--device` | `str` | Opcional | `'cuda'` | Dispositivo de computação (`'cuda'` ou `'cpu'`). |
| `--variant` | `str` | Opcional | `None` | Variante de quantização/arquitetura: `m0` (FP32 AR), `m1` (FP32 NAR), `m2` (FP16 NAR), `m3` (INT8 Naive), `m4` (INT8 W8A8), `m5` (INT8 Integer-Only PTQ), `m6` (INT8 Integer-Only QAT). |
| `--use_int_flashattention` | Flag booleana | Opcional | `False` | Ativa o algoritmo INT-FlashAttention (arXiv:2409.16997v2) com GEMMs INT8 e online softmax. |
| `--use_sage_attention` | Flag booleana | Opcional | `False` | Ativa o algoritmo SageAttention (arXiv:2410.02367v9) com Key Smoothing e GEMMs INT8. |
| `--sage_mode` | `str` | Opcional | `'sageattn_b'` | Modo de precisão do SageAttention: `sageattn_b` (matriz V em FP16/FP32) ou `sageattn_vb` (matriz V quantizada em INT8). |
| `--max_samples` | `int` | Opcional | `None` | Limite máximo de amostras avaliadas (útil para auditoria rápida). |
| `--cased` | Flag booleana | Opcional | `False` | Diferenciação entre letras maiúsculas e minúsculas na avaliação. |
| `--punctuation` | Flag booleana | Opcional | `False` | Considera pontuação no charset de teste. |
| `--new` | Flag booleana | Opcional | `False` | Avalia nos novos datasets benchmark de Scene Text Recognition. |
| `--rotation` | `int` | Opcional | `0` | Rotação prévia da imagem (graus anti-horário). |

---

### 1.5 `train.py`
Script oficial de treinamento do PARSeq gerenciado via PyTorch Lightning e Hydra. Suporta precisão mista (FP16/BF16), DDP (Distributed Data Parallel) multi-GPU e Stochastic Weight Averaging (SWA).

```bash
python train.py [dataset=VeSV_pad] [model=parseq] [trainer.devices=1]
```

| Argumento / Override | Tipo | Status | Padrão | Descrição |
|---|---|---|---|---|
| `config_name` | Hydra Config | Opcional | `main` | Arquivo base em `configs/main.yaml`. |
| `dataset` | Config Group | Opcional | Definido em config | Dataset de treino e validação. |
| `model` | Config Group | Opcional | `parseq` | Configuração da arquitetura do modelo. |
| `pretrained` | `str` | Opcional | `None` | Inicializa pesos a partir de um checkpoint pré-treinado. |
| `ckpt_path` | `str` | Opcional | `None` | Caminho para retomar o treinamento (resume). |
| `trainer.accelerator` | `str` | Opcional | `'gpu'` | Dispositivo acelerador (`'gpu'`, `'cpu'`). |
| `trainer.devices` | `int` | Opcional | `1` | Número de GPUs utilizadas no treino. |
| `trainer.max_epochs` | `int` | Opcional | Definido em config | Número máximo de épocas. |
| `data.root_dir` | `str` | Opcional | `'data'` | Caminho do diretório de dados. |

---

### 1.6 `tune.py`
Script de exploração e sintonia fina de hiperparâmetros (HMM/Taxa de aprendizado/Weight Decay) usando Ray Tune com AxSearch e parada prematura por mediana (`MedianStoppingRule`).

```bash
python tune.py [model=parseq] [tune.lr.min=1e-5] [tune.lr.max=1e-3]
```

| Argumento / Override | Tipo | Status | Padrão | Descrição |
|---|---|---|---|---|
| `config_name` | Hydra Config | Opcional | `tune` | Arquivo base em `configs/tune.yaml`. |
| `tune.lr.min` | `float` | Opcional | Definido em config | Limite inferior da taxa de aprendizado na escala logarítmica. |
| `tune.lr.max` | `float` | Opcional | Definido em config | Limite superior da taxa de aprendizado na escala logarítmica. |
| `trainer.max_epochs` | `int` | Opcional | Definido em config | Número de épocas por tentativa de trial. |

---

### 1.7 `run_experiment.py`
Script de orquestração unificado que executa a campanha experimental científica de ponta a ponta (PyTorch -> QAT -> Exportação ONNX -> Compilação TensorRT -> Benchmarking de Vídeo -> Geração de Gráficos e Tabelas).

```bash
python run_experiment.py [--checkpoint pretrained/parseq_alpr_98.5.ckpt] [--dataset VeSV_pad] [--samples 300] [--quick]
```

| Argumento / Flag | Tipo | Status | Padrão | Descrição |
|---|---|---|---|---|
| `--checkpoint` | `str` | Opcional | `pretrained/parseq_alpr_98.5.ckpt` | Caminho do checkpoint base para inicialização dos experimentos. |
| `--dataset` | `str` | Opcional | `'VeSV_pad'` | Nome do dataset ALPR de referência para os testes. |
| `--samples` | `int` | Opcional | `300` | Quantidade de amostras para o estágio de avaliação de acurácia. |
| `--quick` | Flag booleana | Opcional | `False` | Executa em modo rápido (reduz iterações de benchmark para validação ágil). |

---

## 2. Pipeline de Quantização, Compilação e Orquestração (`tools/`)

### 2.1 `tools/run_full_matrix.py`
Orquestrador central da matriz científica de experimentos. Itera automaticamente por todas as configurações de modelo (M0 FP32 AR, M1 FP32 NAR, M2 FP16 NAR, M3 INT8 Naive, M4 INT8 W8A8, M5 INT8 Integer-Only PTQ, M6 INT8 Integer-Only QAT, com e sem INT-FlashAttention e com diferentes níveis de fusão de kernels).

```bash
python tools/run_full_matrix.py [--samples 50] [--force]
```

| Argumento / Flag | Tipo | Status | Padrão | Descrição |
|---|---|---|---|---|
| `--samples` | `int` | Opcional | `50` | Número de amostras de placas avaliadas para calcular acurácia exata, NED e CER de cada variante. |
| `--force` | Flag booleana | Opcional | `False` | Força a re-exportação de todos os grafos ONNX e a recompilação de todas as engines TensorRT, mesmo que os arquivos já existam em disco. |

---

### 2.2 `tools/export_onnx.py`
Exporta o modelo PARSeq do PyTorch para o formato ONNX. Permite controlar a variante arquitetural, versão do opset, flags de fusão de nós (shapes, MHA, MLP, LayerNorm), ativação de INT-FlashAttention e injeção de nós customizados de plugins TensorRT.

```bash
python tools/export_onnx.py --variant m1 --output onnx/parseq_nar.onnx [--fusion_level all] [--use_int_flashattention]
```

| Argumento / Flag | Tipo | Status | Padrão | Descrição |
|---|---|---|---|---|
| `--checkpoint` | `str` | Opcional | `pretrained/parseq_alpr_98.5.ckpt` | Caminho do checkpoint PyTorch de origem. |
| `--variant` | `str` | Opcional | `'m1'` | Variante a exportar (`m0`, `m1`, `m2`, `m3`, `m4`, `m5`, `m6`). |
| `--output` | `str` | Opcional | `'onnx/parseq_nar.onnx'` | Caminho do arquivo `.onnx` de saída. |
| `--opset` | `int` | Opcional | `18` | Versão do ONNX Opset utilizado na exportação. |
| `--all` | Flag booleana | Opcional | `False` | Exporta em lote todas as variantes de M0 a M6. |
| `--fuse_shapes` | Flag booleana | Opcional | `False` | Funde nós redundantes de shape, reshape, cast e gather via ONNX Simplifier. |
| `--fuse_mha` | Flag booleana | Opcional | `False` | Emite padrão canônico de Softmax permitindo fusão em kernel FMHA/FlashAttention pelo TensorRT. |
| `--fuse_mlp` | Flag booleana | Opcional | `False` | Emite GELU canônico para permitir fusão de GEMM FC1 + GELU + FC2. |
| `--fuse_layernorm` | Flag booleana | Opcional | `False` | Emite LayerNorm canônico para fusão no kernel Myelin LayerNorm do TensorRT. |
| `--use_int_flashattention` | Flag booleana | Opcional | `False` | Ativa a atenção em blocos inteiros com INT-FlashAttention (arXiv:2409.16997v2). |
| `--use_sage_attention` | Flag booleana | Opcional | `False` | Ativa a atenção com SageAttention (arXiv:2410.02367v9) com Key Smoothing e quantização INT8. |
| `--sage_mode` | `str` | Opcional | `'sageattn_b'` | Modo do SageAttention: `sageattn_b` ou `sageattn_vb`. |
| `--use_plugin` | Flag booleana | Opcional | `False` | Emite nós compatíveis com plugins customizados de C++/CUDA (`IPluginV2DynamicExt`). |
| `--fusion_level` | `str` | Opcional | `'none'` | Nível pré-configurado de fusão: `none`, `shapes`, `mha`, `mlp` ou `all`. |

---

### 2.3 `tools/build_tensorrt.py`
Compila modelos ONNX para engines otimizadas do NVIDIA TensorRT (`.engine`), suportando calibração INT8, modos FP16/FP32 e perfis dinâmicos de execução.

```bash
python tools/build_tensorrt.py --onnx onnx/model.onnx --output trt/model.engine --precision int8 [--max_batch 64]
# Ou compilar todos:
python tools/build_tensorrt.py --all [--max_batch 64]
```

| Argumento / Flag | Tipo | Status | Padrão | Descrição |
|---|---|---|---|---|
| `--onnx` | `str` | **Obrigatório\*** | `None` | Caminho do modelo `.onnx` de entrada. |
| `--output` | `str` | **Obrigatório\*** | `None` | Caminho do arquivo `.engine` gerado. |
| `--precision` | `str` | Opcional | `'fp32'` | Precisão alvo da engine: `fp32`, `fp16`, `int8` ou `int8_io`. |
| `--max_batch` | `int` | Opcional | `64` | Tamanho máximo de lote permitido pelo Optimization Profile. |
| `--all` | Flag booleana | Opcional | `False` | Constrói automaticamente todas as engines pré-definidas da matriz M0..M6. |

*\*Observação: `--onnx` e `--output` são obrigatórios caso a flag `--all` não seja informada.*

---

### 2.4 `tools/calibrate.py`
Executa calibração pós-treinamento (Post-Training Quantization - PTQ) capturando tensores de ativação em camadas lineares, de atenção e convolucionais, calculando escalas simétricas e métricas de saturação e SQNR (Signal-to-Quantization-Noise Ratio).

```bash
python tools/calibrate.py [--checkpoint pretrained/parseq_alpr_98.5.ckpt] [--dataset VeSV_pad] [--output_dir results/calibration]
```

| Argumento / Flag | Tipo | Status | Padrão | Descrição |
|---|---|---|---|---|
| `--checkpoint` | `str` | Opcional | `pretrained/parseq_alpr_98.5.ckpt` | Caminho do checkpoint com os pesos pré-treinados do modelo. |
| `--dataset` | `str` | Opcional | `'VeSV_pad'` | Nome do dataset utilizado para alimentar amostras de calibração. |
| `--data_dir` | `str` | Opcional | `'data'` | Diretório base dos datasets LMDB. |
| `--output_dir` | `str` | Opcional | `'results/calibration'` | Diretório para gravação das estatísticas em `calibration_stats.json`. |

---

### 2.5 `tools/finetune_qat.py`
Realiza treinamento consciente de quantização (Quantization-Aware Training - QAT) inserindo fake-quantizers nos pesos e ativações para recuperar perdas de acurácia causadas pela quantização INT8 (gerando a variante M6).

```bash
python tools/finetune_qat.py [--epochs 3] [--lr 1e-5] [--batch_size 32] [--output pretrained/parseq_alpr_qat_m6.ckpt]
```

| Argumento / Flag | Tipo | Status | Padrão | Descrição |
|---|---|---|---|---|
| `--checkpoint` | `str` | Opcional | `pretrained/parseq_alpr_98.5.ckpt` | Checkpoint de partida em ponto flutuante. |
| `--output` | `str` | Opcional | `pretrained/parseq_alpr_qat_m6.ckpt` | Caminho do checkpoint de saída para salvar os pesos fine-tunados. |
| `--dataset` | `str` | Opcional | `'VeSV_pad'` | Dataset utilizado no fine-tuning. |
| `--epochs` | `int` | Opcional | `3` | Número de épocas de ajuste QAT. |
| `--lr` | `float` | Opcional | `1e-5` | Taxa de aprendizado inicial (learning rate). |
| `--batch_size` | `int` | Opcional | `32` | Tamanho do mini-batch durante o treinamento. |
| `--max_train_samples`| `int` | Opcional | `2000` | Limite de amostras de treino por época. |
| `--val_samples` | `int` | Opcional | `200` | Limite de amostras de validação por época. |

---

## 3. Avaliação e Benchmarking Científico (`tools/`)

### 3.1 `tools/evaluate_alpr.py`
Módulo de avaliação científica de reconhecimento de placas de veículos (ALPR). Avalia taxa exata de placas (Exact Plate Accuracy) com intervalo de confiança de 95% via bootstrap, Normalized Edit Distance (NED), Character Error Rate (CER), e distribuição de erros por tamanho de placa e matriz de confusão de caracteres.

```bash
python tools/evaluate_alpr.py --checkpoint trt/parseq_m1_nar_fp16.engine [--dataset VeSV_pad] [--batch_size 64]
```

| Argumento / Flag | Tipo | Status | Padrão | Descrição |
|---|---|---|---|---|
| `--checkpoint` | `str` | Opcional | `pretrained/parseq_alpr_98.5.ckpt` | Caminho do modelo (`.ckpt`, `.onnx` ou `.engine`). |
| `--model` | `str` | Opcional | `None` | Caminho explícito para o modelo (sobrescreve `--checkpoint`). |
| `--variant` | `str` | Opcional | `'m1'` | Variante arquitetural quando avaliado a partir de `.ckpt` (`m0`..`m6`). |
| `--base_checkpoint` | `str` | Opcional | `pretrained/parseq_alpr_98.5.ckpt` | Checkpoint base para carregar tokenizer e metadados quando o alvo for `.onnx` ou `.engine`. |
| `--dataset` | `str` | Opcional | `'VeSV_pad'` | Nome do dataset LMDB a avaliar. |
| `--data_root` | `str` | Opcional | `'data'` | Diretório base dos datasets. |
| `--batch_size` | `int` | Opcional | `64` | Tamanho do lote durante o teste. |
| `--max_samples` | `int` | Opcional | `None` | Número máximo de amostras avaliadas (avaliação completa se `None`). |
| `--device` | `str` | Opcional | `'cuda'` | Dispositivo (`'cuda'` ou `'cpu'`). |
| `--use_int_flashattention` | Flag booleana | Opcional | `False` | Habilita módulo INT-FlashAttention durante a avaliação do checkpoint. |
| `--use_sage_attention` | Flag booleana | Opcional | `False` | Habilita módulo SageAttention durante a avaliação do checkpoint PyTorch. |
| `--sage_mode` | `str` | Opcional | `'sageattn_b'` | Modo do SageAttention (`sageattn_b` ou `sageattn_vb`). |
| `--output` | `str` | Opcional | `None` | Caminho de arquivo JSON para gravação estruturada das métricas. |

---

### 3.2 `tools/benchmark_tensorrt.py`
Benchmark de alto rigor científico para medição de latência e throughput de engines TensorRT compiladas. Coleta média, mediana, desvio padrão, percentis estatísticos (p50, p90, p95, p99) e audita os tipos de camadas compiladas na engine.

```bash
python tools/benchmark_tensorrt.py --engine trt/parseq_m1_nar_fp16.engine [--batch_size 1] [--iterations 200]
```

| Argumento / Flag | Tipo | Status | Padrão | Descrição |
|---|---|---|---|---|
| `--engine` | `str` | **Obrigatório** | — | Caminho para a engine TensorRT (`.engine`). |
| `--batch_size` | `int` | Opcional | `1` | Tamanho do lote durante as medições de inferência. |
| `--warmup` | `int` | Opcional | `50` | Número de iterações de aquecimento da GPU. |
| `--iterations` | `int` | Opcional | `200` | Número de repetições cronometradas com CUDA Events. |
| `--output` | `str` | Opcional | `None` | Caminho de arquivo JSON para salvar os resultados brutos. |

---

### 3.3 `tools/benchmark_pytorch.py`
Mede latência e taxa de transferência de modelos PyTorch diretamente na GPU e de ponta a ponta (End-to-End), discriminando o tempo consumido por cada estágio: transferência Host-to-Device (H2D), Encoder ViT, Decoder autoregressivo/não-autoregressivo, Head linear de predição e transferência Device-to-Host (D2H), além do pico de VRAM.

```bash
python tools/benchmark_pytorch.py [--variant m1] [--batch_size 1] [--iterations 200]
```

| Argumento / Flag | Tipo | Status | Padrão | Descrição |
|---|---|---|---|---|
| `--checkpoint` | `str` | Opcional | `pretrained/parseq_alpr_98.5.ckpt` | Caminho para o checkpoint PyTorch base. |
| `--variant` | `str` | Opcional | `'m1'` | Variante arquitetural a instanciar e testar (`m0`..`m6`). |
| `--batch_size` | `int` | Opcional | `1` | Tamanho do lote de teste. |
| `--device` | `str` | Opcional | `'cuda'` | Dispositivo de computação (`'cuda'` ou `'cpu'`). |
| `--warmup` | `int` | Opcional | `50` | Iterações de warmup prévio. |
| `--iterations` | `int` | Opcional | `200` | Iterações de medição sincronizada. |
| `--output` | `str` | Opcional | `None` | Caminho de arquivo JSON para salvar o relatório de breakdown. |

---

### 3.4 `tools/benchmark_video.py`
Simula processamento contínuo de fluxo de vídeo em tempo real (taxa de entrada de 30 FPS ou 60 FPS), medindo se a engine mantém o framerate ou gera quadros descartados (dropped frames), além de monitorar o consumo de VRAM ao longo do stream.

```bash
python tools/benchmark_video.py --engine trt/parseq_m1_nar_fp16.engine [--frames 1000] [--video path/video.mp4]
```

| Argumento / Flag | Tipo | Status | Padrão | Descrição |
|---|---|---|---|---|
| `--engine` | `str` | **Obrigatório** | — | Caminho para a engine TensorRT (`.engine`). |
| `--video` | `str` | Opcional | `None` | Caminho de arquivo de vídeo real (`.mp4`, `.avi`). Se não fornecido, gera stream sintético a partir do dataset. |
| `--frames` | `int` | Opcional | `1000` | Número total de quadros a processar na simulação. |
| `--output` | `str` | Opcional | `None` | Caminho de arquivo JSON para salvar estatísticas de streaming. |

---

### 3.5 `tools/collect_scientific_results.py`
Consolida e atualiza automaticamente os resultados de métricas mestre em `results/metrics.json` e `results/metrics.csv`, disparando em seguida a geração de gráficos e tabelas acadêmicas.

```bash
python tools/collect_scientific_results.py
```
*Não requer parâmetros adicionais via linha de comando; localiza automaticamente as engines geradas e executa auditoria consolidada.*

---

### 3.6 `tools/generate_figures_tables.py`
Lê os dados consolidados em `results/metrics.json` e gera todas as 15 figuras em PNG de alta resolução (300 DPI) para o artigo científico (ex: curvas de Pareto latência vs acurácia, decomposição por camada, sensibilidade de quantização) e 12 tabelas acadêmicas completas formatadas em CSV e código fonte LaTeX (`.tex`).

```bash
python tools/generate_figures_tables.py
```
*Não requer parâmetros adicionais via linha de comando; consome automaticamente o arquivo de métricas do repositório.*

---

## 4. Validação Científica e Auditoria Numérica (`tools/`)

### 4.1 `tools/validate_int_flashattention.py`
Executa a suíte de auditoria científica do algoritmo INT-FlashAttention (arXiv:2409.16997v2). Executa 4 testes rigorosos:
1. **Auditoria de Dtypes e Acumuladores**: Garante que os tensores $Q, K, V$ sejam `torch.int8` e que os GEMMs operem com acumulador em `torch.int32`.
2. **Invariância de Tiling por Blocos**: Confirma que o particionamento em blocos produz o mesmo resultado que o bloco completo (diferença $\le 10^{-5}$).
3. **Paridade Numérica e Acurácia**: Compara a saída quantizada contra a atenção em float32 (Cosine Similarity $\ge 0.999$, MRE $\le 3\%$).
4. **Decomposição de Erro em 7 Estágios**: Mede SQNR, erro da matriz de scores $S$, da matriz de probabilidades $P$, do produto parcial $PV$ e da saída final.

```bash
python tools/validate_int_flashattention.py
```
*Não requer parâmetros de linha de comando; executa a suíte de testes completa com validações asseridas em código.*

---

### 4.2 `tools/validate_trt_plugins.py`
Valida a implementação dos plugins customizados em C++/CUDA (`IPluginV2DynamicExt`) compilados para a arquitetura NVIDIA Ampere (`sm_86`).
1. **Paridade Numérica**: Testa cada kernel CUDA isoladamente (INT-FlashAttention, LayerNorm, GELU, Softmax) contra as referências PyTorch.
2. **Auditoria de Camadas Compiladas**: Inspeciona a engine TensorRT e verifica a fusão de 342 nós em 161 camadas compiladas (exatamente 12 nós de atenção fundidos).
3. **Acurácia ALPR com Plugins**: Valida que a acurácia exata de leitura de placas permanece em 90.00%.

```bash
python tools/validate_trt_plugins.py
```
*Não requer parâmetros de linha de comando; executa a auditoria completa de plugins e camadas.*

---

### 4.3 `tools/validate_sage_attention.py`
Executa a suíte de auditoria científica e matemática do algoritmo SageAttention (arXiv:2410.02367v9). Conduz 5 auditorias rigorosas:
1. **Invariância Matemática do Key Smoothing sob Softmax**: Comprova que $\text{Softmax}(Q K^T) = \text{Softmax}(Q (K - \bar{k})^T)$ com erro residual $< 10^{-6}$ e Cosine Similarity $> 0.999999$.
2. **Auditoria de Tipos de Dados e Acumuladores de Hardware**: Verifica que os tensores $Q, K$ são efetivamente quantizados para `torch.int8` e que o GEMM acumula em `torch.int32`.
3. **Invariância de Tiling por Blocos**: Valida que o processamento em tiles com online softmax produz resultados indistinguíveis do cálculo completo para blocos de tamanho $\{16, 32, 64, 128\}$.
4. **Paridade Numérica e Fidelidade contra FP32 no Encoder PARSeq**: Mede a fidelidade das ativações de saída do bloco do Encoder contra FP32 ($\text{CosSim} > 0.999$, $\text{SQNR} > 41$ dB).
5. **Resiliência a Outliers de Ativação vs INT-FlashAttention**: Injeta outliers sintéticos em canais específicos de $K$ e demonstra empiricamente que o Key Smoothing supera o INT-FlashAttention sem suavização por mais de $+15$ dB de SQNR.

```bash
python tools/validate_sage_attention.py
```
*Não requer parâmetros de linha de comando; executa as 5 auditorias com asserts científicos.*

---

### 4.4 `tools/validate_onnx.py`
Audita a concordância numérica entre a saída do grafo ONNX executado pelo ONNX Runtime e a saída de referência do modelo PyTorch original.

```bash
python tools/validate_onnx.py [--checkpoint pretrained/parseq_alpr_98.5.ckpt] [--onnx onnx/parseq_nar.onnx] [--variant m1] [--samples 20]
```

| Argumento / Flag | Tipo | Status | Padrão | Descrição |
|---|---|---|---|---|
| `--checkpoint` | `str` | Opcional | `pretrained/parseq_alpr_98.5.ckpt` | Checkpoint PyTorch de referência. |
| `--onnx` | `str` | Opcional | `'onnx/parseq_nar.onnx'` | Grafo `.onnx` a ser verificado no ONNX Runtime. |
| `--variant` | `str` | Opcional | `'m1'` | Variante do modelo associada ao grafo. |
| `--samples` | `int` | Opcional | `20` | Quantidade de tensores aleatórios para calcular Max Diff, MSE e Cosine Sim. |

---

### 4.5 `tools/validate_tensorrt.py`
Compara as predições e os logits gerados pela engine TensorRT contra o modelo de referência PyTorch, identificando divergências e informando o índice do primeiro caractere divergente caso ocorra descasamento.

```bash
python tools/validate_tensorrt.py --engine trt/parseq_m1_nar_fp16.engine [--precision fp16] [--samples 20]
```

| Argumento / Flag | Tipo | Status | Padrão | Descrição |
|---|---|---|---|---|
| `--engine` | `str` | **Obrigatório** | — | Caminho para a engine TensorRT (`.engine`). |
| `--checkpoint` | `str` | Opcional | `pretrained/parseq_alpr_98.5.ckpt` | Checkpoint PyTorch de referência. |
| `--precision` | `str` | Opcional | `'fp32'` | Precisão esperada para verificação dos limiares de tolerância. |
| `--samples` | `int` | Opcional | `20` | Número de amostras de teste para a comparação. |

---

### 4.6 `tools/layer_sensitivity.py`
Avalia individualmente cada camada não-linear do Transformer (GELU e Softmax) sob diferentes estratégias de quantização inteira (I-BERT, I-ViT e IPTQ), gerando um relatório em CSV e a atribuição ótima de aproximação polinomial em JSON.

```bash
python tools/layer_sensitivity.py [--checkpoint pretrained/parseq_alpr_98.5.ckpt] [--dataset VeSV_pad] [--samples 128]
```

| Argumento / Flag | Tipo | Status | Padrão | Descrição |
|---|---|---|---|---|
| `--checkpoint` | `str` | Opcional | `pretrained/parseq_alpr_98.5.ckpt` | Checkpoint de partida. |
| `--dataset` | `str` | Opcional | `'VeSV_pad'` | Dataset utilizado para alimentar ativações reais nas camadas. |
| `--samples` | `int` | Opcional | `128` | Quantidade de amostras para o estudo de sensibilidade. |
| `--output_dir` | `str` | Opcional | `'results/sensitivity'` | Diretório para gravação do relatório `layer_sensitivity_report.csv`. |

---

## 5. Utilitários de Dados e Conversores (`tools/`)

### 5.1 `tools/create_lmdb_dataset.py`
Converte uma pasta com imagens e um arquivo de texto com anotações (`gtFile`) para uma base binária no formato LMDB de alta velocidade para treinamento e teste.

```bash
python tools/create_lmdb_dataset.py <inputPath> <gtFile> <outputPath> [--checkValid True]
```

| Parâmetro | Tipo | Status | Padrão | Descrição |
|---|---|---|---|---|
| `inputPath` | `str` (Positional) | **Obrigatório** | — | Diretório raiz onde estão localizadas as imagens. |
| `gtFile` | `str` (Positional) | **Obrigatório** | — | Arquivo de texto contendo as linhas `caminho_imagem rotulo_texto`. |
| `outputPath` | `str` (Positional) | **Obrigatório** | — | Caminho do diretório de destino da base LMDB criada. |
| `--checkValid` | `bool` | Opcional | `True` | Se `True`, valida se cada arquivo de imagem pode ser aberto e decodificado antes de gravar. |

---

### 5.2 `tools/filter_lmdb.py`
Filtra bases LMDB existentes, removendo imagens corrompidas ou com altura/largura inferiores a um limiar mínimo, consolidando os dados válidos em uma nova base LMDB.

```bash
python tools/filter_lmdb.py <inputs...> --output <output_lmdb> [--min_image_dim 8]
```

| Argumento / Flag | Tipo | Status | Padrão | Descrição |
|---|---|---|---|---|
| `inputs` | `str` (Lista Positional) | **Obrigatório** | — | Um ou mais caminhos de bases LMDB de entrada (`nargs='+'`). |
| `--output` | `str` | **Obrigatório** | — | Diretório de destino da nova base LMDB filtrada. |
| `--min_image_dim` | `int` | Opcional | `8` | Dimensão mínima (pixels) de largura e altura; amostras menores são descartadas. |

---

### 5.3 `tools/coco_2_converter.py`
Processa e recorta anotações do dataset COCO Text / TextOCR a partir dos metadados brutos e coordenadas de caixas delimitadoras.

```bash
python tools/coco_2_converter.py <root_path> <n_proc>
```

| Argumento / Flag | Tipo | Status | Padrão | Descrição |
|---|---|---|---|---|
| `root_path` | `str` (Positional) | **Obrigatório** | — | Diretório raiz contendo imagens e arquivos de anotação do TextOCR. |
| `n_proc` | `int` (Positional) | **Obrigatório** | `1` | Número de processos paralelos (multiprocessing) para acelerar o corte das imagens. |

---

### 5.4 `tools/lsvt_converter.py`
Processa anotações do dataset LSVT (Large-scale Street View Text), filtrando caracteres não-latinos e gerando imagens cortadas por palavra.

```bash
python tools/lsvt_converter.py <root_path> <n_proc>
```

| Argumento / Flag | Tipo | Status | Padrão | Descrição |
|---|---|---|---|---|
| `root_path` | `str` (Positional) | **Obrigatório** | — | Diretório raiz do dataset LSVT. |
| `n_proc` | `int` (Positional) | **Obrigatório** | `1` | Quantidade de processos paralelos para a extração. |

---

### 5.5 `tools/textocr_converter.py`
Converte anotações do dataset TextOCR, com algoritmo heurístico para retificação de rotação de texto orientado em 90°, 180° ou 270°.

```bash
python tools/textocr_converter.py <root_path> <n_proc> [--rectify_pose]
```

| Argumento / Flag | Tipo | Status | Padrão | Descrição |
|---|---|---|---|---|
| `root_path` | `str` (Positional) | **Obrigatório** | — | Diretório raiz contendo o TextOCR. |
| `n_proc` | `int` (Positional) | **Obrigatório** | `1` | Número de processos simultâneos. |
| `--rectify_pose` | Flag booleana | Opcional | `False` | Rotaciona heuristicamente imagens com texto inclinado/vertical para alinhamento horizontal. |

---

### 5.6 `tools/openvino_converter.py`
Processa anotações OpenVINO do dataset Open Images, recortando caixas delimitadoras legíveis em idioma inglês.

```bash
python tools/openvino_converter.py <root_path> <n_proc>
```

| Argumento / Flag | Tipo | Status | Padrão | Descrição |
|---|---|---|---|---|
| `root_path` | `str` (Positional) | **Obrigatório** | — | Diretório contendo imagens e anotações OpenVINO. |
| `n_proc` | `int` (Positional) | **Obrigatório** | `1` | Número de processos em paralelo. |

---

### 5.7 `tools/test_abinet_lm_acc.py`
Avalia isoladamente a acurácia do modelo de linguagem (LM) do ABINet utilizando o texto ground-truth como entrada com máscaras de ruído.

```bash
python tools/test_abinet_lm_acc.py <checkpoint> [--data_root data] [--batch_size 512]
```

| Argumento / Flag | Tipo | Status | Padrão | Descrição |
|---|---|---|---|---|
| `checkpoint` | `str` (Positional) | **Obrigatório** | — | Pesos pré-treinados do ABINet (ex: `best-train-abinet.pth`). |
| `--data_root` | `str` | Opcional | `'data'` | Diretório base dos dados. |
| `--batch_size` | `int` | Opcional | `512` | Tamanho do lote de teste. |
| `--num_workers` | `int` | Opcional | `4` | Quantidade de workers do DataLoader. |
| `--new` | Flag booleana | Opcional | `False` | Testa nos novos datasets de benchmark. |
| `--device` | `str` | Opcional | `'cuda'` | Dispositivo de computação (`'cuda'` ou `'cpu'`). |

---

### 5.8 `tools/case_sensitive_str_datasets_converter.py`
Converte datasets de texto sensíveis a maiúsculas/minúsculas organizados em pastas `label/` (com arquivos `.txt` individuais) e `IMG/` (com arquivos `.jpg` ou `.png`) gerando o índice `lmdb.txt`.

```bash
python tools/case_sensitive_str_datasets_converter.py <dataset_dir>
```

| Argumento | Tipo | Status | Descrição |
|---|---|---|---|
| `sys.argv[1]` | `str` (Positional) | **Obrigatório** | Diretório base contendo as subpastas `label/` e `IMG/`. |

---

### 5.9 `tools/mlt19_converter.py`
Utilitário para processamento do dataset multi-língue MLT19, filtrando apenas as amostras cujo script esteja identificado como `Latin` ou `Symbols`.

```bash
python tools/mlt19_converter.py <dataset_root>
```

| Argumento | Tipo | Status | Descrição |
|---|---|---|---|
| `sys.argv[1]` | `str` (Positional) | **Obrigatório** | Diretório raiz onde se encontra o arquivo `gt.txt` do MLT19. |

---

### 5.10 `tools/art_converter.py`
Processa o arquivo `train_task2_labels.json` do dataset ART (Arbitrary-shaped Text), descartando textos ilegíveis ou não-latinos e gerando o índice tabulado `gt.txt`.

```bash
python tools/art_converter.py
```
*Não possui argumentos de linha de comando; lê diretamente os arquivos `train_task2_labels.json` no diretório de execução atual.*

---

### 5.11 `tools/coco_text_converter.py`
Script simples que lê os arquivos `train_words_gt.txt` e `val_words_gt.txt` do dataset COCO-Text e gera os arquivos formatados `train_lmdb.txt` e `val_lmdb.txt`.

```bash
python tools/coco_text_converter.py
```
*Não possui argumentos de linha de comando; processa os arquivos de anotação com nomes pré-definidos no diretório de trabalho.*
