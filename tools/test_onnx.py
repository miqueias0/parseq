#!/usr/bin/env python3
"""High-Performance ONNX Model Evaluator for PARSeq / STRHub.
Extracts maximum throughput and latency from exported .onnx models using batched execution
on ONNX Runtime (CUDA / TensorRT / CPU), measuring:
- Exact Plate / Sequence Accuracy (%)
- 1 - Normalized Edit Distance (1 - NED %)
- Sequence Confidence (%)
- Average Label Length
- Batch & Per-Sample Latency (ms)
- Inference Throughput (FPS)
"""

import os
import sys
import glob

# Ensure CUDA and cuDNN (libcudnn.so.9) from Python site-packages/nvidia are in LD_LIBRARY_PATH
def _configure_cuda_and_cudnn_paths():
    if os.environ.get("_PARSEQ_CUDA_ENV_SET") == "1":
        return

    candidate_dirs = []
    # Search site-packages from sys.path and known virtualenvs
    search_sp = list(sys.path)
    for venv_pattern in (
        "/home/mon25/modelos/parseq/.venv/lib/python*/site-packages",
        "/home/mon25/modelos/parseq_full_int8/.venv/lib/python*/site-packages",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".venv", "lib", "python*", "site-packages"),
    ):
        search_sp.extend(glob.glob(venv_pattern))

    for p in set(search_sp):
        if "site-packages" in p and os.path.isdir(p):
            nvidia_dir = os.path.join(p, "nvidia")
            if os.path.isdir(nvidia_dir):
                candidate_dirs.extend(glob.glob(os.path.join(nvidia_dir, "*", "lib")))
            trt_libs = os.path.join(p, "tensorrt_libs")
            if os.path.isdir(trt_libs):
                candidate_dirs.append(trt_libs)

    for cuda_dir in ("/usr/local/cuda/lib64", "/usr/local/cuda-12.4/lib64", "/usr/local/cuda/targets/x86_64-linux/lib"):
        if os.path.isdir(cuda_dir):
            candidate_dirs.append(cuda_dir)

    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    existing_paths = set(os.path.abspath(x) for x in current_ld.split(":") if x)
    to_add = [os.path.abspath(d) for d in candidate_dirs if os.path.isdir(d) and os.path.abspath(d) not in existing_paths]

    if to_add:
        unique_to_add = list(dict.fromkeys(to_add))
        new_ld = ":".join(unique_to_add + ([current_ld] if current_ld else []))
        os.environ["LD_LIBRARY_PATH"] = new_ld
        os.environ["_PARSEQ_CUDA_ENV_SET"] = "1"
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cur_py = os.environ.get("PYTHONPATH", "")
        if repo_root not in cur_py.split(":"):
            os.environ["PYTHONPATH"] = f"{repo_root}:{cur_py}" if cur_py else repo_root
        try:
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception:
            pass

_configure_cuda_and_cudnn_paths()

import time
import string
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

# Ensure UTF-8 output encoding across platforms
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from nltk import edit_distance
import numpy as np
import torch
import onnxruntime as ort

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args


@dataclass
class Result:
    dataset: str
    num_samples: int
    accuracy: float
    ned: float
    confidence: float
    label_length: float
    latency_ms: float = 0.0
    fps: float = 0.0


def print_results_table(results: List[Result], file=None):
    if not results:
        print("Nenhum resultado para exibir.", file=file)
        return
    w = max(map(len, [r.dataset for r in results]))
    w = max(w, len('Dataset'), len('Combined'))

    print('| {:<{w}} | # samples | Accuracy | 1 - NED | Confidence | Label Length | Latency (ms) | Throughput (FPS) |'.format('Dataset', w=w), file=file)
    print('|:{:-<{w}}:|----------:|---------:|--------:|-----------:|-------------:|-------------:|-----------------:|'.format('----', w=w), file=file)

    c = Result('Combined', 0, 0, 0, 0, 0, 0, 0)
    total_time_combined = 0.0

    for res in results:
        c.num_samples += res.num_samples
        c.accuracy += res.num_samples * res.accuracy
        c.ned += res.num_samples * res.ned
        c.confidence += res.num_samples * res.confidence
        c.label_length += res.num_samples * res.label_length
        c.latency_ms += res.num_samples * res.latency_ms
        if res.fps > 0:
            total_time_combined += res.num_samples / res.fps

        print(
            f'| {res.dataset:<{w}} | {res.num_samples:>9} | {res.accuracy:>8.2f} | {res.ned:>7.2f} '
            f'| {res.confidence:>10.2f} | {res.label_length:>12.2f} | {res.latency_ms:>12.2f} | {res.fps:>16.1f} |',
            file=file,
        )

    if c.num_samples > 0:
        c.accuracy /= c.num_samples
        c.ned /= c.num_samples
        c.confidence /= c.num_samples
        c.label_length /= c.num_samples
        c.latency_ms /= c.num_samples
        c.fps = c.num_samples / total_time_combined if total_time_combined > 0 else 0.0

    print('|-{:-<{w}}-|-----------|----------|---------|------------|--------------|--------------|------------------|'.format('----', w=w), file=file)
    print(
        f'| {c.dataset:<{w}} | {c.num_samples:>9} | {c.accuracy:>8.2f} | {c.ned:>7.2f} '
        f'| {c.confidence:>10.2f} | {c.label_length:>12.2f} | {c.latency_ms:>12.2f} | {c.fps:>16.1f} |',
        file=file,
    )


class FastONNXEvaluator:
    """High-throughput batched evaluator using ONNX Runtime."""
    def __init__(
        self,
        onnx_path: str,
        base_system,
        device: str = "cuda",
        provider: Optional[str] = None
    ):
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"Arquivo ONNX não encontrado em: {onnx_path}")

        self.tokenizer = base_system.tokenizer
        self.charset_adapter = base_system.charset_adapter
        self.hparams = base_system.hparams
        self.onnx_path = onnx_path

        # Setup execution providers
        available_providers = ort.get_available_providers()
        selected_providers = []

        if provider is not None:
            p_lower = provider.lower()
            if "tensorrt" in p_lower:
                selected_providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            elif "cuda" in p_lower:
                selected_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif "cpu" in p_lower:
                selected_providers = ["CPUExecutionProvider"]
            else:
                selected_providers = [provider]
        else:
            if "cpu" in device.lower():
                selected_providers = ["CPUExecutionProvider"]
            elif "CUDAExecutionProvider" in available_providers:
                selected_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif "TensorrtExecutionProvider" in available_providers:
                selected_providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                selected_providers = ["CPUExecutionProvider"]

        # Filter against available providers
        providers_to_use = [p for p in selected_providers if p in available_providers]
        if not providers_to_use:
            providers_to_use = ["CPUExecutionProvider"]

        # Configure session options for maximum performance
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_opts.log_severity_level = 3  # WARNING only

        # Configure provider options
        provider_options = []
        for p in providers_to_use:
            if p == "CUDAExecutionProvider":
                dev_id = 0
                if ":" in device:
                    try:
                        dev_id = int(device.split(":")[-1])
                    except ValueError:
                        dev_id = 0
                provider_options.append({
                    "device_id": str(dev_id),
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": "DEFAULT",
                    "do_copy_in_default_stream": "1",
                })
            else:
                provider_options.append({})

        print(f"Carregando Modelo ONNX: {onnx_path}")
        print(f"Providers selecionados: {providers_to_use}")

        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_opts,
            providers=providers_to_use,
            provider_options=provider_options
        )

        self.active_provider = self.session.get_providers()[0]
        print(f"Provider ativo no ONNX Runtime: {self.active_provider}")

        # Input and output tensor names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        input_shape = self.session.get_inputs()[0].shape

        # Detect dynamic vs fixed batch
        self.is_dynamic_batch = (
            len(input_shape) > 0
            and (input_shape[0] is None or isinstance(input_shape[0], str) or input_shape[0] <= 0)
        )
        self.fixed_batch_size = input_shape[0] if not self.is_dynamic_batch else None
        print(f"Entrada: '{self.input_name}' {input_shape} | Saída: '{self.output_name}' | Dynamic Batch: {self.is_dynamic_batch}")

    def evaluate_batch(self, images: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Executa inferência em lote com temporização de alta resolução."""
        bs = images.shape[0]

        # Convert images to contiguous float32 numpy array
        if images.is_cuda:
            imgs_np = images.contiguous().cpu().numpy()
        else:
            imgs_np = images.contiguous().numpy()

        if imgs_np.dtype != np.float32:
            imgs_np = imgs_np.astype(np.float32)

        # Handle models with fixed batch size if needed
        if not self.is_dynamic_batch and self.fixed_batch_size is not None and bs != self.fixed_batch_size:
            fixed_bs = self.fixed_batch_size
            logits_list = []
            total_infer_time = 0.0
            for start in range(0, bs, fixed_bs):
                end = min(start + fixed_bs, bs)
                chunk = imgs_np[start:end]
                actual_chunk_len = chunk.shape[0]
                if actual_chunk_len < fixed_bs:
                    # Pad to fixed_bs
                    pad_shape = list(chunk.shape)
                    pad_shape[0] = fixed_bs - actual_chunk_len
                    chunk = np.concatenate([chunk, np.zeros(pad_shape, dtype=np.float32)], axis=0)

                t0 = time.perf_counter()
                out_np = self.session.run([self.output_name], {self.input_name: chunk})[0]
                t_chunk = time.perf_counter() - t0
                total_infer_time += t_chunk
                logits_list.append(torch.from_numpy(out_np[:actual_chunk_len]))
            logits = torch.cat(logits_list, dim=0)
            return logits, total_infer_time

        # Fast path: full batch executed in a single vectorized ORT call
        t0 = time.perf_counter()
        out_np = self.session.run([self.output_name], {self.input_name: imgs_np})[0]
        t_infer = time.perf_counter() - t0
        logits = torch.from_numpy(out_np)
        return logits, t_infer

    def test_step(self, batch) -> Dict[str, Any]:
        """Processa um lote e calcula acurácia, NED, confiança e tempo de inferência."""
        images, labels = batch
        bs = images.shape[0]

        logits, t_infer = self.evaluate_batch(images)

        probs = logits.softmax(-1)
        preds, prob_tuples = self.tokenizer.decode(probs)

        correct = 0
        ned = 0.0
        confidence = 0.0
        label_length = 0

        for pred, prob, gt in zip(preds, prob_tuples, labels):
            confidence += prob.prod().item()
            pred = self.charset_adapter(pred)
            ned += edit_distance(pred, gt) / max(len(pred), len(gt), 1)
            if pred == gt:
                correct += 1
            label_length += len(pred)

        return {
            "num_samples": bs,
            "correct": correct,
            "ned": ned,
            "confidence": confidence,
            "label_length": label_length,
            "infer_time_s": t_infer,
        }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Avaliador de Alta Performance para Modelos PARSeq Exportados em ONNX (.onnx)")
    parser.add_argument('model', nargs='?', default=None, help="Caminho do arquivo .onnx a ser testado")
    parser.add_argument('--model_path', '--onnx', dest='model_opt', default=None, help="Caminho do arquivo .onnx alternativo")
    parser.add_argument('--base_checkpoint', default='pretrained/parseq_alpr_98.5.ckpt',
                        help="Checkpoint base (.ckpt) para carregamento do tokenizer, charset e dimensões de entrada")
    parser.add_argument('--data_root', default='data', help="Diretório raiz dos dados LMDB")
    parser.add_argument('--batch_size', type=int, default=64, help="Tamanho do lote para inferência máxima paralela")
    parser.add_argument('--num_workers', type=int, default=4, help="Número de workers no DataLoader")
    parser.add_argument('--cased', action='store_true', default=False, help="Comparação considerando maiúsculas e minúsculas")
    parser.add_argument('--punctuation', action='store_true', default=False, help="Verificar pontuação")
    parser.add_argument('--new', action='store_true', default=False, help="Avaliar nos novos datasets de benchmark")
    parser.add_argument('--rotation', type=int, default=0, help="Ângulo de rotação da imagem em graus (anti-horário)")
    parser.add_argument('--device', default='cuda',
                        help="Dispositivo alvo: 'cuda', 'cuda:0', ou 'cpu'")
    parser.add_argument('--provider', default=None, choices=['cuda', 'cpu', 'tensorrt'],
                        help="Execution Provider explícito para ONNX Runtime (padrão: automático)")
    parser.add_argument('--datasets', '--dataset', nargs='+', default=None,
                        help="Datasets específicos para avaliação (ex: VeSV_pad RodoSol_pad UFPR_ALPR_pad)")
    parser.add_argument('--max_samples', type=int, default=None, help="Limite máximo de amostras avaliadas por dataset")
    parser.add_argument('--output', '--log_file', default=None, help="Arquivo customizado para salvar o relatório de resultados")

    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)

    chosen_model = args.model if args.model is not None else args.model_opt
    if chosen_model is None:
        parser.error("É necessário especificar o arquivo .onnx como argumento posicional ou através de --model/--onnx.")

    if not chosen_model.endswith(".onnx"):
        print(f"[Aviso] O arquivo '{chosen_model}' não possui extensão .onnx. Prosseguindo...")

    # Build charset_test
    charset_test = string.digits + string.ascii_lowercase
    if args.cased:
        charset_test += string.ascii_uppercase
    if args.punctuation:
        charset_test += string.punctuation
    kwargs.update({'charset_test': charset_test})

    # Load base system for tokenizer and dataset hparams
    if not os.path.exists(args.base_checkpoint):
        # Check fallback
        fallback_candidates = [
            "pretrained/parseq_alpr_98.5.ckpt",
            "pretrained/parseq_alpr_qat_m6.ckpt",
            "checkpoints/parseq_alpr_98.5.ckpt",
        ]
        found_ckpt = None
        for fc in fallback_candidates:
            if os.path.exists(fc):
                found_ckpt = fc
                break
        if found_ckpt:
            print(f"[Aviso] Checkpoint base '{args.base_checkpoint}' não encontrado. Usando fallback: '{found_ckpt}'")
            args.base_checkpoint = found_ckpt
        else:
            raise FileNotFoundError(f"Checkpoint base não encontrado: {args.base_checkpoint}. Forneça via --base_checkpoint.")

    print(f"Carregando tokenizer e hiperparâmetros de: {args.base_checkpoint}")
    base_sys = load_from_checkpoint(args.base_checkpoint, **kwargs).eval().cpu()

    evaluator = FastONNXEvaluator(
        onnx_path=chosen_model,
        base_system=base_sys,
        device=args.device,
        provider=args.provider
    )

    hp = base_sys.hparams
    datamodule = SceneTextDataModule(
        root_dir=args.data_root,
        train_dir='_unused_',
        img_size=hp.img_size,
        max_label_length=hp.max_label_length,
        charset_train=hp.charset_train,
        charset_test=hp.charset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=False,
        rotation=args.rotation,
    )

    if args.datasets:
        test_set = sorted(set(args.datasets))
    else:
        raw_set = list(SceneTextDataModule.TEST_BENCHMARK_SUB + SceneTextDataModule.TEST_BENCHMARK)
        if args.new:
            raw_set += list(SceneTextDataModule.TEST_NEW)
        test_set = sorted([s for s in set(raw_set) if os.path.exists(os.path.join(args.data_root, 'test', s))])
        if not test_set:
            test_set = ['VeSV_pad']

    print(f"Datasets selecionados para avaliação: {test_set}")
    print(f"Batch size: {args.batch_size} | Num workers: {args.num_workers}")

    results = {}
    max_width = max(map(len, test_set)) if test_set else 10

    for name, dataloader in datamodule.test_dataloaders(test_set).items():
        total = 0
        correct = 0
        ned = 0.0
        confidence = 0.0
        label_length = 0
        total_infer_time_s = 0.0

        desc_label = f'{name:>{max_width}}'
        for imgs, labels in tqdm(iter(dataloader), desc=desc_label):
            if args.max_samples and total >= args.max_samples:
                break

            step_res = evaluator.test_step((imgs, labels))
            total += step_res["num_samples"]
            correct += step_res["correct"]
            ned += step_res["ned"]
            confidence += step_res["confidence"]
            label_length += step_res["label_length"]
            total_infer_time_s += step_res["infer_time_s"]

        accuracy = 100.0 * correct / total if total > 0 else 0.0
        mean_ned = 100.0 * (1.0 - ned / total) if total > 0 else 0.0
        mean_conf = 100.0 * confidence / total if total > 0 else 0.0
        mean_label_length = label_length / total if total > 0 else 0.0
        latency_ms = (total_infer_time_s * 1000.0) / total if total > 0 else 0.0
        fps = total / total_infer_time_s if total_infer_time_s > 0 else 0.0

        results[name] = Result(
            dataset=name,
            num_samples=total,
            accuracy=accuracy,
            ned=mean_ned,
            confidence=mean_conf,
            label_length=mean_label_length,
            latency_ms=latency_ms,
            fps=fps,
        )

    log_path = args.output if args.output is not None else (chosen_model + '.log.txt')
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

    result_list = [results[s] for s in test_set if s in results]

    print("\n" + "=" * 92)
    print("RELATÓRIO DE AVALIAÇÃO ONNX:")
    print("=" * 92)
    print_results_table(result_list, file=sys.stdout)

    try:
        with open(log_path, 'w', encoding='utf-8') as f:
            print("Datasets:", file=f)
            print_results_table(result_list, file=f)
        print(f"\nResultados salvos com sucesso em: {log_path}")
    except Exception as e:
        print(f"\n[Aviso] Falha ao gravar log em '{log_path}': {e}")


if __name__ == '__main__':
    main()
