#!/usr/bin/env python3
"""High-Performance TensorRT Engine Evaluator for PARSeq / STRHub.
Extracts maximum throughput and latency from exported .engine models using batched execution
directly on GPU Tensor Cores via TensorRT runtime, measuring:
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
import tensorrt as trt

from strhub.quant.plugins.trt_plugins import register_parseq_plugins, get_trt_logger
register_parseq_plugins()

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


class FastTensorRTEvaluator:
    """High-throughput batched evaluator directly utilizing TensorRT execution contexts and GPU streams."""
    def __init__(
        self,
        engine_path: str,
        base_system,
        device: str = "cuda",
        profile_index: int = 0,
        execution_mode: str = "pipelined"
    ):
        if not os.path.isfile(engine_path):
            raise FileNotFoundError(f"Arquivo TensorRT Engine não encontrado em: {engine_path}")

        if not torch.cuda.is_available():
            print("[Aviso] torch.cuda.is_available() reportou False no ambiente atual. Tentando inicializar TensorRT em GPU...")

        self.tokenizer = base_system.tokenizer
        self.charset_adapter = base_system.charset_adapter
        self.hparams = base_system.hparams
        self.engine_path = engine_path
        self.device = torch.device(device if "cuda" in device else "cuda")
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(self.device)
            except Exception:
                pass
        self.profile_index = profile_index
        self.execution_mode = execution_mode.lower().strip()

        logger = get_trt_logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        print(f"Carregando TensorRT Engine: {engine_path}")
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Falha ao desserializar engine TensorRT em: {engine_path}")

        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream(device=self.device)

        # Identify input and output tensor names
        self.input_name = "images"
        self.output_name = "logits"

        if hasattr(self.engine, "num_io_tensors"):
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                mode = self.engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT:
                    self.input_name = name
                elif mode == trt.TensorIOMode.OUTPUT:
                    self.output_name = name

        # Inspect data types
        in_dtype = self.engine.get_tensor_dtype(self.input_name)
        self.torch_in_dtype = torch.float16 if in_dtype == trt.DataType.HALF else torch.float32

        out_dtype = self.engine.get_tensor_dtype(self.output_name)
        self.torch_out_dtype = torch.float16 if out_dtype == trt.DataType.HALF else torch.float32

        # Inspect dynamic profile dimensions
        min_shape, opt_shape, max_shape = self.engine.get_tensor_profile_shape(self.input_name, self.profile_index)
        self.min_batch = min_shape[0]
        self.opt_batch = opt_shape[0]
        self.max_batch = max_shape[0]
        self.img_h = max_shape[2]
        self.img_w = max_shape[3]

        print(
            f"Perfil de Otimização TensorRT (index {self.profile_index}): "
            f"batch=[min:{self.min_batch}, opt:{self.opt_batch}, max:{self.max_batch}] | "
            f"input='{self.input_name}' ({in_dtype}) | output='{self.output_name}' ({out_dtype})"
        )

        # Probe output dimensions with opt_batch
        self.context.set_input_shape(self.input_name, (self.opt_batch, 3, self.img_h, self.img_w))
        out_shape = tuple(self.context.get_tensor_shape(self.output_name))
        self.out_len = out_shape[1]
        self.num_classes = out_shape[2]

        # Preallocate contiguous GPU buffer pool up to max_batch
        self.d_input = torch.empty(
            (self.max_batch, 3, self.img_h, self.img_w),
            dtype=self.torch_in_dtype,
            device=self.device
        )
        self.d_output = torch.empty(
            (self.max_batch, self.out_len, self.num_classes),
            dtype=self.torch_out_dtype,
            device=self.device
        )
        self.d_single_in = torch.empty(
            (1, 3, self.img_h, self.img_w),
            dtype=self.torch_in_dtype,
            device=self.device
        )
        self.d_single_out = torch.empty(
            (1, self.out_len, self.num_classes),
            dtype=self.torch_out_dtype,
            device=self.device
        )

        # Determine execution strategy
        if self.execution_mode == "pipelined":
            self.supports_multi_batch = False
            print("[Modo de Execução] Pipelining CUDA assíncrono em stream ativo (acurácia 100% preservada).")
        elif self.execution_mode == "native":
            self.supports_multi_batch = True
            print("[Modo de Execução] Despacho único direto nativo forçado.")
        else: # auto
            self.supports_multi_batch = self._probe_multi_batch_support()
            if self.supports_multi_batch:
                print("[Modo de Execução] Motor com suporte nativo a lote multi-amostra no decoder.")
            else:
                print("[Modo de Execução] Decodificador single-sequence detectado: usando pipelining CUDA assíncrono em stream (acurácia 100% preservada).")

    def _probe_multi_batch_support(self) -> bool:
        """Verifica se o motor suporta inferência multi-amostra nativa no decoder decodificando 2 amostras."""
        if self.max_batch < 2:
            return False
        try:
            probe_bs = 2
            probe_in = torch.randn(
                (probe_bs, 3, self.img_h, self.img_w),
                dtype=self.torch_in_dtype,
                device=self.device
            )
            probe_out = torch.empty(
                (probe_bs, self.out_len, self.num_classes),
                dtype=self.torch_out_dtype,
                device=self.device
            )

            self.context.set_input_shape(self.input_name, (probe_bs, 3, self.img_h, self.img_w))
            self.context.set_tensor_address(self.input_name, int(probe_in.data_ptr()))
            self.context.set_tensor_address(self.output_name, int(probe_out.data_ptr()))

            with torch.cuda.stream(self.stream):
                ok = self.context.execute_async_v3(self.stream.cuda_stream)
            self.stream.synchronize()

            if not ok:
                return False

            probs = probe_out.cpu().softmax(-1)
            preds, _ = self.tokenizer.decode(probs)
            pred0 = preds[0] if len(preds) > 0 else ""
            pred1 = preds[1] if len(preds) > 1 else ""
            if len(pred1) == 0 and len(pred0) > 0:
                return False

            diff = (probe_out[0] - probe_out[1]).abs().max().item()
            if diff < 1e-4:
                return False

            return True
        except Exception:
            return False

    def evaluate_batch(self, images: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Executa inferência em lote diretamente nos Tensor Cores da GPU."""
        bs = images.shape[0]

        # Asynchronously transfer batch to GPU with proper precision
        imgs_gpu = images.to(device=self.device, dtype=self.torch_in_dtype, non_blocking=True)

        if self.supports_multi_batch:
            if bs <= self.max_batch:
                # Single vectorized TensorRT kernel execution
                self.context.set_input_shape(self.input_name, (bs, 3, self.img_h, self.img_w))
                self.context.set_tensor_address(self.input_name, int(self.d_input.data_ptr()))
                self.context.set_tensor_address(self.output_name, int(self.d_output.data_ptr()))

                with torch.cuda.stream(self.stream):
                    self.d_input[:bs].copy_(imgs_gpu, non_blocking=True)
                    t0 = time.perf_counter()
                    self.context.execute_async_v3(self.stream.cuda_stream)
                    self.stream.synchronize()
                    t_infer = time.perf_counter() - t0

                logits = self.d_output[:bs].to(dtype=torch.float32, copy=True)
                return logits, t_infer

            # If dataloader batch_size exceeds engine's max_batch, chunk efficiently
            logits_chunks = []
            total_infer_time = 0.0

            with torch.cuda.stream(self.stream):
                for start in range(0, bs, self.max_batch):
                    end = min(start + self.max_batch, bs)
                    chunk_bs = end - start

                    self.context.set_input_shape(self.input_name, (chunk_bs, 3, self.img_h, self.img_w))
                    self.context.set_tensor_address(self.input_name, int(self.d_input.data_ptr()))
                    self.context.set_tensor_address(self.output_name, int(self.d_output.data_ptr()))

                    self.d_input[:chunk_bs].copy_(imgs_gpu[start:end], non_blocking=True)
                    t0 = time.perf_counter()
                    self.context.execute_async_v3(self.stream.cuda_stream)
                    self.stream.synchronize()
                    total_infer_time += (time.perf_counter() - t0)

                    logits_chunks.append(self.d_output[:chunk_bs].to(dtype=torch.float32, copy=True))

            logits = torch.cat(logits_chunks, dim=0)
            return logits, total_infer_time
        else:
            # High-throughput asynchronous pipelined execution
            self.context.set_input_shape(self.input_name, (1, 3, self.img_h, self.img_w))
            self.context.set_tensor_address(self.input_name, int(self.d_single_in.data_ptr()))
            self.context.set_tensor_address(self.output_name, int(self.d_single_out.data_ptr()))

            if self.d_output.shape[0] < bs:
                self.d_output = torch.empty(
                    (bs, self.out_len, self.num_classes),
                    dtype=self.torch_out_dtype,
                    device=self.device
                )

            with torch.cuda.stream(self.stream):
                t0 = time.perf_counter()
                for i in range(bs):
                    self.d_single_in.copy_(imgs_gpu[i:i+1], non_blocking=True)
                    self.context.execute_async_v3(self.stream.cuda_stream)
                    self.d_output[i:i+1].copy_(self.d_single_out, non_blocking=True)
                self.stream.synchronize()
                t_infer = time.perf_counter() - t0

            logits = self.d_output[:bs].to(dtype=torch.float32, copy=True)
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

    def close(self):
        """Libera com precisão todos os recursos CUDA e contextos de execução do TensorRT."""
        if hasattr(self, "d_input"):
            del self.d_input
        if hasattr(self, "d_output"):
            del self.d_output
        if hasattr(self, "d_single_in"):
            del self.d_single_in
        if hasattr(self, "d_single_out"):
            del self.d_single_out
        if hasattr(self, "context") and self.context is not None:
            del self.context
            self.context = None
        if hasattr(self, "engine") and self.engine is not None:
            del self.engine
            self.engine = None
        if hasattr(self, "stream") and self.stream is not None:
            del self.stream
            self.stream = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Avaliador de Alta Performance para Modelos PARSeq Exportados em TensorRT (.engine)")
    parser.add_argument('model', nargs='?', default=None, help="Caminho do arquivo .engine a ser testado")
    parser.add_argument('--model_path', '--engine', dest='model_opt', default=None, help="Caminho do arquivo .engine alternativo")
    parser.add_argument('--base_checkpoint', default='pretrained/parseq_alpr_98.5.ckpt',
                        help="Checkpoint base (.ckpt) para carregamento do tokenizer, charset e dimensões de entrada")
    parser.add_argument('--data_root', default='data', help="Diretório raiz dos dados LMDB")
    parser.add_argument('--batch_size', type=int, default=64, help="Tamanho do lote para inferência paralela em GPU")
    parser.add_argument('--num_workers', type=int, default=4, help="Número de workers no DataLoader")
    parser.add_argument('--cased', action='store_true', default=False, help="Comparação considerando maiúsculas e minúsculas")
    parser.add_argument('--punctuation', action='store_true', default=False, help="Verificar pontuação")
    parser.add_argument('--new', action='store_true', default=False, help="Avaliar nos novos datasets de benchmark")
    parser.add_argument('--rotation', type=int, default=0, help="Ângulo de rotação da imagem em graus (anti-horário)")
    parser.add_argument('--device', default='cuda', help="Dispositivo CUDA alvo (ex: 'cuda', 'cuda:0', 'cuda:1')")
    parser.add_argument('--profile_index', type=int, default=0, help="Índice do Optimization Profile compilado no engine")
    parser.add_argument('--datasets', '--dataset', nargs='+', default=None,
                        help="Datasets específicos para avaliação (ex: VeSV_pad RodoSol_pad UFPR_ALPR_pad)")
    parser.add_argument('--max_samples', type=int, default=None, help="Limite máximo de amostras avaliadas por dataset")
    parser.add_argument('--output', '--log_file', default=None, help="Arquivo customizado para salvar o relatório de resultados")
    parser.add_argument('--execution_mode', default='pipelined', choices=['pipelined', 'auto', 'native'],
                        help="Estratégia de execução na GPU: 'pipelined' (padrão, garante 100%% de acurácia em qualquer engine PARSeq via stream CUDA assíncrono), 'auto' (detecta via probe), ou 'native' (lote único direto)")

    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)

    chosen_model = args.model if args.model is not None else args.model_opt
    if chosen_model is None:
        parser.error("É necessário especificar o arquivo .engine como argumento posicional ou através de --model/--engine.")

    if not chosen_model.endswith(".engine"):
        print(f"[Aviso] O arquivo '{chosen_model}' não possui extensão .engine. Prosseguindo...")

    # Build charset_test
    charset_test = string.digits + string.ascii_lowercase
    if args.cased:
        charset_test += string.ascii_uppercase
    if args.punctuation:
        charset_test += string.punctuation
    kwargs.update({'charset_test': charset_test})

    # Load base system for tokenizer and dataset hparams
    if not os.path.exists(args.base_checkpoint):
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

    evaluator = None
    try:
        evaluator = FastTensorRTEvaluator(
            engine_path=chosen_model,
            base_system=base_sys,
            device=args.device,
            profile_index=args.profile_index,
            execution_mode=args.execution_mode
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
        print("RELATÓRIO DE AVALIAÇÃO TENSORRT:")
        print("=" * 92)
        print_results_table(result_list, file=sys.stdout)

        try:
            with open(log_path, 'w', encoding='utf-8') as f:
                print("Datasets:", file=f)
                print_results_table(result_list, file=f)
            print(f"\nResultados salvos com sucesso em: {log_path}")
        except Exception as e:
            print(f"\n[Aviso] Falha ao gravar log em '{log_path}': {e}")

    finally:
        if evaluator is not None:
            evaluator.close()


if __name__ == '__main__':
    main()
