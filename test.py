#!/usr/bin/env python3
# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import string
import sys
from dataclasses import dataclass

from tqdm import tqdm

import torch


def setup_tensorrt_library_paths():
    """Auto-detects and loads TensorRT, cuDNN, and cuBLAS shared libraries from pip packages."""
    import ctypes

    lib_dirs = []
    for candidate in [
        "/home/mon25/modelos/.venv/lib/python3.12/site-packages",
        os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages"),
    ]:
        if os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.append(candidate)

    for base in list(sys.path):
        if not os.path.isdir(base):
            continue
        trt_dir = os.path.join(base, "tensorrt_libs")
        if os.path.isdir(trt_dir) and trt_dir not in lib_dirs:
            lib_dirs.append(trt_dir)
        nvidia_dir = os.path.join(base, "nvidia")
        if os.path.isdir(nvidia_dir):
            for sub in ["cuda_runtime", "cublas", "cudnn", "curand", "cufft", "cusolver", "cusparse"]:
                sub_lib = os.path.join(nvidia_dir, sub, "lib")
                if os.path.isdir(sub_lib) and sub_lib not in lib_dirs:
                    lib_dirs.append(sub_lib)

    if lib_dirs:
        old_ld = os.environ.get("LD_LIBRARY_PATH", "")
        new_ld = ":".join(lib_dirs) + (f":{old_ld}" if old_ld else "")
        os.environ["LD_LIBRARY_PATH"] = new_ld

        libs_to_load = [
            "libcudart.so.12",
            "libcublasLt.so.12",
            "libcublas.so.12",
            "libcudnn.so.9",
            "libcurand.so.10",
            "libcufft.so.11",
            "libnvinfer.so.10",
            "libnvinfer_plugin.so.10",
            "libnvonnxparser.so.10",
        ]
        for lib_name in libs_to_load:
            for lib_dir in lib_dirs:
                lib_path = os.path.join(lib_dir, lib_name)
                if os.path.exists(lib_path):
                    try:
                        ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                        break
                    except Exception:
                        pass


setup_tensorrt_library_paths()

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


def print_results_table(results: list[Result], file=None):
    w = max(map(len, map(getattr, results, ['dataset'] * len(results))))
    w = max(w, len('Dataset'), len('Combined'))
    print('| {:<{w}} | # samples | Accuracy | 1 - NED | Confidence | Label Length |'.format('Dataset', w=w), file=file)
    print('|:{:-<{w}}:|----------:|---------:|--------:|-----------:|-------------:|'.format('----', w=w), file=file)
    c = Result('Combined', 0, 0, 0, 0, 0)
    for res in results:
        c.num_samples += res.num_samples
        c.accuracy += res.num_samples * res.accuracy
        c.ned += res.num_samples * res.ned
        c.confidence += res.num_samples * res.confidence
        c.label_length += res.num_samples * res.label_length
        print(
            f'| {res.dataset:<{w}} | {res.num_samples:>9} | {res.accuracy:>8.2f} | {res.ned:>7.2f} '
            f'| {res.confidence:>10.2f} | {res.label_length:>12.2f} |',
            file=file,
        )
    c.accuracy /= c.num_samples
    c.ned /= c.num_samples
    c.confidence /= c.num_samples
    c.label_length /= c.num_samples
    print('|-{:-<{w}}-|-----------|----------|---------|------------|--------------|'.format('----', w=w), file=file)
    print(
        f'| {c.dataset:<{w}} | {c.num_samples:>9} | {c.accuracy:>8.2f} | {c.ned:>7.2f} '
        f'| {c.confidence:>10.2f} | {c.label_length:>12.2f} |',
        file=file,
    )


class ONNXModelWrapper:
    """Wrapper to evaluate ONNX models directly through PyTorch SceneTextDataModule."""

    def __init__(self, onnx_path: str, ref_checkpoint: str = 'pretrained=parseq', device: str = 'cuda', **kwargs):
        import onnxruntime as ort
        from nltk import edit_distance
        from strhub.models.base import BatchResult

        self.edit_distance = edit_distance
        mll = kwargs.get('max_label_length', 25)
        self.ref_model = load_from_checkpoint(ref_checkpoint, max_label_length=mll, **kwargs).eval()
        self.hparams = self.ref_model.hparams
        provider_choice = kwargs.pop('provider', 'auto')
        # Ensure all shapes are inferred for TensorRT Execution Provider
        if provider_choice in ["tensorrt", "auto"] and "cuda" in device:
            try:
                import onnx
                from onnx import shape_inference
                m = onnx.load(str(onnx_path))
                if len(m.graph.value_info) < len(m.graph.node):
                    m_inf = shape_inference.infer_shapes(m, check_type=True)
                    onnx.save(m_inf, str(onnx_path))
            except Exception:
                pass

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 4

        available = ort.get_available_providers()
        trt_cache_dir = os.path.join(os.path.dirname(str(onnx_path)) or ".", "trt_cache")
        os.makedirs(trt_cache_dir, exist_ok=True)
        img_size = getattr(self.hparams, 'img_size', (32, 128))
        eval_batch = kwargs.pop('batch_size', 512)
        max_batch = max(eval_batch, 512)
        trt_options = {
            "trt_fp16_enable": True,
            "trt_int8_enable": True,
            "trt_max_workspace_size": 2147483648,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": trt_cache_dir,
            "trt_profile_min_shapes": f"images:1x3x{img_size[0]}x{img_size[1]}",
            "trt_profile_max_shapes": f"images:{max_batch}x3x{img_size[0]}x{img_size[1]}",
            "trt_profile_opt_shapes": f"images:{eval_batch}x3x{img_size[0]}x{img_size[1]}",
        }
        if provider_choice == "tensorrt" or (provider_choice == "auto" and "TensorrtExecutionProvider" in available and "cuda" in device):
            providers = [("TensorrtExecutionProvider", trt_options), "CUDAExecutionProvider", "CPUExecutionProvider"]
        elif "cuda" in device and provider_choice in ["cuda", "auto"]:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(str(onnx_path), sess_opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def test_step(self, batch, batch_idx):
        images, labels = batch
        img_np = images.cpu().numpy()
        ort_outs = self.session.run(None, {self.input_name: img_np})
        logits_np = ort_outs[0]
        logits = torch.from_numpy(logits_np)
        probs = logits.softmax(-1)
        preds, probs = self.ref_model.tokenizer.decode(probs)

        correct = 0
        total = 0
        ned = 0
        confidence = 0
        label_length = 0
        for pred, prob, gt in zip(preds, probs, labels):
            confidence += prob.prod().item()
            pred = self.ref_model.charset_adapter(pred)
            ned += self.edit_distance(pred, gt) / max(len(pred), len(gt))
            if pred == gt:
                correct += 1
            total += 1
            label_length += len(pred)
        return dict(output=self.BatchResult(total, correct, ned, confidence, label_length, None, None))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>' or path to '.onnx' file)")
    parser.add_argument('--ref_checkpoint', default='pretrained=parseq', help="Reference checkpoint for tokenizer when evaluating .onnx")
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cased', action='store_true', default=False, help='Cased comparison')
    parser.add_argument('--punctuation', action='store_true', default=False, help='Check punctuation')
    parser.add_argument('--rotation', type=int, default=0, help='Angle of rotation (counter clockwise) in degrees.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dataset', default=None, help='Specific dataset under data_root/test/ to evaluate (e.g. VeSV_pad)')
    parser.add_argument('--max_label_length', type=int, default=None, help='Override max_label_length')
    parser.add_argument('--new', action='store_true', default=False, help='Evaluate on new benchmark datasets')
    parser.add_argument(
        '--quant_method',
        choices=['none', 'real_int8', 'smoothquant_int8', 'unified_int8', 'int_flashattn', 'ibert', 'jetfire_fqt', 'dynamic', 'qat'],
        default='none',
        help='INT8 Quantization method to evaluate',
    )
    parser.add_argument('--provider', default='auto', choices=['auto', 'tensorrt', 'cuda', 'cpu'], help='ONNX Runtime provider')
    parser.add_argument('--block_size', type=int, default=32, help='Block size for Jetfire / INT-FlashAttention (default: 32)')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
    if args.max_label_length is not None:
        kwargs['max_label_length'] = args.max_label_length

    charset_test = string.digits + string.ascii_lowercase
    if args.cased:
        charset_test += string.ascii_uppercase
    if args.punctuation:
        charset_test += string.punctuation
    kwargs.update({'charset_test': charset_test})
    print(f'Additional keyword arguments: {kwargs}')

    if args.checkpoint.endswith('.onnx'):
        ref_ckpt = getattr(args, 'ref_checkpoint', 'pretrained=parseq')
        model = ONNXModelWrapper(
            args.checkpoint,
            ref_checkpoint=ref_ckpt,
            device=args.device,
            provider=args.provider,
            batch_size=args.batch_size,
            **kwargs,
        )
        active_p = model.session.get_providers()[0]
        print(f"Loaded ONNX Model: {args.checkpoint} (ExecutionProvider: {active_p})")
    else:
        model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
        if args.quant_method != 'none':
            from strhub.models.quantization import PARSeqQuantizer
            print(f'Applying quantization method: {args.quant_method} (block_size={args.block_size})...')
            model = PARSeqQuantizer.quantize(model, method=args.quant_method, block_size=args.block_size, inplace=True)
            if args.quant_method != 'dynamic':
                model = model.to(args.device)
    hp = model.hparams
    datamodule = SceneTextDataModule(
        args.data_root,
        '_unused_',
        hp.img_size,
        hp.max_label_length,
        hp.charset_train,
        hp.charset_test,
        args.batch_size,
        args.num_workers,
        False,
        rotation=args.rotation,
    )

    if args.dataset is not None:
        test_set = [args.dataset]
    else:
        from pathlib import Path
        test_dir = Path(args.data_root) / 'test'
        if test_dir.exists():
            existing_subdirs = [d.name for d in test_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
            std_set = SceneTextDataModule.TEST_BENCHMARK_SUB + SceneTextDataModule.TEST_BENCHMARK
            if getattr(args, 'new', False):
                std_set += SceneTextDataModule.TEST_NEW
            common = [s for s in std_set if s in existing_subdirs]
            if len(common) > 0:
                test_set = sorted(set(common))
            elif len(existing_subdirs) > 0:
                test_set = sorted(existing_subdirs)
            else:
                test_set = sorted(set(std_set))
        else:
            test_set = SceneTextDataModule.TEST_BENCHMARK_SUB + SceneTextDataModule.TEST_BENCHMARK
            if getattr(args, 'new', False):
                test_set += SceneTextDataModule.TEST_NEW
            test_set = sorted(set(test_set))

    results = {}
    max_width = max(map(len, test_set)) if len(test_set) > 0 else 10
    eval_device = torch.device(args.device)
    for name, dataloader in datamodule.test_dataloaders(test_set).items():
        total = 0
        correct = 0
        ned = 0
        confidence = 0
        label_length = 0
        for imgs, labels in tqdm(iter(dataloader), desc=f'{name:>{max_width}}'):
            res = model.test_step((imgs.to(eval_device), labels), -1)['output']
            total += res.num_samples
            correct += res.correct
            ned += res.ned
            confidence += res.confidence
            label_length += res.label_length
        accuracy = 100 * correct / total
        mean_ned = 100 * (1 - ned / total)
        mean_conf = 100 * confidence / total
        mean_label_length = label_length / total
        results[name] = Result(name, total, accuracy, mean_ned, mean_conf, mean_label_length)

    if args.dataset is not None:
        result_groups = {'Custom': [args.dataset]}
    else:
        result_groups = {}
        sub_bench = [s for s in SceneTextDataModule.TEST_BENCHMARK_SUB if s in results]
        if sub_bench:
            result_groups['Benchmark (Subset)'] = sub_bench
        full_bench = [s for s in SceneTextDataModule.TEST_BENCHMARK if s in results]
        if full_bench:
            result_groups['Benchmark'] = full_bench
        if getattr(args, 'new', False):
            new_bench = [s for s in SceneTextDataModule.TEST_NEW if s in results]
            if new_bench:
                result_groups['New'] = new_bench
        remaining = [s for s in results if not any(s in g for g in result_groups.values())]
        if remaining:
            result_groups['Evaluation'] = remaining

    out_log_path = (args.checkpoint if not args.checkpoint.startswith('pretrained=') else 'parseq_eval') + '.log.txt'
    with open(out_log_path, 'w') as f:
        for out in [f, sys.stdout]:
            for group, subset in result_groups.items():
                entries = [results[s] for s in subset if s in results]
                if entries:
                    print(f'{group} set:', file=out)
                    print_results_table(entries, out)
                    print('\n', file=out)


if __name__ == '__main__':
    main()
