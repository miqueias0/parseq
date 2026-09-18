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

import os
import argparse
import string
import sys
from dataclasses import dataclass
from typing import Optional

# Ensure UTF-8 output encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from tqdm import tqdm

import torch

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
    if not results:
        print("Nenhum resultado para exibir.", file=file)
        return
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


class ONNXEvaluator:
    """Wrapper that runs ONNX inference and integrates directly with test.py evaluation loop."""
    def __init__(self, onnx_path: str, base_system, device: str = "cuda"):
        import onnxruntime as ort
        self.tokenizer = base_system.tokenizer
        self.charset_adapter = base_system.charset_adapter
        self.hparams = base_system.hparams
        self.device = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "cuda" in str(self.device) and "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
        print(f"Loading ONNX Model: {onnx_path} using {providers[0]}")
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def parameters(self):
        yield torch.empty(0, dtype=torch.float32)

    def test_step(self, batch, batch_idx):
        from strhub.models.base import BatchResult
        from nltk import edit_distance
        images, labels = batch
        bs = images.shape[0]

        logits_list = []
        for i in range(bs):
            img_np = images[i:i+1].cpu().numpy()
            out_np = self.session.run([self.output_name], {self.input_name: img_np})[0]
            logits_list.append(torch.from_numpy(out_np))
        logits = torch.cat(logits_list, dim=0)

        probs = logits.softmax(-1)
        preds, prob_tuples = self.tokenizer.decode(probs)

        total = correct = ned = confidence = label_length = 0
        for pred, prob, gt in zip(preds, prob_tuples, labels):
            confidence += prob.prod().item()
            pred = self.charset_adapter(pred)
            ned += edit_distance(pred, gt) / max(len(pred), len(gt), 1)
            if pred == gt:
                correct += 1
            total += 1
            label_length += len(pred)
        return dict(output=BatchResult(total, correct, ned, confidence, label_length, None, None))


class TensorRTEvaluator:
    """Wrapper that runs TensorRT engine inference with dedicated CUDA stream."""
    def __init__(self, engine_path: str, base_system, device: str = "cuda"):
        import tensorrt as trt
        self.tokenizer = base_system.tokenizer
        self.charset_adapter = base_system.charset_adapter
        self.hparams = base_system.hparams
        self.device = torch.device("cuda")

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        print(f"Loading TensorRT Engine: {engine_path}")
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

        in_dtype = self.engine.get_tensor_dtype("images")
        self.torch_in_dtype = torch.float16 if in_dtype == trt.DataType.HALF else torch.float32
        out_dtype = self.engine.get_tensor_dtype("logits")
        self.torch_out_dtype = torch.float16 if out_dtype == trt.DataType.HALF else torch.float32

        # Preallocate static slice buffers to avoid allocating torch.empty on each sample
        self.d_single_in = torch.empty((1, 3, 32, 128), dtype=self.torch_in_dtype, device="cuda")
        self.context.set_input_shape("images", (1, 3, 32, 128))
        out_shape = tuple(self.context.get_tensor_shape("logits"))
        self.d_single_out = torch.empty(out_shape, dtype=self.torch_out_dtype, device="cuda")
        self.out_len = out_shape[1]
        self.num_classes = out_shape[2]
        self.context.set_tensor_address("images", int(self.d_single_in.data_ptr()))
        self.context.set_tensor_address("logits", int(self.d_single_out.data_ptr()))

    def parameters(self):
        yield torch.empty(0, dtype=torch.float32)

    def test_step(self, batch, batch_idx):
        from strhub.models.base import BatchResult
        from nltk import edit_distance

        images, labels = batch
        bs = images.shape[0]

        # Transfer full batch to GPU in a single DMA burst
        d_batch_imgs = images.to(device="cuda", dtype=self.torch_in_dtype, non_blocking=True)
        batch_logits = torch.empty((bs, self.out_len, self.num_classes), dtype=torch.float32, device="cuda")

        # Pipelined execution without CPU blocking per sample
        with torch.cuda.stream(self.stream):
            for i in range(bs):
                self.d_single_in.copy_(d_batch_imgs[i:i+1], non_blocking=True)
                self.context.execute_async_v3(self.stream.cuda_stream)
                batch_logits[i:i+1].copy_(self.d_single_out, non_blocking=True)

        # Single stream sync per batch
        self.stream.synchronize()

        probs = batch_logits.softmax(-1)
        preds, prob_tuples = self.tokenizer.decode(probs)

        total = correct = ned = confidence = label_length = 0
        for pred, prob, gt in zip(preds, prob_tuples, labels):
            confidence += prob.prod().item()
            pred = self.charset_adapter(pred)
            ned += edit_distance(pred, gt) / max(len(pred), len(gt), 1)
            if pred == gt:
                correct += 1
            total += 1
            label_length += len(pred)
        return dict(output=BatchResult(total, correct, ned, confidence, label_length, None, None))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (.ckpt, .pt, .pth) or (.onnx) or (.engine)")
    parser.add_argument('--base_checkpoint', default='pretrained/parseq_alpr_98.5.ckpt',
                        help="Base checkpoint providing tokenizer and hparams when testing .onnx or .engine")
    parser.add_argument('--checks', default=None,
                        help="Base checkpoint providing tokenizer and hparams when testing .onnx or .engine")
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--cased', action='store_true', default=False, help='Cased comparison')
    parser.add_argument('--punctuation', action='store_true', default=False, help='Check punctuation')
    parser.add_argument('--new', action='store_true', default=False, help='Evaluate on new benchmark datasets')
    parser.add_argument('--rotation', type=int, default=0, help='Angle of rotation (counter clockwise) in degrees.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--variant', default=None, choices=['m0', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6'],
                        help="Quantization / Architecture variant: m0 (FP32 AR), m1 (FP32 NAR), m2 (FP16 NAR), m3 (INT8 Naive), m4 (INT8 W8A8), m5 (INT8 Integer-Only PTQ I-BERT/IPTQ-ViT), m6 (INT8 Integer-Only QAT)")
    parser.add_argument('--use_int_flashattention', action='store_true', default=False,
                        help="Habilita INT-FlashAttention (arXiv:2409.16997v2) com GEMMs INT8 e online softmax fundido")
    parser.add_argument('--use_sage_attention', action='store_true', default=False,
                        help="Habilita SageAttention (arXiv:2410.02367v9 - ICLR 2025) com smoothing de K e GEMMs INT8")
    parser.add_argument('--sage_mode', type=str, default="sageattn_b", choices=["sageattn_b", "sageattn_vb"],
                        help="SageAttention mode: sageattn_b (float V) ou sageattn_vb (fully INT8)")
    parser.add_argument('--datasets', '--dataset', nargs='+', default=None,
                        help="Datasets to evaluate (e.g. VeSV_pad RodoSol_pad UFPR_ALPR_pad)")
    parser.add_argument('--max_samples', type=int, default=None, help="Optional maximum number of samples per dataset")
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)

    charset_test = string.digits + string.ascii_lowercase
    if args.cased:
        charset_test += string.ascii_uppercase
    if args.punctuation:
        charset_test += string.punctuation
    kwargs.update({'charset_test': charset_test})
    print(f'Additional keyword arguments: {kwargs}')

    ckpt_lower = args.checkpoint.lower()

    if ckpt_lower.endswith(".onnx"):
        base_sys = load_from_checkpoint(args.base_checkpoint, **kwargs).eval()
        model = ONNXEvaluator(args.checkpoint, base_sys, device=args.device)
    elif ckpt_lower.endswith(".engine"):
        base_sys = load_from_checkpoint(args.base_checkpoint, **kwargs).eval()
        model = TensorRTEvaluator(args.checkpoint, base_sys, device=args.device)
    else:
        # Checkpoint loading
        if "qat" in ckpt_lower or "m6" in ckpt_lower:
            # Handle QAT custom checkpoint
            base_sys = load_from_checkpoint(args.base_checkpoint, **kwargs).eval().to(args.device)
            from strhub.models.parseq.quantized_parseq import create_model_variant
            base_sys.model = create_model_variant(
                "m6", base_sys.model,
                use_int_flashattention=args.use_int_flashattention,
                use_sage_attention=args.use_sage_attention,
                sage_mode=args.sage_mode,
            ).eval().to(args.device)
            if os.path.exists(args.checkpoint):
                ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
                sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
                clean_sd = {k.replace("model.", ""): v for k, v in sd.items()}
                base_sys.model.load_state_dict(clean_sd, strict=False)
                print(f"Loaded QAT weights from {args.checkpoint} into M6.")
            model = base_sys
        else:
            model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)

            # Wrap model with requested i-parseq / quantization variant
            if args.variant:
                from strhub.models.parseq.quantized_parseq import create_model_variant
                print(f"Applying quantization/architecture variant: {args.variant.upper()}")
                model.model = create_model_variant(
                    args.variant, model.model,
                    use_int_flashattention=args.use_int_flashattention,
                    use_sage_attention=args.use_sage_attention,
                    sage_mode=args.sage_mode,
                ).eval().to(args.device)
                if args.variant == "m6" and os.path.exists("pretrained/parseq_alpr_qat_m6.ckpt"):
                    ckpt = torch.load("pretrained/parseq_alpr_qat_m6.ckpt", map_location=args.device, weights_only=False)
                    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
                    clean_sd = {k.replace("model.", ""): v for k, v in sd.items()}
                    model.model.load_state_dict(clean_sd, strict=False)
                    print("Loaded trained QAT checkpoint into variant M6.")

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

    if args.datasets:
        test_set = sorted(set(args.datasets))
    else:
        raw_set = list(SceneTextDataModule.TEST_BENCHMARK_SUB + SceneTextDataModule.TEST_BENCHMARK)
        if args.new:
            raw_set += list(SceneTextDataModule.TEST_NEW)
        test_set = sorted([s for s in set(raw_set) if os.path.exists(os.path.join(args.data_root, 'test', s))])
        if not test_set:
            test_set = ['VeSV_pad']

    results = {}
    max_width = max(map(len, test_set))
    model_dtype = next(model.parameters()).dtype if list(model.parameters()) else torch.float32

    for name, dataloader in datamodule.test_dataloaders(test_set).items():
        total = 0
        correct = 0
        ned = 0
        confidence = 0
        label_length = 0
        for imgs, labels in tqdm(iter(dataloader), desc=f'{name:>{max_width}}'):
            if args.max_samples and total >= args.max_samples:
                break
            imgs_in = imgs.to(device=model.device, dtype=model_dtype)
            res = model.test_step((imgs_in, labels), -1)['output']
            total += res.num_samples
            correct += res.correct
            ned += res.ned
            confidence += res.confidence
            label_length += res.label_length
        accuracy = 100 * correct / total if total > 0 else 0.0
        mean_ned = 100 * (1 - ned / total) if total > 0 else 0.0
        mean_conf = 100 * confidence / total if total > 0 else 0.0
        mean_label_length = label_length / total if total > 0 else 0.0
        results[name] = Result(name, total, accuracy, mean_ned, mean_conf, mean_label_length)

    result_groups = {
        'Datasets': test_set,
    }
    log_path = args.checkpoint + '.log.txt'
    try:
        with open(log_path, 'w', encoding='utf-8') as f:
            for out in [f, sys.stdout]:
                for group, subset in result_groups.items():
                    print(f'{group}:', file=out)
                    print_results_table([results[s] for s in subset if s in results], out)
                    print('\n', file=out)
    except Exception as e:
        for group, subset in result_groups.items():
            print(f'{group}:')
            print_results_table([results[s] for s in subset if s in results])
            print('\n')


if __name__ == '__main__':
    main()
