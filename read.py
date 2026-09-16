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

from PIL import Image

import torch

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--images', nargs='+', help='Images to read')
    parser.add_argument('--device', default='cuda')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
    print(f'Additional keyword arguments: {kwargs}')

    if args.checkpoint.endswith('.onnx'):
        import onnxruntime as ort
        mll = kwargs.get('max_label_length', 25)
        ref_model = load_from_checkpoint('pretrained=parseq', max_label_length=mll, **kwargs).eval()
        img_transform = SceneTextDataModule.get_transform(ref_model.hparams.img_size)
        sess_opts = ort.SessionOptions()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'cuda' in args.device else ['CPUExecutionProvider']
        session = ort.InferenceSession(args.checkpoint, sess_opts, providers=providers)
        input_name = session.get_inputs()[0].name
        print(f"Running ONNX inference with provider: {session.get_providers()[0]}")

        for fname in args.images:
            image = Image.open(fname).convert('RGB')
            image_np = img_transform(image).unsqueeze(0).numpy()
            logits_np = session.run(None, {input_name: image_np})[0]
            p = torch.from_numpy(logits_np).softmax(-1)
            pred, p = ref_model.tokenizer.decode(p)
            print(f'{fname}: {pred[0]} (confidence: {p[0].prod().item():.4f})')
        return

    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)

    for fname in args.images:
        # Load image and prepare for input
        image = Image.open(fname).convert('RGB')
        image = img_transform(image).unsqueeze(0).to(args.device)

        p = model(image).softmax(-1)
        pred, p = model.tokenizer.decode(p)
        print(f'{fname}: {pred[0]}')


if __name__ == '__main__':
    main()
