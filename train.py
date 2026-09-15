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
import math
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.model_summary import summarize

from strhub.data.module import SceneTextDataModule
from strhub.models.base import BaseSystem
from strhub.models.utils import get_pretrained_weights


# Copied from OneCycleLR
def _annealing_cos(start, end, pct):
    'Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0.'
    cos_out = math.cos(math.pi * pct) + 1
    return end + (start - end) / 2.0 * cos_out


def get_swa_lr_factor(warmup_pct, swa_epoch_start, div_factor=25, final_div_factor=1e4) -> float:
    """Get the SWA LR factor for the given `swa_epoch_start`. Assumes OneCycleLR Scheduler."""
    total_steps = 1000  # Can be anything. We use 1000 for convenience.
    start_step = int(total_steps * warmup_pct) - 1
    end_step = total_steps - 1
    step_num = int(total_steps * swa_epoch_start) - 1
    pct = (step_num - start_step) / (end_step - start_step)
    return _annealing_cos(1, 1 / (div_factor * final_div_factor), pct)


@hydra.main(config_path='configs', config_name='main', version_base='1.2')
def main(config: DictConfig):
    trainer_strategy = 'auto'
    with open_dict(config):
        # Resolve absolute path to data.root_dir
        config.data.root_dir = hydra.utils.to_absolute_path(config.data.root_dir)
        # Special handling for GPU-affected config
        gpu = config.trainer.get('accelerator') == 'gpu' and torch.cuda.is_available()
        if config.trainer.get('accelerator') == 'gpu' and not torch.cuda.is_available():
            print("[Warning] GPU requested in trainer.accelerator, but CUDA is not available. Falling back to CPU.")
            config.trainer.accelerator = 'cpu'
            config.trainer.devices = 1
        devices = config.trainer.get('devices', 0)
        if gpu:
            # Use mixed-precision training
            config.trainer.precision = 'bf16-mixed' if torch.get_autocast_gpu_dtype() is torch.bfloat16 else '16-mixed'
        if gpu and devices > 1:
            # Use DDP with optimizations
            trainer_strategy = DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True)
            # Scale steps-based config
            config.trainer.val_check_interval //= devices
            max_steps = config.trainer.get('max_steps', -1)
            if max_steps is None:
                config.trainer.max_steps = -1
            elif config.trainer.max_steps > 0:
                config.trainer.max_steps //= devices

    # Special handling for PARseq
    if config.model.get('perm_mirrored', False):
        assert config.model.perm_num % 2 == 0, 'perm_num should be even if perm_mirrored = True'

    model: BaseSystem = hydra.utils.instantiate(config.model)
    # If specified, use pretrained weights to initialize the model
    if config.pretrained is not None:
        m = model.model if config.model._target_.endswith('PARSeq') else model
        weights = get_pretrained_weights(config.pretrained)
        model_state = m.state_dict()
        filtered_weights = {}
        for k, v in weights.items():
            if k in model_state:
                if v.shape == model_state[k].shape:
                    filtered_weights[k] = v
                else:
                    # Pos queries slicing along sequence length dimension
                    if k == 'pos_queries' and v.dim() == model_state[k].dim() and v.shape[0] == model_state[k].shape[0]:
                        min_len = min(model_state[k].shape[1], v.shape[1])
                        sliced = model_state[k].clone()
                        sliced[:, :min_len] = v[:, :min_len]
                        filtered_weights[k] = sliced
                        print(f"[Pretrained] Sliced '{k}' from {list(v.shape)} to {list(sliced.shape)}")
                    else:
                        print(f"[Pretrained] Skipping mismatched key '{k}' (checkpoint: {list(v.shape)}, model: {list(model_state[k].shape)}) - preserving downstream initialization.")
            else:
                filtered_weights[k] = v
        m.load_state_dict(filtered_weights, strict=False)

    # Quantization preparation for fine-tuning (QAT or Jetfire FQT)
    quant_method = config.get('quantize', None) or config.get('quant_method', None)
    if quant_method:
        from strhub.models.quantization import PARSeqQuantizer
        block_size = config.get('quant_block_size', 64)
        print(f"[Quantization] Preparing model for '{quant_method.upper()}' fine-tuning (block_size={block_size})...")
        model = PARSeqQuantizer.quantize(model, method=quant_method, block_size=block_size)

    print(summarize(model, max_depth=2))

    datamodule: SceneTextDataModule = hydra.utils.instantiate(config.data)

    checkpoint = ModelCheckpoint(
        monitor='val_accuracy',
        mode='max',
        save_top_k=3,
        save_last=True,
        filename='{epoch}-{step}-{val_accuracy:.4f}-{val_NED:.4f}',
    )
    swa_epoch_start = 0.75
    swa_lr = config.model.lr * get_swa_lr_factor(config.model.warmup_pct, swa_epoch_start)
    swa = StochasticWeightAveraging(swa_lr, swa_epoch_start)
    cwd = (
        HydraConfig.get().runtime.output_dir
        if config.ckpt_path is None
        else str(Path(config.ckpt_path).parents[1].absolute())
    )
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        logger=TensorBoardLogger(cwd, '', '.'),
        strategy=trainer_strategy,
        enable_model_summary=False,
        callbacks=[checkpoint, swa],
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=config.ckpt_path)


if __name__ == '__main__':
    main()
