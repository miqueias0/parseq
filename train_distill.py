#!/usr/bin/env python3
# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
# Distillation Training Runner for Custom PARSeq Models

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
from train import get_swa_lr_factor


@hydra.main(config_path='configs', config_name='main', version_base='1.2')
def main(config: DictConfig):
    trainer_strategy = 'auto'
    with open_dict(config):
        # Resolve absolute path to data.root_dir
        config.data.root_dir = hydra.utils.to_absolute_path(config.data.root_dir)

        # Configurações de Destilação
        # Redireciona o target do modelo para o sistema de destilação
        config.model._target_ = 'strhub.models.parseq.distill_system.PARSeqDistill'

        # Repassa opções de destilação da linha de comando para o modelo
        if 'teacher_ckpt' in config:
            config.model.teacher_ckpt = config.teacher_ckpt
        elif 'teacher_ckpt' not in config.model:
            config.model.teacher_ckpt = None

        if 'alpha' in config:
            config.model.alpha = config.alpha
        elif 'alpha' not in config.model:
            config.model.alpha = 0.5

        if 'temperature' in config:
            config.model.temperature = config.temperature
        elif 'temperature' not in config.model:
            config.model.temperature = 2.0

        if 'use_teacher_conf' in config:
            config.model.use_teacher_conf = config.use_teacher_conf
        elif 'use_teacher_conf' not in config.model:
            config.model.use_teacher_conf = True

        if 'distill_mode' in config:
            config.model.distill_mode = config.distill_mode
        elif 'distill_mode' not in config.model:
            config.model.distill_mode = 'perms'

        # Atualiza o nome do modelo para identificar saídas de destilação
        model_name = config.model.get('name', 'parseq')
        if not model_name.endswith('-distill'):
            config.model.name = f"{model_name}-distill"

        # Special handling for GPU-affected config
        gpu = config.trainer.get('accelerator') == 'gpu'
        devices = config.trainer.get('devices', 0)
        if gpu:
            # Use mixed-precision training
            config.trainer.precision = 'bf16-mixed' if torch.get_autocast_gpu_dtype() is torch.bfloat16 else '16-mixed'
        if gpu and devices > 1:
            # Use DDP with optimizations
            trainer_strategy = DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True)
            # Scale steps-based config
            config.trainer.val_check_interval //= devices
            if config.trainer.get('max_steps', -1) > 0:
                config.trainer.max_steps //= devices

    # Special handling for PARseq
    if config.model.get('perm_mirrored', False):
        assert config.model.perm_num % 2 == 0, 'perm_num should be even if perm_mirrored = True'

    print(f"\n========================================================")
    print(f" Iniciando Treino com Destilação:")
    print(f" Aluno: {config.model.name}")
    print(f" Professor: configs/model/parseq.yaml (ckpt: {config.model.teacher_ckpt})")
    print(f" Alpha: {config.model.alpha} | Temp: {config.model.temperature} | Conf: {config.model.use_teacher_conf}")
    print(f"========================================================\n")

    model: BaseSystem = hydra.utils.instantiate(config.model)

    # Se pretrained especificado para o aluno
    if config.pretrained is not None:
        model.model.load_state_dict(get_pretrained_weights(config.pretrained))

    print(model)

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
