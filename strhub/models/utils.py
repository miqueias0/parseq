from pathlib import PurePath
from typing import Sequence

import yaml

import torch
from torch import nn


class InvalidModelError(RuntimeError):
    """Exception raised for any model-related error (creation, loading)"""


_WEIGHTS_URL = {
    'parseq-tiny': 'https://github.com/baudm/parseq/releases/download/v1.0.0/parseq_tiny-e7a21b54.pt',
    'parseq-patch16-224': 'https://github.com/baudm/parseq/releases/download/v1.0.0/parseq_small_patch16_224-fcf06f5a.pt',
    'parseq': 'https://github.com/baudm/parseq/releases/download/v1.0.0/parseq-bb5792a6.pt',
    'abinet': 'https://github.com/baudm/parseq/releases/download/v1.0.0/abinet-1d1e373e.pt',
    'trba': 'https://github.com/baudm/parseq/releases/download/v1.0.0/trba-cfaed284.pt',
    'vitstr': 'https://github.com/baudm/parseq/releases/download/v1.0.0/vitstr-26d0fcf4.pt',
    'crnn': 'https://github.com/baudm/parseq/releases/download/v1.0.0/crnn-679d0e31.pt',
}


def _get_config(experiment: str, **kwargs):
    """Emulates hydra config resolution"""
    root = PurePath(__file__).parents[2]
    with open(root / 'configs/main.yaml', 'r') as f:
        config = yaml.load(f, yaml.Loader)['model']
    with open(root / 'configs/charset/94_full.yaml', 'r') as f:
        config.update(yaml.load(f, yaml.Loader)['model'])
    with open(root / f'configs/experiment/{experiment}.yaml', 'r') as f:
        exp = yaml.load(f, yaml.Loader)
    # Apply base model config
    model = exp['defaults'][0]['override /model']
    with open(root / f'configs/model/{model}.yaml', 'r') as f:
        config.update(yaml.load(f, yaml.Loader))
    # Apply experiment config
    if 'model' in exp:
        config.update(exp['model'])
    config.update(kwargs)
    # Workaround for now: manually cast the lr to the correct type.
    config['lr'] = float(config['lr'])
    return config


def _get_model_class(key):
    if 'abinet' in key:
        from .abinet.system import ABINet as ModelClass
    elif 'crnn' in key:
        from .crnn.system import CRNN as ModelClass
    elif 'parseq' in key:
        from .parseq.system import PARSeq as ModelClass
    elif 'trba' in key:
        from .trba.system import TRBA as ModelClass
    elif 'trbc' in key:
        from .trba.system import TRBC as ModelClass
    elif 'vitstr' in key:
        from .vitstr.system import ViTSTR as ModelClass
    else:
        raise InvalidModelError(f"Unable to find model class for '{key}'")
    return ModelClass


def get_pretrained_weights(experiment):
    try:
        url = _WEIGHTS_URL[experiment]
    except KeyError:
        raise InvalidModelError(f"No pretrained weights found for '{experiment}'") from None
    return torch.hub.load_state_dict_from_url(url=url, map_location='cpu', check_hash=True)


def create_model(experiment: str, pretrained: bool = False, **kwargs):
    try:
        config = _get_config(experiment, **kwargs)
    except FileNotFoundError:
        raise InvalidModelError(f"No configuration found for '{experiment}'") from None
    ModelClass = _get_model_class(experiment)
    model = ModelClass(**config)
    if pretrained:
        m = model.model if 'parseq' in experiment else model
        m.load_state_dict(get_pretrained_weights(experiment))
    return model


import os


def load_from_checkpoint(checkpoint_path: str, **kwargs):
    if checkpoint_path.startswith('pretrained='):
        model_id = checkpoint_path.split('=', maxsplit=1)[1]
        model = create_model(model_id, True, **kwargs)
        return model

    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False

    ModelClass = _get_model_class(checkpoint_path)

    # Check if checkpoint is a custom dictionary (e.g. QAT checkpoint saved by torch.save)
    if os.path.exists(checkpoint_path):
        try:
            raw = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except Exception:
            raw = None

        if isinstance(raw, dict) and 'pytorch-lightning_version' not in raw:
            base_candidates = [
                'pretrained/parseq_alpr_98.5.ckpt',
                os.path.join(os.path.dirname(checkpoint_path), 'parseq_alpr_98.5.ckpt'),
            ]
            base_path = next((p for p in base_candidates if os.path.exists(p) and p != checkpoint_path), None)
            if base_path:
                model = ModelClass.load_from_checkpoint(base_path, **kwargs)
            else:
                model = create_model('parseq', pretrained=False)

            sd = raw.get('model_state_dict', raw.get('state_dict', raw))
            is_qat = any('q_weight' in k or 'weight_scale' in k for k in sd.keys()) or 'qat' in checkpoint_path.lower()
            if is_qat:
                from strhub.models.parseq.quantized_parseq import create_model_variant
                model.model = create_model_variant('m6', model.model)

            clean_sd = {k.replace('model.', ''): v for k, v in sd.items()}
            target = model.model if hasattr(model, 'model') else model
            target.load_state_dict(clean_sd, strict=False)
            return model

    model = ModelClass.load_from_checkpoint(checkpoint_path, **kwargs)
    return model


def parse_model_args(args):
    kwargs = {}
    arg_types = {t.__name__: t for t in [int, float, str]}
    arg_types['bool'] = lambda v: v.lower() == 'true'  # special handling for bool
    for arg in args:
        name, value = arg.split('=', maxsplit=1)
        name, arg_type = name.split(':', maxsplit=1)
        kwargs[name] = arg_types[arg_type](value)
    return kwargs


def init_weights(module: nn.Module, name: str = '', exclude: Sequence[str] = ()):
    """Initialize the weights using the typical initialization schemes used in SOTA models."""
    if any(map(name.startswith, exclude)):
        return
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
