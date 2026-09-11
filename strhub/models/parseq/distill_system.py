# Scene Text Recognition Model Hub
# Knowledge Distillation System for Custom PARSeq models
# Based on sviptr-distill response-based distillation and confidence weighting.

import math
from pathlib import Path
from typing import Any, Optional, Sequence, Union
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from pytorch_lightning.utilities.types import STEP_OUTPUT
from timm.optim import create_optimizer_v2
from torch.optim.lr_scheduler import OneCycleLR

from strhub.models.parseq.system import PARSeq
from strhub.models.parseq.model import PARSeq as Model
from strhub.models.utils import get_pretrained_weights


def _load_yaml_config(config_path: Union[str, Path]) -> dict:
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg if cfg is not None else {}


def _get_teacher_config() -> dict:
    """Carrega sempre a arquitetura padrão do professor de configs/model/parseq.yaml."""
    root_dir = Path(__file__).resolve().parents[3]
    teacher_yaml = root_dir / 'configs' / 'model' / 'parseq.yaml'
    if teacher_yaml.exists():
        cfg = _load_yaml_config(teacher_yaml)
        cfg.pop('_target_', None)
        cfg.pop('name', None)
        return cfg
    return {
        'patch_size': [4, 8],
        'embed_dim': 384,
        'enc_num_heads': 6,
        'enc_mlp_ratio': 4,
        'enc_depth': 12,
        'dec_num_heads': 12,
        'dec_mlp_ratio': 4,
        'dec_depth': 1,
        'lr': 4.5e-4,
        'weight_decay': 1e-3,
        'perm_num': 4,
        'perm_forward': True,
        'perm_mirrored': True,
        'perm_forward_weight': 2.0,
        'label_smoothing': 0.05,
        'dropout': 0.15,
        'decode_ar': False,
        'refine_iters': 1,
    }


class PARSeqDistill(PARSeq):
    """Modelo de Destilação de Conhecimento para PARSeq.

    Utiliza SEMPRE a arquitetura configs/model/parseq.yaml como Professor e
    destila conhecimento para qualquer arquitetura PARSeq de aluno customizada.
    
    Características herdadas do sviptr-distill:
      - Destilação de probabilidade (Hinton KD) com temperatura (T).
      - Ponderação por amostra via confiança do professor (teacher_confidence):
        amostras onde o professor tem baixa confiança pesam menos na destilação,
        permitindo que a loss de rótulo real (CrossEntropy) domine e protegendo
        o aluno de imitar erros do professor.
      - Alinhamento de contexto pelas permutações autorregressivas/bidirecionais.
      - Alinhamento automático de vocabulário e resolução entre professor e aluno.
    """

    def __init__(
        self,
        charset_train: str,
        charset_test: str,
        max_label_length: int,
        batch_size: int,
        lr: float,
        warmup_pct: float,
        weight_decay: float,
        img_size: Sequence[int],
        patch_size: Sequence[int],
        embed_dim: int,
        enc_num_heads: int,
        enc_mlp_ratio: int,
        enc_depth: int,
        dec_num_heads: int,
        dec_mlp_ratio: int,
        dec_depth: int,
        perm_num: int,
        perm_forward: bool,
        perm_mirrored: bool,
        decode_ar: bool,
        refine_iters: int,
        dropout: float,
        perm_forward_weight: float = 2.0,
        label_smoothing: float = 0.05,
        # Hiperparâmetros de Destilação
        teacher_ckpt: Optional[str] = None,
        alpha: float = 0.5,
        temperature: float = 2.0,
        use_teacher_conf: bool = True,
        teacher_img_size: Optional[Sequence[int]] = None,
        distill_mode: str = 'perms',
        **kwargs: Any,
    ) -> None:
        # 1. Inicializa o modelo aluno (self) com sua própria arquitetura
        super().__init__(
            charset_train=charset_train,
            charset_test=charset_test,
            max_label_length=max_label_length,
            batch_size=batch_size,
            lr=lr,
            warmup_pct=warmup_pct,
            weight_decay=weight_decay,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            enc_num_heads=enc_num_heads,
            enc_mlp_ratio=enc_mlp_ratio,
            enc_depth=enc_depth,
            dec_num_heads=dec_num_heads,
            dec_mlp_ratio=dec_mlp_ratio,
            dec_depth=dec_depth,
            perm_num=perm_num,
            perm_forward=perm_forward,
            perm_mirrored=perm_mirrored,
            decode_ar=decode_ar,
            refine_iters=refine_iters,
            dropout=dropout,
            perm_forward_weight=perm_forward_weight,
            label_smoothing=label_smoothing,
            **kwargs,
        )
        self.save_hyperparameters()

        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.use_teacher_conf = bool(use_teacher_conf)
        self.distill_mode = str(distill_mode).lower()

        # 2. Carrega o Professor (configs/model/parseq.yaml)
        # Carrega o modelo professor com seus PRÓPRIOS hparams do checkpoint para evitar mismatch de dimensões
        self.teacher = self._build_and_load_teacher(teacher_ckpt)

        # Determina resolução geométrica do professor
        if teacher_img_size is not None:
            self.teacher_img_size = tuple(teacher_img_size)
        else:
            t_h = getattr(self.teacher.hparams, 'img_size', None)
            if t_h is not None:
                self.teacher_img_size = tuple(t_h)
            else:
                self.teacher_img_size = (32, 128)

        # 3. Congela o professor completamente
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # 4. Configura mapeamento de vocabulário aluno -> professor se charsets diferirem
        self._setup_charset_remap()

    def _build_and_load_teacher(self, teacher_ckpt: Optional[str]) -> PARSeq:
        """Carrega o modelo professor com sua própria arquitetura nativa (parseq.yaml)."""
        if teacher_ckpt and Path(teacher_ckpt).is_file():
            print(f"[PARSeqDistill] Carregando professor via load_from_checkpoint: {teacher_ckpt}")
            try:
                # Carrega o checkpoint com os hiperparâmetros próprios do professor (ex.: max_label_length=25, 94 chars)
                teacher = PARSeq.load_from_checkpoint(teacher_ckpt)
                print("[PARSeqDistill] Professor instanciado e pesos carregados com sucesso!")
                return teacher
            except Exception as e:
                print(f"[PARSeqDistill] load_from_checkpoint padrão falhou ({e}), tentando reconstrução via hparams...")
                ckpt = torch.load(teacher_ckpt, map_location='cpu')
                hparams = ckpt.get('hyper_parameters', {})
                if hparams:
                    teacher = PARSeq(**hparams)
                    state_dict = ckpt.get('state_dict', ckpt)
                    clean_sd = {k.replace('model.', ''): v for k, v in state_dict.items() if not k.startswith('teacher.')}
                    try:
                        teacher.model.load_state_dict(clean_sd, strict=True)
                    except Exception:
                        teacher.model.load_state_dict(clean_sd, strict=False)
                    return teacher
                raise e

        # Se nenhum checkpoint fornecido ou especificado 'pretrained'
        print("[PARSeqDistill] Inicializando professor padrão configs/model/parseq.yaml com pesos oficiais 'parseq'...")
        teacher_cfg = _get_teacher_config()
        full_charset = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
        teacher = PARSeq(
            charset_train=full_charset,
            charset_test=full_charset,
            max_label_length=25,
            batch_size=self.batch_size,
            lr=teacher_cfg.get('lr', 4.5e-4),
            warmup_pct=self.warmup_pct,
            weight_decay=teacher_cfg.get('weight_decay', 1e-3),
            img_size=[32, 128],
            patch_size=teacher_cfg.get('patch_size', [4, 8]),
            embed_dim=teacher_cfg.get('embed_dim', 384),
            enc_num_heads=teacher_cfg.get('enc_num_heads', 6),
            enc_mlp_ratio=teacher_cfg.get('enc_mlp_ratio', 4),
            enc_depth=teacher_cfg.get('enc_depth', 12),
            dec_num_heads=teacher_cfg.get('dec_num_heads', 12),
            dec_mlp_ratio=teacher_cfg.get('dec_mlp_ratio', 4),
            dec_depth=teacher_cfg.get('dec_depth', 1),
            perm_num=teacher_cfg.get('perm_num', 4),
            perm_forward=teacher_cfg.get('perm_forward', True),
            perm_mirrored=teacher_cfg.get('perm_mirrored', True),
            decode_ar=teacher_cfg.get('decode_ar', False),
            refine_iters=teacher_cfg.get('refine_iters', 1),
            dropout=teacher_cfg.get('dropout', 0.15),
            perm_forward_weight=teacher_cfg.get('perm_forward_weight', 2.0),
            label_smoothing=teacher_cfg.get('label_smoothing', 0.05),
        )
        try:
            teacher.model.load_state_dict(get_pretrained_weights('parseq'))
            print("[PARSeqDistill] Pesos pré-treinados oficiais do PARSeq carregados!")
        except Exception as e:
            print(f"[PARSeqDistill] Aviso: Não foi possível carregar pretrained oficial ({e}).")
        return teacher

    def _setup_charset_remap(self) -> None:
        """Mapeia os logits de saída do aluno para as colunas equivalentes do professor.
        A cabeça (head) do PARSeq produz previsões para [EOS] (índice 0) e os caracteres (1..num_classes).
        """
        s_tok = self.tokenizer
        t_tok = self.teacher.tokenizer

        num_student_classes = len(s_tok) - 2
        num_teacher_classes = len(t_tok) - 2

        if len(s_tok) == len(t_tok) and getattr(s_tok, '_itos', None) == getattr(t_tok, '_itos', None):
            self.char_remap = None
            return

        remap = []
        for k in range(num_student_classes):
            if k == 0:
                # Índice 0 é sempre [EOS] em ambos
                remap.append(0)
            else:
                char = s_tok._itos[k]
                if char in t_tok._stoi:
                    remap.append(t_tok._stoi[char])
                else:
                    remap.append(0)

        self.register_buffer('char_remap', torch.tensor(remap, dtype=torch.long))
        print(f"[PARSeqDistill] Remapeamento de vocabulário ativo: Aluno ({num_student_classes} classes) -> Professor ({num_teacher_classes} classes)")

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, 'teacher'):
            self.teacher.eval()
        return self

    def configure_optimizers(self):
        """Otimiza estritamente os parâmetros do aluno (self.model), ignorando o professor."""
        agb = self.trainer.accumulate_grad_batches
        lr_scale = agb * math.sqrt(self.trainer.num_devices) * self.batch_size / 256.0
        lr = lr_scale * self.lr
        optim = create_optimizer_v2(self.model, 'adamw', lr, self.weight_decay)
        sched = OneCycleLR(
            optim, lr, self.trainer.estimated_stepping_batches, pct_start=self.warmup_pct, cycle_momentum=False
        )
        return {'optimizer': optim, 'lr_scheduler': {'scheduler': sched, 'interval': 'step'}}

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Remove os pesos do professor do checkpoint salvo para gerar arquivos leves (somente o aluno)."""
        state_dict = checkpoint.get('state_dict', {})
        keys_to_remove = [k for k in state_dict if k.startswith('teacher.')]
        for k in keys_to_remove:
            del state_dict[k]

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, labels = batch
        dev = self._device

        # 1. Alinha resolução de entrada do professor se necessário
        if self.teacher_img_size is not None and tuple(images.shape[-2:]) != tuple(self.teacher_img_size):
            teacher_images = F.interpolate(
                images, size=tuple(self.teacher_img_size), mode='bicubic', align_corners=False
            )
        else:
            teacher_images = images

        # 2. Tokenização para Aluno e Professor
        tgt_s = self.tokenizer.encode(labels, dev)
        tgt_t = self.teacher.tokenizer.encode(labels, dev)

        # Sincroniza comprimento de sequência das anotações (ambos contêm exatamente os mesmos caracteres até EOS)
        min_seq_len = tgt_s.shape[1]
        tgt_t = tgt_t[:, :min_seq_len]

        # 3. Codificação visual
        memory_s = self.model.encode(images)

        with torch.no_grad():
            memory_t = self.teacher.model.encode(teacher_images)

            # Cálculo de Confiança do Professor (mecanismo central do sviptr-distill)
            t_eval_logits = self.teacher.forward(teacher_images)
            if self.char_remap is not None:
                t_eval_logits = t_eval_logits[..., self.char_remap]
            t_eval_probs = t_eval_logits.softmax(dim=-1)

            # Produto das probabilidades máximas nos caracteres válidos (excluindo padding)
            tgt_non_pad = (tgt_s != self.pad_id)[:, 1:]  # descarta <bos>
            max_probs = t_eval_probs.max(dim=-1).values
            min_l = min(max_probs.shape[1], tgt_non_pad.shape[1])
            mask_conf = tgt_non_pad[:, :min_l]
            p_conf = max_probs[:, :min_l]
            p_masked = torch.where(mask_conf, p_conf, torch.ones_like(p_conf))
            teacher_confidence = p_masked.prod(dim=1).clamp(min=0.0, max=1.0)  # (N,)

        # 4. Permutações de Treinamento
        tgt_perms = self.gen_tgt_perms(tgt_s)
        tgt_s_in = tgt_s[:, :-1]
        tgt_s_out = tgt_s[:, 1:]
        tgt_s_pad_mask = (tgt_s_in == self.pad_id) | (tgt_s_in == self.eos_id)

        tgt_t_in = tgt_t[:, :-1]
        tgt_t_pad_mask = (tgt_t_in == self.teacher.pad_id) | (tgt_t_in == self.teacher.eos_id)

        loss = 0.0
        loss_numel = 0.0
        total_ce = 0.0
        total_kd = 0.0
        n = (tgt_s_out != self.pad_id).sum().item()

        for i, perm in enumerate(tgt_perms):
            tgt_mask, query_mask = self.generate_attn_masks(perm)

            # Decodificação do Aluno
            out_s = self.model.decode(tgt_s_in, memory_s, tgt_mask, tgt_s_pad_mask, tgt_query_mask=query_mask)
            logits_s = self.model.head(out_s)  # (N, L, C_s)

            # Decodificação do Professor na mesma permutação
            with torch.no_grad():
                out_t = self.teacher.model.decode(tgt_t_in, memory_t, tgt_mask, tgt_t_pad_mask, tgt_query_mask=query_mask)
                logits_t = self.teacher.model.head(out_t)  # (N, L, C_t)
                if self.char_remap is not None:
                    logits_t = logits_t[..., self.char_remap]

            weight = self.perm_forward_weight if (i == 0 and self.perm_forward) else 1.0

            # 4.1 Loss de Rótulo Real (CrossEntropy com label smoothing)
            loss_ce = F.cross_entropy(
                logits_s.flatten(end_dim=1),
                tgt_s_out.flatten(),
                ignore_index=self.pad_id,
                label_smoothing=self.label_smoothing,
            )

            # 4.2 Loss de Destilação (Hinton KD)
            valid_mask = (tgt_s_out != self.pad_id).float()
            log_probs_s = (logits_s / self.temperature).log_softmax(dim=-1)
            probs_t = (logits_t / self.temperature).softmax(dim=-1)

            kl = F.kl_div(log_probs_s, probs_t, reduction='none').sum(dim=-1)  # (N, L)
            kl = kl * valid_mask

            # Ponderação pela confiança do professor (sviptr-distill)
            if self.use_teacher_conf:
                kl = kl * teacher_confidence.unsqueeze(1)

            num_valid = valid_mask.sum().clamp(min=1.0)
            loss_kd = (kl.sum() / num_valid) * (self.temperature ** 2)

            # Combinação das perdas
            perm_loss = (1.0 - self.alpha) * loss_ce + self.alpha * loss_kd

            loss += weight * n * perm_loss
            loss_numel += weight * n
            total_ce += weight * n * loss_ce.item()
            total_kd += weight * n * loss_kd.item()

            if i == 1:
                tgt_s_out = torch.where(tgt_s_out == self.eos_id, self.pad_id, tgt_s_out)
                n = (tgt_s_out != self.pad_id).sum().item()

        loss = loss / loss_numel

        # 5. Destilação NAR opcional
        if self.distill_mode in ('both', 'all'):
            with torch.no_grad():
                t_nar_probs = (t_eval_logits / self.temperature).softmax(dim=-1)
            s_eval_logits = self.forward(images, self.max_label_length)
            s_nar_log_probs = (s_eval_logits / self.temperature).log_softmax(dim=-1)
            nar_kl = F.kl_div(s_nar_log_probs, t_nar_probs, reduction='batchmean') * (self.temperature ** 2)
            loss = loss + 0.2 * self.alpha * nar_kl

        # Registra métricas detalhadas
        self.log('loss', loss, prog_bar=True)
        self.log('loss_ce', total_ce / loss_numel)
        self.log('loss_kd', total_kd / loss_numel)
        self.log('teacher_conf', teacher_confidence.mean(), prog_bar=True)

        return loss
