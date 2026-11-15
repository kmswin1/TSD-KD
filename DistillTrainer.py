# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import textwrap
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    is_wandb_available,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_peft_available

from trl.models import prepare_deepspeed
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.gkd_config import GKDConfig
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.utils import (
    DataCollatorForChatML,
    disable_dropout_in_model,
    empty_cache,
    generate_model_card,
    get_comet_experiment_url,
    entropy_from_logits
)

if is_peft_available():
    from peft import PeftConfig

if is_wandb_available():
    import wandb

def normalize_entropy(entropy, mask=None):
    """Normalize entropy independently for each response over valid tokens."""
    entropy = entropy.float()
    if mask is not None:
        entropy = entropy.masked_fill(~mask.bool(), 0.0)
    total = entropy.sum(dim=-1, keepdim=True)
    return entropy / total.clamp_min(torch.finfo(entropy.dtype).tiny)


def top_k_entropy_position(norm_entropy, top_p=0.1, mask=None):
    """Return per-response *lengths* (not indices) of cumulative-entropy openers.

    Includes the token that first brings cumulative entropy to ``top_p``.
    Zero-valid-token rows have length zero.  Assumes response starts at column 0.
    """
    if not 0.0 <= top_p <= 1.0:
        raise ValueError("top_p must be in [0, 1]")
    B, L = norm_entropy.shape
    if L == 0 or top_p == 0:
        return torch.zeros(B, dtype=torch.long, device=norm_entropy.device)
    if mask is None:
        mask = torch.ones_like(norm_entropy, dtype=torch.bool)
    else:
        mask = mask.bool()

    cumulative = norm_entropy.float().cumsum(dim=-1)
    reaches = (cumulative >= top_p) & mask
    has_valid = mask.any(dim=-1)
    # If a response contains only zero-entropy tokens, include its first
    # valid token so that the opener is not silently empty.
    first = torch.where(reaches.any(dim=-1), reaches.long().argmax(dim=-1),
                        mask.long().argmax(dim=-1))
    return torch.where(has_valid, first + 1, torch.zeros_like(first))


def select_top_entropy_tokens(entropies, mask, ratio=0.1):
    """Select exactly ceil(ratio * valid_length) highest-entropy tokens per row."""
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("ratio must be in [0, 1]")
    mask = mask.bool()
    num_valid = mask.sum(dim=-1)
    num_to_keep = torch.ceil(num_valid.float() * ratio).long()
    if entropies.size(1) == 0:
        return mask
    # Stable ordering breaks entropy ties by the original token position.
    scores = entropies.detach().float().masked_fill(~mask, -float("inf"))
    ordering = torch.argsort(scores, dim=-1, descending=True, stable=True)
    ranks = torch.argsort(ordering, dim=-1)
    return mask & (ranks < num_to_keep.unsqueeze(-1))


def indirect_kd_loss(student_logits, teacher_logits, mask=None, T=1.0, K=10):
    """PL negative log likelihood for the teacher's descending candidate order."""
    B, L, V = student_logits.shape
    if T <= 0:
        raise ValueError("T must be positive")
    if K < 1:
        raise ValueError("K must be >= 1")
    if L == 0 or (mask is not None and not bool(mask.any())):
        return student_logits.sum() * 0.0

    # Compute ranking scores in FP32 for numerical robustness under AMP.
    student = student_logits.float() / T
    teacher = teacher_logits.float() / T
    k = min(K, V)
    top_student, candidate_indices = torch.topk(student, k, dim=-1)
    teacher_on_candidates = torch.gather(teacher, dim=-1, index=candidate_indices)
    teacher_preference = torch.argsort(teacher_on_candidates, dim=-1, descending=True)
    ranked_student = torch.gather(top_student, dim=-1, index=teacher_preference)
    # PL conditions on the candidates remaining at each stage (suffix sum).
    log_suffix = torch.logcumsumexp(ranked_student.flip(-1), dim=-1).flip(-1)
    per_token_loss = -(ranked_student - log_suffix).sum(dim=-1)

    if mask is not None:
        valid = mask.to(dtype=per_token_loss.dtype)
        return (per_token_loss * valid).sum() / valid.sum().clamp_min(1)
    return per_token_loss.mean()


class DistillTrainer(SFTTrainer):
    _tag_names = ["trl", "tsd-kd"]

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        teacher_model: Union[PreTrainedModel, nn.Module, str] = None,
        args: Optional[GKDConfig] = None,
        data_collator: Optional[DataCollator] = None,  # type: ignore
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional["PeftConfig"] = None,
        formatting_func: Optional[Callable] = None,
        token_entropy_percentile_threshold: Optional[float] = 1.0,
        indirect_kd_alpha: Optional[float] = 0.1,
    ):
        # add remove_unused_columns=False to the dataclass args
        args.remove_unused_columns = False
        data_collator = DataCollatorForChatML(tokenizer=processing_class, max_length=args.max_length)

        super().__init__(
            model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            formatting_func=formatting_func,
        )

        if args.teacher_model_init_kwargs is None:
            teacher_model_init_kwargs = {}
        elif not isinstance(teacher_model, str):
            raise ValueError(
                "You passed teacher_model_init_kwargs to the GKDConfig, but your teacher_model is already instantiated."
            )
        else:
            teacher_model_init_kwargs = args.teacher_model_init_kwargs
            teacher_model_init_kwargs["torch_dtype"] = (
                teacher_model_init_kwargs["torch_dtype"]
                if teacher_model_init_kwargs["torch_dtype"] in ["auto", None]
                else getattr(torch, teacher_model_init_kwargs["torch_dtype"])
            )
