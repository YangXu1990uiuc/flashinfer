# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Cheap NVFP4 auto-admission; source and shortlist checks live in the runner."""

import torch

from ....fused_moe.api import QuantFormat, RoutingInputMode
from ..activations import activation_name
from ..support import shortlisted_moe_geometry


def _token_limit(experts, hidden, intermediate, topk, activation):
    """Only the validated NVFP4 SwiGLU geometries admit longer prefill calls."""
    if activation == "swiglu" and (experts, hidden, intermediate, topk) in (
        (64, 2048, 1408, 6),
        (12, 7168, 3072, 2),
    ):
        return 32768
    return 12288


def is_eligible(config, act, arch):
    x = act.hidden_states_q
    if not (
        arch == 107
        and config.quant.pair == (QuantFormat.NVFP4, QuantFormat.NVFP4)
        and config.quant.output == QuantFormat.BF16
        and act.routing_input_mode == RoutingInputMode.PackedPrecomputed
        and x.ndim == 2
        and x.dtype == torch.uint8
    ):
        return False
    try:
        name = activation_name(config.activation)
    except NotImplementedError:
        return False
    hidden = 2 * x.shape[1]
    limit = _token_limit(
        config.routing.num_experts,
        hidden,
        config.experts.intermediate_size,
        config.routing.top_k,
        name,
    )
    return shortlisted_moe_geometry(config, act, hidden_size=hidden, max_tokens=limit)


def create_runner(config, device):
    from .moe import automatic_candidate

    return automatic_candidate(config, device)
