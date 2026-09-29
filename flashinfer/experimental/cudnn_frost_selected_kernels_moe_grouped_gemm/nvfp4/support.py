# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Cheap NVFP4 auto-admission; source and shortlist checks live in the runner."""

import torch

from ....fused_moe.api import QuantFormat, RoutingInputMode
from ..activations import activation_name
from ..support import shortlisted_moe_geometry


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
    geometry = (
        config.routing.num_experts,
        2 * x.shape[1],
        config.experts.intermediate_size,
        config.routing.top_k,
    )
    if name == "swiglu" and geometry == (128, 2048, 768, 8):
        return 0 < act.num_tokens <= 12288
    return shortlisted_moe_geometry(config, act, hidden_size=2 * x.shape[1])


def create_runner(config, device):
    from .moe import automatic_candidate

    return automatic_candidate(config, device)
