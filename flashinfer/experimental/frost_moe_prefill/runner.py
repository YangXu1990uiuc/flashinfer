# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Explicit research runner; deliberately absent from automatic registration."""

import ast
import functools
import hashlib
import os
from pathlib import Path
import tempfile

import torch

from ...fused_moe.backends.cudnn_frost import runtime as common
from ...fused_moe.backends.cudnn_frost.activations import activation_name
from ...fused_moe.backends.cudnn_frost.nvfp4 import moe, runtime
from ...fused_moe.backends.cudnn_frost.tuning import prepared_state
from .fc1_epilogue_source import _transform as transform_epilogue

_PROFILES = ("gather", "wide", "wide_gather", "absolute_wide_gather")


def eligible_geometry(tokens, hidden, intermediate, experts, topk, activation):
    return (
        activation == "swiglu"
        and 1024 <= tokens <= 12288
        and (experts, hidden, intermediate, topk)
        in ((64, 2048, 1408, 6), (128, 2048, 768, 8))
    )


def eligible_fc1(kernel):
    meta = kernel.tactic_metadata
    return (
        kernel.fc1
        and not kernel.swap_ab
        and "output_scale" in kernel.launch_tail
        and meta.get("store_mode") == "stg"
        and meta.get("cta_group") == 2
        and meta.get("cta_tile", {}).get("m") == 128
        and meta.get("cta_tile", {}).get("n") == 256
    )


@functools.lru_cache(maxsize=1)
def source_identity():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for name in (
        "runner.py",
        "fc1_epilogue_source.py",
        "output_source.py",
        "moe_nvfp4.cu",
    ):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def absolute_input(source, swap_ab):
    """Change only input addressing; output expert-boundary updates stay intact."""
    start = source.index("        previous_group_begin = cutlass.Int32(-1)")
    end = source.index("                for k_tile_idx in range(num_k_tiles):", start)
    source = (
        source[:start]
        + source[start:end].replace("moe_aligned_offsets", "True")
        + source[end:]
    )
    if swap_ab:
        before = "                sfb_n_block = coord_n_desc // 128"
        after = "                sfb_n_block = start_sf_block_n + coord_n_group // 128"
        loop = "    for _sfb_op in _sfb_operands:\n"
        extra = "        rest_n = _sfb_op.shape[0] // (512 * rest_k)\n"
    else:
        before = "                sfa_m_block = coord_m_desc // 128"
        after = "                sfa_m_block = start_sf_block_m + coord_m_group // 128"
        loop = "    for _sfa_op in _sfa_operands:\n"
        extra = "        rest_m = _sfa_op.shape[0] // (512 * rest_k)\n"
    if source.count(before) != 1 or source.count(loop) != 1:
        raise ValueError("Unsupported Frost input descriptor structure")
    return source.replace(before, after).replace(loop, loop + extra)


def load_variant(kernel, device, profile):
    wide = kernel.fc1 and "wide" in profile
    absolute = profile.startswith("absolute")
    if not wide and not absolute:
        return common._load_kernel(kernel, device)
    source = kernel.source_path.read_text()
    if absolute:
        source = absolute_input(source, kernel.swap_ab)
    if wide:
        source = transform_epilogue(source)
    ast.parse(source)
    digest = hashlib.sha256(source.encode()).hexdigest()
    from ...jit.env import FLASHINFER_GEN_SRC_DIR

    root = FLASHINFER_GEN_SRC_DIR / "frost_prefill_poc"
    root.mkdir(parents=True, exist_ok=True)
    path = root / (digest + ".py")
    if not path.exists():
        fd, temporary = tempfile.mkstemp(dir=root, suffix=".py.tmp")
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(source)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return common._load_source(path, digest, kernel.arch, device.index)


@functools.lru_cache(maxsize=1)
def _module():
    from ...jit.core import gen_jit_spec, sm107a_nvcc_flags

    return gen_jit_spec(
        "frost_nvfp4_prefill_poc_" + source_identity()[:16],
        [Path(__file__).with_name("moe_nvfp4.cu")],
        extra_cuda_cflags=sm107a_nvcc_flags,
    ).build_and_load()


class PrefillResearchRunner(moe.CudnnFrostNvfp4MoeRunner):
    """Retain every base candidate and append bounded, separately keyed variants.

    The parent owns validation, runtime pointers, graph retention and staged
    tuning. This subclass adds no automatic registration or production policy.
    Instantiate explicitly in the research harness, before graph capture.
    """

    backend_key = "cudnn_frost_nvfp4_prefill_poc"

    def __init__(self, config, device, *, profiles=_PROFILES):
        self.profiles = tuple(profiles)
        if not self.profiles or any(p not in _PROFILES for p in self.profiles):
            raise ValueError("Unknown Frost prefill research profile")
        super().__init__(config, device)

    def _cache_key_extras(self):
        return super()._cache_key_extras() + (self.profiles, source_identity())

    def pack_inputs(self, act, weights):
        inputs = super().pack_inputs(act, weights)
        state = inputs.launch_state
        if getattr(state, "prefill_poc_ready", False):
            return inputs
        t, packed_h = inputs[1].shape
        h = packed_h * 2
        i = self.config.experts.intermediate_size
        e, k = self.config.routing.num_experts, self.config.routing.top_k
        if not eligible_geometry(
            t, h, i, e, k, activation_name(self.config.activation)
        ):
            return inputs
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare Frost prefill research plans before capture")
        module = _module()
        required = state.workspace.numel()
        extra = {}
        # Keep the parent's three-part keys for staged tuning. Experimental
        # variants are considered only by the complete-pipeline outer tuner.
        for a in state.first:
            for b in state.second:
                base = (moe._TAG, a.tactic, b.tactic)
                for profile in self.profiles:
                    if "wide" in profile and not eligible_fc1(a):
                        continue
                    if profile.startswith("absolute") and e != 64:
                        continue
                    key = base + (
                        "prefill-poc",
                        profile,
                        common._tactic_digest(source_identity()),
                    )
                    plan = module.make_plan(
                        load_variant(a, self.device, profile),
                        load_variant(b, self.device, profile),
                        t,
                        h,
                        i,
                        e,
                        k,
                        self.device.index,
                        a.workspace_bytes,
                        b.workspace_bytes,
                        a.gated,
                        [runtime._TAIL_SLOTS[name] for name in a.launch_tail],
                        [runtime._TAIL_SLOTS[name] for name in b.launch_tail],
                        a.swap_ab,
                        b.swap_ab,
                        self.config.quant.swizzled_scale_factors is True,
                        False,
                        "gather" in profile,
                    )
                    extra[key] = plan
                    required = max(required, plan["workspace_size"]())
        if required > state.workspace.numel():
            # No graph can reference a newly prepared state's old workspace.
            state.workspace = torch.empty(
                required, dtype=torch.uint8, device=self.device
            )
        for key, plan in extra.items():
            state.plans[key], state.launches[key] = plan, plan["run"]
        state.prefill_poc_ready = True
        return inputs

    def get_valid_tactics(self, inputs, profile):
        original = super().get_valid_tactics(inputs, profile)
        state = prepared_state(self, inputs, hidden_multiplier=2)
        if state is None:
            return original
        selected = {key for key in original if len(key) == 3}
        return list(
            dict.fromkeys(
                original
                + [
                    key
                    for key in state.launches
                    if len(key) == 6 and key[:3] in selected
                ]
            )
        )
