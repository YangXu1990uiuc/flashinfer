# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Qualification for the explicit, bounded Frost prefill research runner."""

import ast
from dataclasses import replace

import pytest
import torch

from flashinfer.experimental.frost_moe_prefill.runner import (
    PrefillResearchRunner,
    absolute_input,
    eligible_fc1,
    eligible_geometry,
)
from flashinfer.experimental.frost_moe_prefill.fc1_epilogue_source import _transform
from flashinfer.fused_moe import (
    CudnnFrostNvfp4Config,
    ExecutionConfig,
    MoEActivationPack,
    MoEWeightPack,
    SwiGLU,
)
from flashinfer.fused_moe.backends.cudnn_frost.nvfp4 import moe
from tests.moe.test_unified_moe_frost import _make_frost_case


def test_prefill_boundaries():
    for tokens in (1024, 8192, 12288):
        assert eligible_geometry(tokens, 2048, 1408, 64, 6, "swiglu")
        assert eligible_geometry(tokens, 2048, 768, 128, 8, "swiglu")
    for tokens in (0, 1, 1023, 12289):
        assert not eligible_geometry(tokens, 2048, 1408, 64, 6, "swiglu")
    assert not eligible_geometry(8192, 2048, 1408, 64, 6, "geglu")
    assert not eligible_geometry(8192, 4096, 1408, 64, 6, "swiglu")


def _relative_l2(a, b):
    assert torch.isfinite(a).all()
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-20)).item()


@pytest.mark.parametrize("geometry", ((64, 2048, 1408, 6), (128, 2048, 768, 8)))
@pytest.mark.parametrize("swizzled", (False, True))
@pytest.mark.parametrize("tokens", (1024, 1031))
def test_prefill_variants_routing_and_replay(geometry, swizzled, tokens):
    _, _, _, config, _, view = _make_frost_case("nvfp4", geometry)
    config = replace(
        config,
        quant=replace(config.quant, swizzled_scale_factors=swizzled),
        execution=ExecutionConfig(tune_max_num_tokens=12288),
    )
    e, h, _, k = geometry
    weights = MoEWeightPack({"cutlass_nvfp4": view})
    x = torch.randn(tokens, h, dtype=torch.bfloat16, device="cuda")
    if swizzled:
        from flashinfer.quantization import fp4_quantize

        # The canonical MoELayer helper always returns linear scales. Exercise
        # the separately supported swizzled input with the public quantizer.
        xq, xsf = fp4_quantize(
            x,
            torch.ones(1, dtype=torch.float32, device="cuda"),
            is_sf_swizzled_layout=True,
        )
        xsf = xsf.reshape(-1)
    else:
        xq, xsf = CudnnFrostNvfp4Config.prepare_activations(x, quant=config.quant)
    ids = torch.randint(0, e, (tokens, k), dtype=torch.int32, device="cuda")
    # Empty experts, duplicate slots and invalid routes exercise the metadata
    # contract. Invalid rows must contribute zero, including after graph replay.
    ids.remainder_(max(2, e // 2))
    ids[:4, 0] = -1
    ids[4:8, 0] = e
    ids[8:12, :] = 1
    scores = torch.rand(tokens, k, device="cuda").softmax(-1).bfloat16().float()
    act = MoEActivationPack(xq, xsf, ids, scores)
    runner = PrefillResearchRunner(config, torch.device("cuda", 0))
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(act, weights)
    state = inputs.launch_state
    variants = [key for key in state.launches if len(key) == 6]
    assert variants
    if any(eligible_fc1(kernel) for kernel in state.first):
        assert any("wide" in key[4] for key in variants), "FC1 candidate not exercised"
    assert any(key[4] == "gather" for key in variants)
    for key in variants:
        # Fix the physical GEMM pair on both sides. The stable adapter and
        # original source still own the three-part baseline key.
        ref = runner.forward(inputs, key[:3]).clone()
        result = runner.forward(inputs, key).clone()
        assert _relative_l2(result, ref) < 0.003, key
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replayed = runner.forward(inputs, key)
        ids.copy_(ids.roll(1, 0))
        scores.copy_(scores.flip(-1))
        inputs[1].copy_(inputs[1].roll(1, 0))
        ref = runner.forward(inputs, key[:3]).clone()
        replayed.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert _relative_l2(replayed, ref) < 0.003, key
        del graph
    scores.zero_()
    for key in variants:
        assert torch.count_nonzero(runner.forward(inputs, key)).item() == 0
    scores.fill_(1.0 / k)
    ids.fill_(-1)
    for key in variants:
        assert torch.count_nonzero(runner.forward(inputs, key)).item() == 0
    assert runner.pack_inputs(act, weights).launch_state is state


def test_source_transform_rejects_unknown_geometry():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("Requires packaged SM107 runtime materialization")
    first, _ = moe._selected_kernels(
        8192,
        2048,
        1408,
        64,
        6,
        torch.device("cuda", 0),
        SwiGLU(),
    )
    kernels = [a for a in first if eligible_fc1(a)]
    assert kernels
    for kernel in kernels:
        source = kernel.source_path.read_text()
        transformed = _transform(absolute_input(source, False))
        ast.parse(transformed)
        assert "num_epilogue_warps = 8" in transformed
        assert "epi_n = 64" in transformed
        with pytest.raises(ValueError):
            _transform(source.replace("num_gemms = 2", "num_gemms = 1"))
