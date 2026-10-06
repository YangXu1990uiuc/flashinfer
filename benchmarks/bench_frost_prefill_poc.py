# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Explicit NVFP4 research comparison on the formal Frost MoELayer baseline.

Run from the repository root with PYTHONPATH including benchmarks. All weights
use the existing backend preparation APIs. Timings include the full MoELayer
GPU work, excluding weight/input preparation, compilation and autotuning.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch

from flashinfer.autotuner import autotune
from flashinfer.experimental.frost_moe_prefill.runner import (
    PrefillResearchRunner,
    source_identity,
)
from flashinfer.fused_moe import (
    BackendOptions,
    CudnnFrostNvfp4Config,
    CuteDslConfig,
    CutlassNvfp4Config,
    MoEActivationPack,
    MoELayer,
    TrtllmFp4Config,
)
from bench_cudnn_frost_common import check_idle_gpu, measure
import bench_cudnn_frost_moe_nvfp4 as benchmark


def relative_l2(actual, reference):
    if not torch.isfinite(actual).all().item():
        raise AssertionError("Nonfinite MoE output")
    return (
        (actual.float() - reference.float()).norm()
        / reference.float().norm().clamp_min(1e-20)
    ).item()


def matched_pair_ablation(layer, act, weights, rounds, batch):
    """Hold the selected physical GEMM pair fixed when attributing a gain."""
    runner, winner = next(reversed(layer._winners.values()))
    inputs = runner.pack_inputs(act, weights)
    base = winner[:3]
    keys = {"original": base}
    keys.update(
        (key[4], key)
        for key in inputs.launch_state.launches
        if len(key) == 6 and key[:3] == base
    )
    reference = runner.forward(inputs, base).clone()
    graphs, outputs, errors = {}, {}, {}
    for label, key in keys.items():
        errors[label] = relative_l2(runner.forward(inputs, key), reference)
        assert errors[label] < 0.003, (label, errors[label])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(batch):
                output = runner.forward(inputs, key)
        graphs[label], outputs[label] = graph, output
    labels = list(graphs)
    samples = {label: [] for label in labels}
    for idx in range(rounds):
        order = labels[idx % len(labels) :] + labels[: idx % len(labels)]
        if idx % 2:
            order.reverse()
        for label in order:
            samples[label].append(measure(graphs[label], 10, batch) * 1000)
    torch.cuda.synchronize()
    return dict(
        physical_pair=str(base),
        median_us={key: statistics.median(value) for key, value in samples.items()},
        samples_us=samples,
        relative_l2=errors,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", default="64,2048,1408,6")
    parser.add_argument("--tokens", default="1024,8192")
    parser.add_argument("--backends", default="frost,poc,cute_dsl,trtllm")
    parser.add_argument(
        "--profiles", default="gather,wide,wide_gather,absolute_wide_gather"
    )
    parser.add_argument("--routing", choices=("uniform", "skew"), default="uniform")
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    geometry = tuple(map(int, args.geometry.split(",")))
    e, h, _, k = geometry
    tokens_list = tuple(map(int, args.tokens.split(",")))
    labels = args.backends.split(",")
    props = torch.cuda.get_device_properties(0)
    check_idle_gpu(props.uuid)
    with args.output.open("x") as file:

        def emit(**record):
            text = json.dumps(record, default=str)
            print(text, flush=True)
            file.write(text + "\n")
            file.flush()

        config, weights, audit, _ = benchmark.prepare_problem(
            SimpleNamespace(tokens=tokens_list, seed=2401, weight_std=0.02),
            "swiglu",
            geometry,
        )
        emit(
            kind="scope",
            geometry=geometry,
            device=str(props),
            torch=torch.__version__,
            baseline="e603a3dd93b14643ecfcecf0b2182ae593acdc32",
            poc_source_identity=source_identity(),
            backend_audit=audit,
            included="complete MoELayer warm CUDA Graph replay, EP1",
            excluded="weight/input preparation, router top-k, compilation, autotuning, serving",
        )
        configs = {
            "frost": CudnnFrostNvfp4Config,
            "poc": CudnnFrostNvfp4Config,
            "cute_dsl": CuteDslConfig,
            "trtllm": TrtllmFp4Config,
            "cutlass": CutlassNvfp4Config,
        }
        for tokens in tokens_list:
            torch.manual_seed(2401 + tokens)
            x = torch.randn(tokens, h, dtype=torch.bfloat16, device="cuda")
            xq, xsf = CutlassNvfp4Config.prepare_activations(x, quant=config.quant)
            logits = torch.rand(tokens, e, device="cuda")
            if args.routing == "skew":
                logits[: tokens // 2, 0] += 2
            ids = logits.topk(k, dim=1).indices.int()
            scores = torch.rand(tokens, k, device="cuda").softmax(-1).bfloat16().float()
            act = MoEActivationPack(xq, xsf, ids, scores)
            layers, outputs, graphs, graph_outputs, keepalive = {}, {}, {}, {}, []
            for label in labels:
                cfg = replace(config, backend=BackendOptions((configs[label](),)))
                layer = MoELayer(cfg)
                layer._additional_candidates = lambda *unused: []
                if label == "poc":
                    runner = PrefillResearchRunner(
                        cfg, torch.device("cuda", 0), profiles=args.profiles.split(",")
                    )
                    runner.check_support()
                    runner.build()
                    layer.runners = [runner]
                layers[label] = layer
                emit(kind="tune_start", tokens=tokens, backend=label)
                start = time.monotonic()
                with autotune(tuning_buckets=(tokens,)):
                    layer(act, weights)
                outputs[label] = layer(act, weights).clone()
                for _ in range(3):
                    layer(act, weights)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(args.batch):
                        result = layer(act, weights)
                graphs[label], graph_outputs[label] = graph, result
                keepalive.append(result)
                result.fill_(float("nan"))
                graph.replay()
                assert relative_l2(result, outputs[label]) < 0.08
                emit(
                    kind="tune_done",
                    tokens=tokens,
                    backend=label,
                    seconds=time.monotonic() - start,
                    winner=layer.winner_backend,
                    tactics=[str(t) for _, t in layer._winners.values()],
                )
            reference = benchmark.reference(
                act, weights.get_view("cutlass_nvfp4"), "swiglu", 8
            )
            errors = {
                label: relative_l2(value[: len(reference)], reference)
                for label, value in outputs.items()
            }
            assert max(errors.values()) < 0.12, errors
            if "frost" in outputs:
                cross = {
                    label: relative_l2(value, outputs["frost"])
                    for label, value in outputs.items()
                }
                assert max(cross.values()) < 0.08, cross
            samples = {label: [] for label in labels}
            telemetry = []
            for round_idx in range(args.rounds):
                check_idle_gpu(props.uuid)
                order = (
                    labels[round_idx % len(labels) :]
                    + labels[: round_idx % len(labels)]
                )
                if round_idx % 2:
                    order.reverse()
                for label in order:
                    samples[label].append(measure(graphs[label], 10, args.batch) * 1000)
                telemetry.append(benchmark.gpu_telemetry())
            if args.ablation and "poc" in layers:
                emit(
                    kind="matched_pair_ablation",
                    tokens=tokens,
                    **matched_pair_ablation(
                        layers["poc"], act, weights, args.rounds, args.batch
                    ),
                )
            # Verify changed data and routing after timing, using the captured storage.
            ids.copy_(ids.roll(1, 0))
            scores.copy_(scores.flip(-1))
            xq.copy_(xq.roll(1, 0))
            changed_errors = {}
            for label in labels:
                expected = layers[label](act, weights).clone()
                graph_outputs[label].fill_(float("nan"))
                graphs[label].replay()
                changed_errors[label] = relative_l2(graph_outputs[label], expected)
            assert max(changed_errors.values()) < 0.08, changed_errors
            emit(
                kind="result",
                tokens=tokens,
                routing=args.routing,
                geometry=geometry,
                median_us={
                    label: statistics.median(values)
                    for label, values in samples.items()
                },
                samples_us=samples,
                reference_relative_l2=errors,
                changed_replay_relative_l2=changed_errors,
                telemetry=telemetry,
            )


if __name__ == "__main__":
    main()
