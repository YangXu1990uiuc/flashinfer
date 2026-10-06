# Frost MoE prefill research handoff

This is an explicit research derivative of the formal Frost MoELayer backend
in flashinfer-ai/flashinfer#6035, pinned to `e603a3dd`. It carries a bounded
subset of the earlier cumulative POC for selective adoption. Yang Xu owns
the research handoff; the POC PR tracks it. It is not a maintained backend or
a request for direct merge. Adoption and release timing belong to the
upstream implementation.

`PrefillResearchRunner` inherits the formal runner's input validation, weight
preparation, graph ownership and staged tuning. Original candidates remain
available. The additional candidates have separate source/compiler identities
and participate in complete-pipeline tuning. The adapter is a research copy
of the pinned upstream adapter, retaining its explicit group-boundary ABI.
Available research profiles depend on the upstream-selected physical kernels;
this runner does not widen the upstream artifact shortlist.

The study is restricted to NVFP4 SwiGLU, tokens 1024–12288, and
`(experts, hidden, intermediate, top_k)` equal to `(64, 2048, 1408, 6)` or
`(128, 2048, 768, 8)`. Other shapes retain the base candidate set. The research
runner is deliberately absent from public configuration, automatic dispatch,
AOT registration and default imports. Input-fusion/input-reuse experiments
and the old tiny-token replacements are not included.
Use the canonical `cutlass_nvfp4` weight view, as the comparison harness does.

Explicit benchmark, from the repository root:

```bash
PYTHONPATH=.:benchmarks python benchmarks/bench_frost_prefill_poc.py \
  --geometry 64,2048,1408,6 --tokens 1024,8192 \
  --backends frost,poc,cute_dsl,trtllm --ablation --output comparison.jsonl
```

The benchmark uses the public weight preparation and ordinary autotuner for
each backend, then times complete MoELayer calls in warm CUDA Graph replay.
It checks outputs and changed-input replay after timing. Weight preparation,
input quantization, routing top-k generation, compilation and tuning are
excluded. It is a synthetic single-layer comparison, not serving E2E.
Hardware-specific evidence is retained in the private handoff.
The optional ablation holds both selected physical GEMM kernels fixed, so
implementation benefits can be separated from configuration selection.

```bash
pytest tests/experimental/test_frost_moe_prefill_poc.py
```

The earlier prototype has unresolved synchronization diagnostics in inherited
scheduling paths. Passing numerical and replay checks here does not resolve
that qualification gap. Keep these candidates experimental until the
adopting implementation completes its architecture/compiler and sanitizer
qualification.

Credit to Yanqin Zhai and the FlashInfer/cuDNN Frost contributors for the
formal runner, native adapter, compiler integration and generated kernels.
This handoff adapts the earlier research candidates to that foundation and
retains the existing source attribution.
