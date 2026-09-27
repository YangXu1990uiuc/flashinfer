# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Immutable compiled tiny-MoE functions; tensor bindings stay per invocation."""

import functools


@functools.lru_cache(maxsize=128)
def build_first(t, swizzled, device, e, i, k, h):
    import torch, cutlass
    from cutlass import cute
    from cutlass.cute.runtime import make_fake_stream
    from . import tiny_fc1_two, tiny_fc1_four

    m = tiny_fc1_two if t == 1 else tiny_fc1_four
    s = t * k

    def fake(dtype, shape, alignment=16):
        strides = []
        v = 1
        for d in reversed(shape):
            strides.insert(0, v)
            v *= d
        return cute.runtime.make_fake_tensor(
            dtype, shape, stride=tuple(strides), assumed_align=alignment
        )

    args = [
        fake(cutlass.Uint8, (t, h // 2)),
        fake(cutlass.Uint8, (1, 128 * h // 16) if swizzled else (t, h // 16)),
        fake(cutlass.Uint8, (e, 2 * i, h // 2)),
        fake(cutlass.Uint8, (2, e, i, h // 16)),
        fake(cutlass.Float32, (e,), 4),
        fake(cutlass.Int32, (t, k), 4),
        fake(cutlass.BFloat16, (t, k, i)),
        fake(cutlass.Uint8, (s * i // 2,)),
        fake(cutlass.Uint8, (128 * s * i // 16,)),
        fake(cutlass.Float32, (1,), 4),
    ]
    with torch.cuda.device(device):
        compiled = cute.compile(
            m._launch,
            *args,
            make_fake_stream(use_tvm_ffi_env_stream=False),
            options="--enable-tvm-ffi"
        )
    return compiled


@functools.lru_cache(maxsize=128)
def build_second(t, device, e, i, k, h):
    import torch, cutlass
    from cutlass import cute
    from cutlass.cute.runtime import make_fake_stream
    from . import tiny_fc2 as m

    s = t * k

    def fake(dtype, shape, alignment=16):
        strides = []
        v = 1
        for d in reversed(shape):
            strides.insert(0, v)
            v *= d
        return cute.runtime.make_fake_tensor(
            dtype, shape, stride=tuple(strides), assumed_align=alignment
        )

    args = [
        fake(cutlass.Uint8, (s * i // 2,)),
        fake(cutlass.Uint8, (128 * s * i // 16,)),
        fake(cutlass.Uint8, (e, h, i // 2)),
        fake(cutlass.Uint8, (e, h, i // 16)),
        fake(cutlass.Float32, (e,), 4),
        fake(cutlass.Int32, (t, k), 4),
        fake(cutlass.Float32, (t, k), 4),
        fake(cutlass.BFloat16, (t, h)),
    ]
    with torch.cuda.device(device):
        return cute.compile(
            m._launch,
            *args,
            make_fake_stream(use_tvm_ffi_env_stream=False),
            options="--enable-tvm-ffi"
        )
