# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Prepared quantization from Frost's existing transposed BF16 SMEM epilogue."""

import ast
import hashlib
import os
import tempfile

BODY = """
                        for _qs in cutlass.range_constexpr(epi_n * (epi_rows_per_mma_m // 16) // (num_epilogue_warps * 32)):
                            _qt = tidx + _qs * (num_epilogue_warps * 32)
                            _token_local = _qt // (epi_rows_per_mma_m // 16)
                            _column_local = (_qt % (epi_rows_per_mma_m // 16)) * 16
                            _token = col + _token_local
                            _column = coord_m + _column_local
                            if _token < group_end and _column < M:
                                _base = (_column_local // 64) * (64 * epi_n) + _token_local * 64
                                _xor = (_token_local % 8) * 8
                                _qvals = [(_tsv_0.data_ptr(_base + (((_column_local + _qi) % 64) ^ _xor))).load().to(cutlass.Float32) for _qi in range(16)]
                                _maxs = [cute.math.abs(_qvals[_qi]) for _qi in range(16)]
                                for _qi in cutlass.range_constexpr(8):
                                    _maxs[_qi] = cutlass.max(_maxs[_qi], _maxs[_qi + 8])
                                for _qi in cutlass.range_constexpr(4):
                                    _maxs[_qi] = cutlass.max(_maxs[_qi], _maxs[_qi + 4])
                                for _qi in cutlass.range_constexpr(2):
                                    _maxs[_qi] = cutlass.max(_maxs[_qi], _maxs[_qi + 2])
                                _amax = cutlass.max(_maxs[0], _maxs[1])
                                _global = _quant_global_pre
                                _sf_bits = cvt_f32_to_e4m3(_amax * cutlass.Float32(1.0 / 6.0) * _global)
                                _sf_f32 = cvt_e4m3_to_f32_via_f16(_sf_bits)
                                _inv = cutlass.Float32(0.0)
                                if _amax != cutlass.Float32(0.0):
                                    _inv = _global * rcp_approx_ftz(_sf_f32)
                                _q = [_v * _inv for _v in _qvals]
                                _lo = cvt_e2m1x8_f32(_q[0], _q[1], _q[2], _q[3], _q[4], _q[5], _q[6], _q[7])
                                _hi = cvt_e2m1x8_f32(_q[8], _q[9], _q[10], _q[11], _q[12], _q[13], _q[14], _q[15])
                                _packed64 = cutlass.Uint64(_lo) | (cutlass.Uint64(_hi) << 32)
                                cute.make_tensor(cute.recast_ptr(quant_output.iterator, dtype=cutlass.Uint64), cute.make_layout(N * M // 16))[_token * (M // 16) + _column // 16] = _packed64
                                _qr = (_token - group_begin).to(cutlass.Uint32)
                                _qc = (_column // 16).to(cutlass.Uint32)
                                _cols = (M // 16).to(cutlass.Uint32)
                                _sfidx = (start_sf_block_n.to(cutlass.Uint64) * 128 + _qr // 128 * 128) * _cols + (_qc // 4) * 512 + (_qr % 32) * 16 + ((_qr % 128) // 32) * 4 + _qc % 4
                                cute.make_tensor(cute.recast_ptr(quant_scales.iterator, dtype=cutlass.Uint8), cute.make_layout(quant_scales.shape[0]))[_sfidx] = _sf_bits.to(cutlass.Uint8)
                        nvvm.barrier_cta_sync(barrier_id=EPI_SYNC_BAR_ID, thread_count=num_epilogue_warps * 32)
"""


def make_source(kernel, root, *, omit_output_descriptor=False):
    s = kernel.source_path.read_text()
    assert "tma_c_m_major = (True,)" in s and "start_sf_block_n" in s
    assert "coord_n_c = group_begin +" in s
    assert s.count("    c_0: cute.Tensor,\n") == 1
    extra = "    quant_output: cute.Tensor,\n    quant_scales: cute.Tensor,\n    quant_global: cute.Tensor,\n"
    s = s.replace("    c_0: cute.Tensor,\n", "    c_0: cute.Tensor,\n" + extra, 1)
    needle = "    tma_c_desc_0: cutlass.GridConstant[_tma.TensorMap],\n"
    assert s.count(needle) == 1
    s = s.replace(needle, needle + extra, 1)
    s = s.replace(
        "        tma_c_desc_list[0],\n    ).launch(",
        "        tma_c_desc_list[0],\n        quant_output, quant_scales, quant_global,\n    ).launch(",
        1,
    )
    s = s.replace(
        "from typing import Callable",
        "from typing import Callable\nfrom flashinfer.cute_dsl.fp4_common import cvt_e2m1x8_f32, cvt_f32_to_e4m3, cvt_e4m3_to_f32_via_f16, rcp_approx_ftz",
        1,
    )
    s = s.replace(
        "        # @@EPILOGUE_SETUP:END@@",
        "        _quant_global_pre = quant_global.iterator.raw_ptr().load(is_invariant=True)\n        # @@EPILOGUE_SETUP:END@@",
        1,
    )
    a = s.index("                        epi_stage_idx = (epi_stage_idx + 1)")
    sync = "                        nvvm.barrier_cta_sync(barrier_id=EPI_SYNC_BAR_ID, thread_count=num_epilogue_warps * 32)"
    a = s.index(sync, a) + len(sync)
    b = s.index("                # The M-major TMA path", a)
    s = s[:a] + BODY + "\n\n" + s[b:]
    needle = "    _fake_stream = make_fake_stream"
    s = s.replace(
        needle,
        "    fake_quant_output = cute.runtime.make_fake_tensor(cutlass.Uint32, (sym_n, sym_m // 8, 1), stride=(sym_m // 8, 1, cute.sym_int64()), assumed_align=16)\n    fake_quant_scales = cute.runtime.make_fake_tensor(cutlass.Float8E4M3FN, (cute.sym_int64(), 1, 1), stride=(1, 1, 1), assumed_align=16)\n    fake_quant_global = cute.runtime.make_fake_tensor(cutlass.Float32, (1, 1, 1), stride=(1, 1, 1), assumed_align=4)\n"
        + needle,
        1,
    )
    assert s.count("fake_scale, fake_c_0, stream=") == 1
    s = s.replace(
        "fake_scale, fake_c_0, stream=",
        "fake_scale, fake_c_0, fake_quant_output, fake_quant_scales, fake_quant_global, stream=",
        1,
    )
    if omit_output_descriptor:
        # No global BF16 TMA output remains in this bounded fused epilogue.
        # Keep descriptor arguments/workspace ABI; omit their unused updates.
        setup = s.index("        # @@EPILOGUE_SETUP:BEGIN@@")
        a = s.index("        epi_block_linear = ", setup)
        b = s.index("        while is_valid != 0:", a)
        s = s[:a] + s[b:]
        a = s.index(
            "                if warp_idx == 0 and cutlass.const_expr(not moe_aligned_offsets):",
            setup,
        )
        b = s.index("                # @@EPILOGUE_DRAIN:BEGIN@@", a)
        assert "_replace_tensormap_global_dim_" in s[a:b]
        s = s[:a] + s[b:]
    ast.parse(s)
    digest = hashlib.sha256(s.encode()).hexdigest()
    path = root / "swap_quant_sources" / (digest + ".py")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".py.tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(s)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    return path, digest
