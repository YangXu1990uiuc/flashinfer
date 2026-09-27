# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
import hashlib
import os
import tempfile


def _pack_throughput_epilogue(source):
    # The prepared throughput envelope has complete 64-element column groups.
    # Keep the per-block arithmetic and BF16 rounding; combine adjacent stores.
    assert "epi_n = 32" in source and "epi_chunk_elems = 16" in source
    source = source.replace("epi_n = 32", "epi_n = 64", 1)
    needle = "                        if row_active and row < group_end:\n                            for j in cutlass.range_constexpr(subtile_w // vsize):"
    replacement = "                        if row_active and row < group_end:\n                            _qpair = cute.make_rmem_tensor((8,), cutlass.Uint32)\n                            _sf_pair = cutlass.Uint32(0)\n                            for j in cutlass.range_constexpr(subtile_w // vsize):"
    assert source.count(needle) == 1
    source = source.replace(needle, replacement)
    start = source.index(
        "                                    _qp = quant_output.iterator.raw_ptr()"
    )
    tail = "[_sfidx] = _sf_bits.to(cutlass.Uint8)"
    end = source.index(tail, start) + len(tail)
    body = """                                    _qpair[(j % 4) * 2] = _lo
                                    _qpair[(j % 4) * 2 + 1] = _hi
                                    _sf_pair = _sf_pair | (_sf_bits.to(cutlass.Uint32) << ((j % 4) * 8))
                                    if cutlass.const_expr(j % 4 == 3):
                                        _qp = quant_output.iterator.raw_ptr()
                                        (_qp + row * (N // 8) + (col_j - 48) // 8).store(_qpair.load().to(cutlass.Int32), alignment=16)
                                        _qr = (row - group_begin).to(cutlass.Uint32)
                                        _qc = ((col_j - 48) // 16).to(cutlass.Uint32)
                                        _cols = (N // 16).to(cutlass.Uint32)
                                        _sfidx = (start_sf_block_m.to(cutlass.Uint64) * 128 + _qr // 128 * 128) * _cols + (_qc // 4) * 512 + (_qr % 32) * 16 + ((_qr % 128) // 32) * 4 + _qc % 4
                                        cute.make_tensor(cute.recast_ptr(quant_scales.iterator, dtype=cutlass.Uint32), cute.make_layout(quant_scales.shape[0] // 4))[_sfidx // 4] = _sf_pair.to(cutlass.Uint32)"""
    return source[:start] + body + source[end:]


def _bf16_block_maximum(source):
    # Every fused profile has already rounded its activation to BF16. Preserve
    # those values and every subsequent scale/quantization operation.
    helper = '@cute.jit\ndef _frost_bfmax2(a: cutlass.Uint32, b: cutlass.Uint32):\n    return cutlass.Uint32(llvm.inline_asm(\n        a.ir_value().type, [a.ir_value(), b.ir_value()],\n        "max.bf16x2 $0, $1, $2;", "=r,r,r",\n        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))\n\n@cute.jit\ndef _frost_bits_float(a: cutlass.Uint32):\n    return cutlass.Float32(llvm.inline_asm(\n        cutlass.Float32(0).ir_value().type, [a.ir_value()],\n        "mov.b32 $0, $1;", "=f,r",\n        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))\n\n'
    body = "                                    _bits = cutlass.Vector.bitcast(_tap_0, cutlass.Uint32)\n                                    _abs_mask = cutlass.Uint32(0x7fff7fff)\n                                    _bm0 = _frost_bfmax2(_bits[0] & _abs_mask, _bits[1] & _abs_mask)\n                                    _bm1 = _frost_bfmax2(_bits[2] & _abs_mask, _bits[3] & _abs_mask)\n                                    _bm2 = _frost_bfmax2(_bits[4] & _abs_mask, _bits[5] & _abs_mask)\n                                    _bm3 = _frost_bfmax2(_bits[6] & _abs_mask, _bits[7] & _abs_mask)\n                                    _bm = _frost_bfmax2(_frost_bfmax2(_bm0, _bm1), _frost_bfmax2(_bm2, _bm3))\n                                    _bm = _frost_bfmax2(_bm, (_bm >> 16) | (_bm << 16))\n                                    _amax = _frost_bits_float(_bm << 16)\n"
    assert "from cutlass._mlir.dialects import" in source and "llvm" in source
    source = source.replace("@cute.kernel", helper + "@cute.kernel", 1)
    begin = source.index("                                    _maxs = ")
    end = source.index("                                    _global = ", begin)
    return source[:begin] + body + source[end:]


def make_fused_source(kernel, root, packed_epilogue=False):
    source = kernel.source_path.read_text()
    assert "@cute.kernel" in source
    assert "epi_chunk_elems = 16" in source
    assert "swap_ab" not in kernel.source_path.name or not kernel.swap_ab
    source = source.replace(
        "from typing import Callable",
        "from typing import Callable\nfrom flashinfer.cute_dsl.fp4_common import cvt_e2m1x8_f32, cvt_f32_to_e4m3, cvt_e4m3_to_f32_via_f16, rcp_approx_ftz",
        1,
    )
    # Both the device and host function receive explicit packed/scaled outputs.
    source = source.replace(
        "    scale: cute.Tensor,\n",
        "    scale: cute.Tensor,\n    quant_output: cute.Tensor,\n    quant_scales: cute.Tensor,\n    quant_global: cute.Tensor,\n",
    )
    source = source.replace(
        "        scale,\n    ).launch(",
        "        scale,\n        quant_output, quant_scales, quant_global,\n    ).launch(",
        1,
    )
    needle = "                                    (gC_tap_0_ptr + (row * out_stride_m_0 + col_j)).store(_tap_0, alignment=VEC_BYTES_TAP_0)"
    assert needle in source
    body = """                                    _qvals = _tap_0.to(cutlass.Float32)
                                    _maxs = [cute.math.abs(_qvals[_qi]) for _qi in range(16)]
                                    for _qi in cutlass.range_constexpr(8):
                                        _maxs[_qi] = cutlass.max(_maxs[_qi], _maxs[_qi + 8])
                                    for _qi in cutlass.range_constexpr(4):
                                        _maxs[_qi] = cutlass.max(_maxs[_qi], _maxs[_qi + 4])
                                    for _qi in cutlass.range_constexpr(2):
                                        _maxs[_qi] = cutlass.max(_maxs[_qi], _maxs[_qi + 2])
                                    _amax = cutlass.max(_maxs[0], _maxs[1])
                                    _global = quant_global.iterator.raw_ptr().load()
                                    _sf_bits = cvt_f32_to_e4m3(_amax * cutlass.Float32(1.0 / 6.0) * _global)
                                    _sf_f32 = cvt_e4m3_to_f32_via_f16(_sf_bits)
                                    _inv = cutlass.Float32(0.0)
                                    if _amax != cutlass.Float32(0.0):
                                        _inv = _global * rcp_approx_ftz(_sf_f32)
                                    _q = _qvals * cutlass.full_like(_qvals, _inv)
                                    _lo = cvt_e2m1x8_f32(_q[0], _q[1], _q[2], _q[3], _q[4], _q[5], _q[6], _q[7])
                                    _hi = cvt_e2m1x8_f32(_q[8], _q[9], _q[10], _q[11], _q[12], _q[13], _q[14], _q[15])
                                    _qp = quant_output.iterator.raw_ptr()
                                    quant_output[row, col_j // 8, 0] = _lo
                                    quant_output[row, col_j // 8 + 1, 0] = _hi
                                    _qr = row - group_begin
                                    _qc = col_j // 16
                                    _cols = N // 16
                                    _sfidx = (start_sf_block_m * 128 + _qr // 128 * 128) * _cols + (_qc // 4) * 512 + (_qr % 32) * 16 + ((_qr % 128) // 32) * 4 + _qc % 4
                                    cute.make_tensor(cute.recast_ptr(quant_scales.iterator, dtype=cutlass.Uint8), cute.make_layout(quant_scales.shape[0]))[_sfidx] = _sf_bits.to(cutlass.Uint8)"""
    source = source.replace(needle, body)
    needle = "    _fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)"
    source = source.replace(
        needle,
        """    fake_quant_output = cute.runtime.make_fake_tensor(cutlass.Uint32, (sym_m, cute.sym_int64(), 1), stride=(cute.sym_int64(), 1, cute.sym_int64()), assumed_align=16)
    fake_quant_scales = cute.runtime.make_fake_tensor(cutlass.Float8E4M3FN, (cute.sym_int64(), 1, 1), stride=(1, 1, 1), assumed_align=16)
    fake_quant_global = cute.runtime.make_fake_tensor(cutlass.Float32, (1, 1, 1), stride=(1, 1, 1), assumed_align=4)
"""
        + needle,
        1,
    )
    source = source.replace(
        "fake_up_alpha, fake_scale, stream=_fake_stream",
        "fake_up_alpha, fake_scale, fake_quant_output, fake_quant_scales, fake_quant_global, stream=_fake_stream",
    )
    needle = "        # @@EPILOGUE_SETUP:END@@"
    assert source.count(needle) == 1
    source = source.replace(
        needle,
        "        _quant_global_pre = quant_global.iterator.raw_ptr().load(is_invariant=True)\n"
        + needle,
    )
    source = source.replace(
        "_global = quant_global.iterator.raw_ptr().load()",
        "_global = _quant_global_pre",
    )
    source = source.replace(
        "_qr = row - group_begin", "_qr = (row - group_begin).to(cutlass.Uint32)"
    )
    source = source.replace(
        "_qc = col_j // 16", "_qc = (col_j // 16).to(cutlass.Uint32)"
    )
    source = source.replace("_cols = N // 16", "_cols = (N // 16).to(cutlass.Uint32)")
    source = source.replace(
        "(start_sf_block_m * 128 + _qr // 128 * 128)",
        "(start_sf_block_m.to(cutlass.Uint64) * 128 + _qr // 128 * 128)",
    )
    source = source.replace(
        """                                    quant_output[row, col_j // 8, 0] = _lo
                                    quant_output[row, col_j // 8 + 1, 0] = _hi""",
        """                                    _packed64 = cutlass.Uint64(_lo) | (cutlass.Uint64(_hi) << 32)
                                    cute.make_tensor(cute.recast_ptr(quant_output.iterator, dtype=cutlass.Uint64), cute.make_layout(m * N // 16))[row * (N // 16) + col_j // 16] = _packed64""",
    )
    if packed_epilogue:
        source = _pack_throughput_epilogue(source)
    source = _bf16_block_maximum(source)
    digest = hashlib.sha256(source.encode()).hexdigest()
    out = root / "fusion_sources" / (digest + ".py")
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        fd, temporary = tempfile.mkstemp(dir=out.parent, suffix=".py.tmp")
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(source)
            os.replace(temporary, out)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return out, digest
