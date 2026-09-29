# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Bounded prepared NVFP4 output source strategy."""

import ast
import hashlib
import os
import tempfile


def _transform(s):
    if "epi_n = 32\n" not in s or "epi_chunk_elems = 16\n" not in s:
        raise ValueError("unsupported quantized output source geometry")
    before = "                        if row_active and row < group_end:\n                            for j in cutlass.range_constexpr(subtile_w // vsize):"
    after = "                        if row_active and row < group_end:\n                            _packed_subtile = cute.make_rmem_tensor(subtile_w // 2, cutlass.Int8)\n                            for j in cutlass.range_constexpr(subtile_w // vsize):"
    assert s.count(before) == 1
    s = s.replace(before, after)
    before = "                                    (gC_tap_0_ptr + (row * out_stride_m_0 + (col_j >> 1))).store(_tap_0, alignment=VEC_BYTES_TAP_0)"
    after = "                                    for _z in cutlass.range_constexpr(8):\n                                        _packed_subtile[j * 8 + _z] = _tap_0[_z]\n                            if col + subtile_w <= N:\n                                (gC_tap_0_ptr + (row * out_stride_m_0 + (col >> 1))).store(_packed_subtile.load().to_vector(), alignment=16)\n"
    assert s.count(before) == 1
    s = s.replace(before, after)
    needle = "                            _packed_subtile = cute.make_rmem_tensor(subtile_w // 2, cutlass.Int8)"
    s = s.replace(
        needle,
        needle
        + "\n                            _scale_subtile = cute.make_rmem_tensor(subtile_w // 16, cutlass.Float8E4M3FN)",
    )
    needle = "                                    (gC_tap_1_ptr + _q0_sidx0).store(_q0_scale0, alignment=1)"
    assert s.count(needle) == 1
    s = s.replace(
        needle, "                                    _scale_subtile[j] = _q0_scale0"
    )
    needle = "                            if col + subtile_w <= N:"
    assert s.count(needle) == 1
    s = s.replace(
        needle,
        needle
        + "\n                                _row_q = row - group_begin\n                                _ncb_q = ((N // 16) + 3) // 4\n                                _col_q = col // 16\n                                _sf_q = start_sf_block_m * _ncb_q * 512 + ((_row_q // 128) * _ncb_q + (_col_q // 4)) * 512 + (_row_q % 32) * 16 + ((_row_q % 128) // 32) * 4 + (_col_q % 4)\n                                (gC_tap_1_ptr + _sf_q).store(_scale_subtile.load().to_vector(), alignment=2)",
    )
    return s


def make_source(kernel, root):
    source = _transform(kernel.source_path.read_text())
    ast.parse(source)
    digest = hashlib.sha256(source.encode()).hexdigest()
    path = root / "output_store_sources" / (digest + ".py")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".py.tmp")
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(source)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return path, digest
