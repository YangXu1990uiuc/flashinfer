# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Bounded prepared NVFP4 FC1 source strategy."""

import ast
import hashlib
import os
import tempfile

from .output_source import _transform as _output_transform


def _transform(source):
    required = (
        "cta_group = 2\n",
        "cta_tile_mnk = (128, 128, 256)\n",
        "mma_size_m = 1\n",
        "num_gemms = 2\n",
        "acc_stages = 1\n",
        "use_acc_overlap = False\n",
        "epi_cols_per_mma_m = 256\n",
    )
    if any(item not in source for item in required):
        raise ValueError("unsupported NVFP4 FC1 source geometry")
    source = _output_transform(source)
    replacements = (
        ("threads_per_cta = 256\n", "threads_per_cta = 384\n"),
        ("    mma_warp_id = 4\n", "    mma_warp_id = 8\n"),
        ("    tma_warp_id = 5\n", "    tma_warp_id = 9\n"),
        ("    scheduler_warp_id = 6\n", "    scheduler_warp_id = 10\n"),
        ("    unused_warp_id = 7\n", "    unused_warp_id = 11\n"),
        ("    num_epilogue_warps = 4\n", "    num_epilogue_warps = 8\n"),
        (
            "row_id_with_warp_offset = base_row_id + warp_idx * 32",
            "row_id_with_warp_offset = base_row_id + (warp_idx % 4) * 32",
        ),
        (
            "epi_spans = _epi_subtile_spans(epi_cols_per_mma_m, epi_n)",
            "epi_spans = _epi_subtile_spans(epi_cols_per_mma_m // 2, epi_n)",
        ),
        (
            "                    row = coord_m + tidx\n",
            "                    row = coord_m + tidx % 128\n",
        ),
        (
            "                        c_rmem_vecs = []",
            "                        subtile_col_offset += (warp_idx // 4) * (epi_cols_per_mma_m // 2)\n                        c_rmem_vecs = []",
        ),
        ("epi_n = 32\n", "epi_n = 64\n"),
        (
            "(gC_tap_1_ptr + _sf_q).store(_scale_subtile.load().to_vector(), alignment=2)",
            "(gC_tap_1_ptr + _sf_q).store(_scale_subtile.load().to_vector(), alignment=4)",
        ),
    )
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError("unsupported NVFP4 FC1 source structure")
        source = source.replace(before, after)
    return source


def make_source(kernel, root):
    source = _transform(kernel.source_path.read_text())
    ast.parse(source)
    digest = hashlib.sha256(source.encode()).hexdigest()
    path = root / "fc1_epilogue_sources" / (digest + ".py")
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
