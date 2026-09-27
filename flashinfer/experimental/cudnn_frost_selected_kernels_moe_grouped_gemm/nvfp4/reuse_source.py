# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Bounded prepared NVFP4 input reuse and task grouping; private absorption POC."""

import ast
import hashlib
import os
import re
import tempfile


def _fc1(code):
    assert "input_rows: cute.Tensor" in code and "fwd_128x256x128" in code
    assert re.findall("^ab_stages = (\\d+)$", code, re.M) == ["5"]
    code = code.replace("ab_stages = 5", "ab_stages = 4", 1)
    first = code.index("                group_nt_m = total_tiles // clusters_along_n")
    last = code.index("                coord_expert = group_idx % num_experts", first)
    code = (
        code[:first]
        + "                cluster_tile_m = local_linear_idx // clusters_along_n\n                coord_n = local_linear_idx % clusters_along_n\n"
        + code[last:]
    )
    needle = "        linear_idx = cutlass.Int32(0)\n        start_linear_idx = cutlass.Int32(0)"
    assert code.count(needle) == 1
    code = code.replace(
        needle,
        "        owned_row_base = cutlass.Int32(0)\n        owned_column = cutlass.Int32(0)\n"
        + needle,
    )
    first = code.index(
        "                if lane == 0:\n                    claimed = nvvm.atomicrmw("
    )
    last = code.index("                claimed = nvvm.shfl_sync(", first)
    replacement = '                if lane == 0:\n                    if owned_column == 0:\n                        owned_row_base = nvvm.atomicrmw(\n                            "add", sched_counter_ptr, clusters_along_n,\n                            mem_order="relaxed", syncscope="gpu",\n                        )\n                    claimed = owned_row_base + owned_column\n                    owned_column += 1\n                    if owned_column == clusters_along_n:\n                        owned_column = cutlass.Int32(0)\n'
    code = code[:first] + replacement + code[last:]

    def rep(a, b, count=1):
        nonlocal code
        assert code.count(a) == count, (a, code.count(a), count)
        code = code.replace(a, b)

    rep("sA_elems * ab_stages,", "sA_elems * 8,")
    rep("if a_data_issue:", "if a_data_issue and tile_n == 0:")
    rep("sA_elems * stage +", "sA_elems * k_tile_idx +")
    rep("sA_bytes * stage)", "sA_bytes * k_tile_idx)", 2)
    rep(
        "ab_only_copy_bytes)",
        "(((num_a_operands * sA_tma_bytes if tile_n == 0 else 0) + num_b_operands * sB_tma_bytes) * cta_group))",
        2,
    )
    rep("sfa_smem_bytes * ab_stages,", "sfa_smem_bytes * 8,")
    rep("sfa_smem_bytes * stage)", "sfa_smem_bytes * k_tile_idx)", 2)
    rep(
        "            _sf_tile_m = (_sf_slot.subview(1)).load()",
        "            _sf_tile_m = (_sf_slot.subview(1)).load()\n            _sf_tile_n = (_sf_slot.subview(2)).load()",
    )
    rep("if _grow < _sf_group_end:", "if _sf_tile_n == 0 and _grow < _sf_group_end:")
    rep(
        "smem_sfa_list[0].data_ptr(sfa_smem_bytes * _sf_stage)",
        "smem_sfa_list[0].data_ptr(sfa_smem_bytes * _sf_ktile)",
    )
    first = code.index("                    _sf_coord_k = _sf_ktile * cta_tile_mnk[2]")
    last = code.index("                    nvvm.bar_warp_sync(0xFFFFFFFF)", first)
    block = code[first:last]
    code = (
        code[:first]
        + "                    if _sf_tile_n == 0:\n"
        + "".join(("    " + line + "\n" for line in block.splitlines()))
        + code[last:]
    )
    ast.parse(code)
    return code


def _fc2_row(code):
    assert "fwd_swap_ab_256x256x128" in code
    assert re.findall("^ab_stages = (\\d+)$", code, re.M) == ["5"]
    first = code.index("                group_nt_n = total_tiles // clusters_along_m")
    last = code.index("                coord_expert = group_idx % num_experts", first)
    code = (
        code[:first]
        + "                cluster_tile_m = local_linear_idx % clusters_along_m\n                coord_n = local_linear_idx // clusters_along_m\n"
        + code[last:]
    )
    needle = "        linear_idx = cutlass.Int32(0)\n        start_linear_idx = cutlass.Int32(0)"
    assert code.count(needle) == 1
    code = code.replace(
        needle,
        "        owned_row_base = cutlass.Int32(0)\n        owned_column = cutlass.Int32(0)\n"
        + needle,
    )
    first = code.index(
        "                if lane == 0:\n                    claimed = nvvm.atomicrmw("
    )
    last = code.index("                claimed = nvvm.shfl_sync(", first)
    code = (
        code[:first]
        + '                if lane == 0:\n                    if owned_column == 0:\n                        owned_row_base = nvvm.atomicrmw(\n                            "add", sched_counter_ptr, clusters_along_m,\n                            mem_order="relaxed", syncscope="gpu",\n                        )\n                    claimed = owned_row_base + owned_column\n                    owned_column += 1\n                    if owned_column == clusters_along_m:\n                        owned_column = cutlass.Int32(0)\n'
        + code[last:]
    )
    return code


def _fc2(source):
    source = _fc2_row(source)
    old = '"add", sched_counter_ptr, clusters_along_m,'
    assert source.count(old) == 1
    source = source.replace(old, '"add", sched_counter_ptr, cutlass.Int32(2),')
    old = "if owned_column == clusters_along_m:"
    assert source.count(old) == 1
    source = source.replace(old, "if owned_column == 2:")
    ast.parse(source)
    return source


def make_source(kernel, root, stage):
    assert stage in ("fc1", "fc2")
    source = (_fc1 if stage == "fc1" else _fc2)(kernel.source_path.read_text())
    ast.parse(source)
    digest = hashlib.sha256(source.encode()).hexdigest()
    out = root / "throughput_reuse_sources" / (digest + ".py")
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
