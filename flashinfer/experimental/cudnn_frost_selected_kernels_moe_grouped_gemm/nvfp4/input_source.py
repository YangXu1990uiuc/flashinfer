# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Bounded prepared NVFP4 FC1 input fusion; private absorption prototype."""

import ast
import hashlib
import os
import re
import tempfile


def _gather_activation(source):
    s = source
    assert (
        "epi_chunk_elems = 16" in s
        and "swap_ab" not in s[s.index("@cute.kernel") : s.index("@cute.kernel") + 250]
    )
    assert s.count("    scale: cute.Tensor,\n") == 2
    s = s.replace(
        "    scale: cute.Tensor,\n",
        "    scale: cute.Tensor,\n    input_rows: cute.Tensor,\n    input_tokens: cutlass.Int32,\n",
    )
    assert s.count("        scale,\n") == 1
    s = s.replace(
        "        scale,\n", "        scale,\n        input_rows, input_tokens,\n"
    )
    s = s.replace(
        "fake_up_alpha, fake_scale,",
        "fake_up_alpha, fake_scale, fake_input_rows, cutlass.Int32(1),",
    )
    s = s.replace(
        "    _fake_stream = make_fake_stream",
        "    fake_input_rows = cute.runtime.make_fake_tensor(cutlass.Int32, (sym_m,), stride=(1,), assumed_align=4)\n    _fake_stream = make_fake_stream",
    )
    old = """                global_dims=[k_sym, m, 1],
                global_strides=[
                    a_stride_m * a_dtype.width // 128,
                    a_stride_l * a_dtype.width // 128,
                ],
                box_dims=[cta_tile_mnk[2], a_tma_box_m, 1],"""
    new = """                global_dims=[k_sym, input_tokens],
                global_strides=[a_stride_m * a_dtype.width // 128],
                box_dims=[cta_tile_mnk[2], 1],"""
    assert s.count(old) == 1
    s = s.replace(old, new)
    s = s.replace(
        "a_desc_load_list = a_desc_tma_ptr_list",
        "a_desc_load_list = [tma_a_descs[_ai].get_ptr() for _ai in range(num_a_operands)]",
    )
    a = s.index("                    if a_data_issue:\n")
    z = s.index("                    if b_data_issue:\n", a)
    body = """                    if a_data_issue:
                        for _ai in cutlass.range_constexpr(num_a_operands):
                            for _am in cutlass.range_constexpr(cta_tile_mnk[0] // a_mcast_slices // a_tma_box_m):
                                if lane < a_tma_box_m // 4:
                                    _grouped_row = group_begin + coord_m_group + _a_off + _am * a_tma_box_m + lane * 4
                                    _ir0 = cutlass.Int32(-1)
                                    _ir1 = cutlass.Int32(-1)
                                    _ir2 = cutlass.Int32(-1)
                                    _ir3 = cutlass.Int32(-1)
                                    if _grouped_row < group_end:
                                        _ir0 = (input_rows.iterator.raw_ptr() + _grouped_row).load()
                                    if _grouped_row + 1 < group_end:
                                        _ir1 = (input_rows.iterator.raw_ptr() + _grouped_row + 1).load()
                                    if _grouped_row + 2 < group_end:
                                        _ir2 = (input_rows.iterator.raw_ptr() + _grouped_row + 2).load()
                                    if _grouped_row + 3 < group_end:
                                        _ir3 = (input_rows.iterator.raw_ptr() + _grouped_row + 3).load()
                                    _gather_four(
                                        smem_a_list[_ai].data_ptr(sA_elems * stage + (_a_off + _am * a_tma_box_m + lane * 4) * a_packed_per_row),
                                        a_desc_load_list[_ai], coord_k, _ir0, _ir1, _ir2, _ir3,
                                        ab_full_mbar_ptr.data_ptr(stage), tma_mcast_mask_a)
"""
    s = s[:a] + body + s[z:]
    helper = """@cute.jit
def _gather_four(smem, desc, column, r0, r1, r2, r3, mbar, mask):
    # Same CTA-pair completion-address rule as the native NVVM TMA wrapper
    # and FlashInfer CuTeDSL Rubin inline_ptx.sm100_tma_gather4_load.
    _mbar_addr = mbar.toint(dtype=cutlass.Int32)
    if cutlass.const_expr(cta_group == 2):
        _mbar_addr = _mbar_addr & cutlass.Int32(-16777217)
    llvm.inline_asm(
        None,
        [smem.toint(dtype=cutlass.Int32).ir_value(), desc.toint(dtype=cutlass.Int64).ir_value(),
         cutlass.Int32(column).ir_value(), r0.ir_value(), r1.ir_value(), r2.ir_value(), r3.ir_value(),
         _mbar_addr.ir_value(), cutlass.Uint16(mask).ir_value()],
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::" + str(cta_group) + " [$0], [$1, {$2, $3, $4, $5, $6}], [$7], $8;",
        "r,l,r,r,r,r,r,r,h,~{memory}",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    )

"""
    s = s.replace(
        "@cute.jit\ndef replace_tensormap_global_dim_0",
        helper + "@cute.jit\ndef replace_tensormap_global_dim_0",
        1,
    )
    constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(s).body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in ("cta_tile_mnk", "a_mcast_slices", "a_tma_box_m")
    }
    assert (
        constants["cta_tile_mnk"][0]
        // constants["a_mcast_slices"]
        // constants["a_tma_box_m"]
        == 1
    )
    start = s.index(
        "                                    _grouped_row = group_begin + coord_m_group"
    )
    end = s.index("                                    _gather_four(", start)
    load = s[start:end]
    load = load.replace(
        " + _a_off + _am * a_tma_box_m",
        " + (n_rank * (cta_tile_mnk[0] // a_mcast_slices) if cutlass.const_expr(a_mcast_slices > 1) else 0)",
    )
    load = "\n".join(line[20:] for line in load.splitlines()) + "\n"
    s = s[:start] + s[end:]
    needle = "                for k_tile_idx in range(num_k_tiles):"
    assert s.count(needle) >= 1
    s = s.replace(needle, load + "\n" + needle, 1)
    tree = ast.parse(s)
    spans = []
    producer_start = (
        s[: s.index("        previous_group_begin = cutlass.Int32(-1)")].count("\n") + 1
    )
    producer_end = (
        s[: s.index("                for k_tile_idx in range(num_k_tiles):")].count(
            "\n"
        )
        + 1
    )
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.For)
            and producer_start < node.lineno < producer_end
            and ast.unparse(node.iter) == "cutlass.range_constexpr(num_a_operands)"
        ):
            spans.append((node.lineno - 1, node.end_lineno))
    assert len(spans) == 3, spans
    lines = s.splitlines(keepends=True)
    for a, z in sorted(spans, reverse=True):
        del lines[a:z]
    s = "".join(lines)
    producer, epilogue = 120, 208
    assert s.count("    epi_reg_count = 232") == 1
    s = s.replace("    epi_reg_count = 232", f"    epi_reg_count = {epilogue}", 1)
    a = s.index("    if warp_idx == tma_warp_id:")
    z = s.index("        lane = tidx % 32", a)
    part = s[a:z]
    assert part.count("nvvm.setmaxregister(prod_reg_count,") == 1
    s = (
        s[:a]
        + part.replace(
            "nvvm.setmaxregister(prod_reg_count,", f"nvvm.setmaxregister({producer},"
        )
        + s[z:]
    )
    for constant in [
        "sfa_smem_bytes = 2048",
        "sf_tma_box_k = 4",
        "sfa_tma_box_mn = 1",
        "num_sfa_operands = 1",
    ]:
        assert constant in s, constant
    s = s.replace(
        "    input_tokens: cutlass.Int32,\n",
        "    input_tokens: cutlass.Int32,\n    original_sf: cute.Tensor,\n    original_sf_swizzled: cutlass.Int32,\n",
    )
    s = s.replace(
        "        input_rows, input_tokens,\n",
        "        input_rows, input_tokens, original_sf, original_sf_swizzled,\n",
    )
    s = s.replace(
        "fake_input_rows, cutlass.Int32(1),",
        "fake_input_rows, cutlass.Int32(1), fake_original_sf, cutlass.Int32(1),",
    )
    s = s.replace(
        "    _fake_stream = make_fake_stream",
        "    fake_original_sf = cute.runtime.make_fake_tensor(cutlass.Uint8, (cute.sym_int64(),), stride=(1,), assumed_align=16)\n    _fake_stream = make_fake_stream",
    )
    a = s.index("                    if a_issue:\n")
    z = s.index("                    if b_issue:\n", a)
    s = s[:a] + s[z:]
    preload = """                _sf_source_rows = [cutlass.Int32(-1) for _ in range(4)]
                for _sr in cutlass.range_constexpr(4):
                    _grow = group_begin + coord_m_group + lane * 4 + _sr
                    if _grow < group_end:
                        _sf_source_rows[_sr] = (input_rows.iterator.raw_ptr() + _grow).load()
"""
    needle = "                for k_tile_idx in range(num_k_tiles):"
    s = s.replace(needle, preload + "\n" + needle, 1)
    manual = """                    _original_sf_ptr = cute.recast_ptr(original_sf.iterator, dtype=cutlass.Uint32).raw_ptr()
                    _shared_sf_ptr = cutlass.inttoptr(smem_sfa_list[0].data_ptr(sfa_smem_bytes * stage).toint(dtype=cutlass.Int32), 3, cutlass.Uint32)
                    for _sr in cutlass.range_constexpr(4):
                        _source = _sf_source_rows[_sr]
                        _local_row = lane * 4 + _sr
                        for _sc in cutlass.range_constexpr(4):
                            _value = cutlass.Uint32(0)
                            if _source >= 0:
                                _sfcol = coord_k // 16 + _sc * 4
                                _src = cutlass.Int64(0)
                                if original_sf_swizzled != 0:
                                    _src = (_source.to(cutlass.Int64) // 128) * 128 * (k // 16) + (_sfcol // 4) * 512 + (_source % 32) * 16 + ((_source % 128) // 32) * 4
                                else:
                                    _src = _source.to(cutlass.Int64) * (k // 16) + _sfcol
                                _value = (_original_sf_ptr + _src // 4).load().to(cutlass.Uint32)
                            _dst = _sc * 512 + (_local_row % 32) * 16 + (_local_row // 32) * 4
                            (_shared_sf_ptr + _dst // 4).store(_value)
                    cute.arch.fence_view_async_shared()
                    nvvm.bar_warp_sync(0xFFFFFFFF)
                    if elect_one:
                        if cutlass.const_expr(cta_group == 2):
                            nvvm.mbarrier_complete_tx(nvvm.mapa(sf_full_mbar_ptr.subview(stage), pair_leader_rank), sfa_smem_bytes, scope=nvvm.MemScope.CLUSTER)
                        else:
                            nvvm.mbarrier_complete_tx(sf_full_mbar_ptr.subview(stage), sfa_smem_bytes)
"""
    needle = "                    ab_iter += 1"
    assert s.count(needle) >= 1
    s = s.replace(needle, manual + "\n" + needle, 1)
    spans = []
    for node in ast.walk(ast.parse(s)):
        if isinstance(node, ast.If) and ast.unparse(node.test) in (
            "elect_one and cutlass.const_expr(not moe_aligned_offsets)",
            "group_begin != previous_group_begin and cutlass.const_expr(not moe_aligned_offsets)",
        ):
            spans.append((node.lineno - 1, node.end_lineno))
    assert len(spans) == 2, spans
    lines = s.splitlines(keepends=True)
    for a, z in sorted(spans, reverse=True):
        del lines[a:z]
    s = "".join(lines)
    return s


def _add_scale_producer(source):
    s = source
    assert "    unused_warp_id = 7" in s and "    tma_warp_id = 5" in s
    s = s.replace(
        "    sched_empty_count = 1 + 1 + num_epilogue_warps",
        "    sched_empty_count = 1 + 1 + 1 + num_epilogue_warps",
        1,
    )
    s = s.replace("    epi_reg_count = 208", "    epi_reg_count = 216", 1)
    tma_regs, sf_regs = (56, 56)
    s = s.replace("nvvm.setmaxregister(120,", f"nvvm.setmaxregister({tma_regs},", 1)
    a = s.index("                _sf_source_rows = [")
    z = s.index("                for k_tile_idx in range(num_k_tiles):", a)
    preload = s[a:z]
    s = s[:a] + s[z:]
    a = s.index("                    _original_sf_ptr =")
    z = s.index("                    ab_iter += 1", a)
    manual = s[a:z]
    s = s[:a] + s[z:]
    mapping = {
        "group_begin": "_sf_group_begin",
        "group_end": "_sf_group_end",
        "coord_m_group": "_sf_coord_m_group",
        "lane": "_sf_lane",
        "stage": "_sf_stage",
        "coord_k": "_sf_coord_k",
    }
    for key, val in mapping.items():
        preload = re.sub(r"\b" + key + r"\b", val, preload)
        manual = re.sub(r"\b" + key + r"\b", val, manual)
    block = (
        f"""    if warp_idx == unused_warp_id:
        nvvm.setmaxregister({sf_regs}, nvvm.SetMaxRegisterAction.DECREASE)
        if cutlass.const_expr(USE_PDL):
            nvvm.griddepcontrol("wait")
        _sf_lane = tidx % 32
        _sf_ab_iter = cutlass.Int32(0)
        _sf_empty_phase = cutlass.Int32(1)
        _sf_sched_stage = cutlass.Int32(0)
        _sf_sched_phase = cutlass.Int32(0)
        _sf_valid = cutlass.Int32(1)
        while _sf_valid != 0:
            while not nvvm.mbarrier_try_wait_parity(sched_full_mbar_ptr.subview(_sf_sched_stage), _sf_sched_phase, time_limit=10_000_000):
                pass
            _sf_slot = sched_storage.subview(_sf_sched_stage * SCHED_SLOT_WORDS)
            _sf_tile_m = (_sf_slot.subview(1)).load()
            _sf_valid = (_sf_slot.subview(3)).load()
            _sf_group_begin = (_sf_slot.subview(4)).load()
            _sf_group_end = (_sf_slot.subview(5)).load()
            nvvm.bar_warp_sync(0xFFFFFFFF)
            if elect_one:
                nvvm.mbarrier_arrive(sched_empty_mbar_ptr.subview(_sf_sched_stage))
            _sf_sched_stage += 1
            if _sf_sched_stage == SCHED_STAGES:
                _sf_sched_stage = cutlass.Int32(0)
                _sf_sched_phase = _sf_sched_phase ^ 1
            if _sf_valid != 0:
                _sf_coord_m_group = _sf_tile_m * cgrp_tile_mnk[0] + m_rank * cta_tile_mnk[0]
"""
        + preload
        + """                for _sf_ktile in range(num_k_tiles):
                    _sf_stage = _sf_ab_iter % ab_stages
                    if _sf_stage == 0 and _sf_ab_iter != 0:
                        _sf_empty_phase = _sf_empty_phase ^ 1
                    while not nvvm.mbarrier_try_wait_parity(ab_empty_mbar_ptr.subview(_sf_stage), _sf_empty_phase, time_limit=10_000_000):
                        pass
                    _sf_coord_k = _sf_ktile * cta_tile_mnk[2]
"""
        + manual
        + """                    _sf_ab_iter += 1
"""
    )
    old = "    if warp_idx == unused_warp_id:\n        nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)"
    assert s.count(old) == 1
    s = s.replace(old, block, 1)
    ast.parse(s)
    return s


def _expand_scale_producers(source):
    s = _add_scale_producer(source)
    s = s.replace("threads_per_cta = 256", "threads_per_cta = 384", 1)
    s = s.replace(
        "sched_empty_count = 1 + 1 + 1 + num_epilogue_warps",
        "sched_empty_count = 1 + 1 + 4 + num_epilogue_warps",
        1,
    )
    a = s.index("    if warp_idx == unused_warp_id:")
    z = s.index("\n\n\n", a)
    # The generated branch is the final device section, before set_name_prefix.
    section = s[a:z]
    section = section.replace(
        "if warp_idx == unused_warp_id:",
        "if warp_idx >= unused_warp_id and warp_idx < unused_warp_id + 4:",
        1,
    )
    section = section.replace("for _ in range(4)", "for _ in range(1)")
    section = section.replace(
        "cutlass.range_constexpr(4):\n                    _grow",
        "cutlass.range_constexpr(1):\n                    _grow",
    )
    section = section.replace(
        "cutlass.range_constexpr(4):\n                        _source",
        "cutlass.range_constexpr(1):\n                        _source",
    )
    section = section.replace(
        "_sf_lane * 4 + _sr", "(warp_idx - unused_warp_id) * 32 + _sf_lane"
    )
    section = section.replace(
        ", sfa_smem_bytes, scope=", ", sfa_smem_bytes // 4, scope="
    )
    section = section.replace(", sfa_smem_bytes)", ", sfa_smem_bytes // 4)")
    s = (
        s[:a]
        + section
        + "\n    if warp_idx == unused_warp_id + 4:\n        nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)"
        + s[z:]
    )
    ast.parse(s)
    return s


def _vectorize_scales(source):
    s = _expand_scale_producers(source)
    a = s.index("                    _original_sf_ptr =")
    z = s.index("                    cute.arch.fence_view_async_shared()", a)
    body = """                    _original_sf_ptr = cute.recast_ptr(original_sf.iterator, dtype=cutlass.Uint32).raw_ptr()
                    _shared_sf_ptr = cutlass.inttoptr(smem_sfa_list[0].data_ptr(sfa_smem_bytes * _sf_stage).toint(dtype=cutlass.Int32), 3, cutlass.Uint32)
                    _source = _sf_source_rows[0]
                    _local_row = (warp_idx - unused_warp_id) * 32 + _sf_lane
                    _sv0 = cutlass.Uint32(0)
                    _sv1 = cutlass.Uint32(0)
                    _sv2 = cutlass.Uint32(0)
                    _sv3 = cutlass.Uint32(0)
                    if _source >= 0:
                        _src = _source.to(cutlass.Int64) * (k // 64) + _sf_coord_k // 64
                        _values = cutlass.Array(base=_original_sf_ptr + _src, shape=4, dtype=cutlass.Uint32).load(0, 4, alignment=16)
                        _sv0 = _values[0].to(cutlass.Uint32)
                        _sv1 = _values[1].to(cutlass.Uint32)
                        _sv2 = _values[2].to(cutlass.Uint32)
                        _sv3 = _values[3].to(cutlass.Uint32)
                    _dst = (_local_row % 32) * 4 + _local_row // 32
                    (_shared_sf_ptr + _dst).store(_sv0)
                    (_shared_sf_ptr + _dst + 128).store(_sv1)
                    (_shared_sf_ptr + _dst + 256).store(_sv2)
                    (_shared_sf_ptr + _dst + 384).store(_sv3)
"""
    s = s[:a] + body + s[z:]
    # Preserve the scheduler while avoiding local spills in this bounded path.
    scheduler = s.index("    if warp_idx == scheduler_warp_id:")
    adjustment = s.index("nvvm.setmaxregister(prod_reg_count,", scheduler)
    assert adjustment - scheduler < 200
    s = s[:adjustment] + s[adjustment:].replace(
        "nvvm.setmaxregister(prod_reg_count,", "nvvm.setmaxregister(48,", 1
    )
    ast.parse(s)
    return s


def make_source(kernel, root):
    source = _vectorize_scales(_gather_activation(kernel.source_path.read_text()))
    ast.parse(source)
    digest = hashlib.sha256(source.encode()).hexdigest()
    out = root / "input_fusion_sources" / (digest + ".py")
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
