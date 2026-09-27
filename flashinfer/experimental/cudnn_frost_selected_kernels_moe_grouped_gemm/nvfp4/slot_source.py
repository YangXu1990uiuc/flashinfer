# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Private bounded T1-T3 slot and ordered cluster-output prototype."""

import ast
import hashlib
import re
from pathlib import Path


def _slot_source(kernel, root):
    s = kernel.source_path.read_text()
    assert not kernel.swap_ab
    a = s.index(
        "        cached_next_end = cutlass.Int32(0)",
        s.index("    if warp_idx == scheduler_warp_id:"),
    )
    b = s.index("        static_linear_idx =", a)
    s = s[:a] + s[b:]
    a = s.index("            group_begin = cached_next_begin", a)
    b = s.index(
        "            while not nvvm.mbarrier_try_wait_parity(\n                sched_empty_mbar_ptr",
        a,
    )
    s = s[:a] + """            _slot_index = linear_idx // clusters_along_n
            is_tile_valid = (_slot_index < num_groups).to(cutlass.Int32)
            group_begin = _slot_index
            group_end = _slot_index + 1
            cluster_tile_m = cutlass.Int32(0)
            coord_n = linear_idx % clusters_along_n
            start_sf_block_m = _slot_index
            group_idx = cutlass.Int32(0)
            if is_tile_valid != 0:
                _raw_expert = cutlass.Int32(first_token_arr[_slot_index])
                if _raw_expert >= 0 and _raw_expert < num_experts:
                    group_idx = _raw_expert
            coord_expert = group_idx

""" + s[b:]
    s = s.replace("cutlass.Float32, (sym_g, 1, 1)", "cutlass.Float32, (sym_e, 1, 1)")
    ast.parse(s)
    d = hashlib.sha256(s.encode()).hexdigest()
    p = root / "r27_slots" / f"{d}.py"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        assert p.read_text() == s
    else:
        p.write_text(s)
    return p, d


def _direct_t1(kernel, root, fc1, swizzled):
    p, _ = _slot_source(kernel, root)
    s = p.read_text()
    if fc1:
        assert "cta_group = 1" in s
        assert "sfa_smem_bytes = 2048" in s
        # One input row is real. TMA safely zero-fills the remaining rows.
        assert s.count("global_dims=[k_sym, m, 1]") == 1
        s = s.replace("global_dims=[k_sym, m, 1]", "global_dims=[k_sym, 1, 1]", 1)
        needle = (
            "(coord_k, coord_m_desc + _a_off + _am * a_tma_box_m, cutlass.Int32(0))"
        )
        assert s.count(needle) == 1
        s = s.replace(
            needle,
            "(coord_k, cutlass.Int32(0) + _a_off + _am * a_tma_box_m, cutlass.Int32(0))",
            1,
        )
        # Preserve host/kernel argument order before fused quantization arguments.
        needle = "    quant_output: cute.Tensor,\n"
        assert s.count(needle) == 2
        s = s.replace(needle, "    direct_input_sf: cute.Tensor,\n" + needle)
        needle = "        quant_output, quant_scales, quant_global,\n"
        assert s.count(needle) == 1
        s = s.replace(needle, "        direct_input_sf,\n" + needle, 1)
        needle = "    _fake_stream = make_fake_stream"
        s = s.replace(
            needle,
            "    fake_direct_input_sf = cute.runtime.make_fake_tensor(cutlass.Uint8, (cute.sym_int64(),), stride=(1,), assumed_align=16)\n"
            + needle,
            1,
        )
        assert s.count(", fake_quant_output,") == 1
        s = s.replace(
            ", fake_quant_output,", ", fake_direct_input_sf, fake_quant_output,", 1
        )
        assert s.count("_aux_scale_pre = (_aux_scale_ptr + 0).load()") == 1
        s = s.replace(
            "_aux_scale_pre = (_aux_scale_ptr + 0).load()",
            "_aux_scale_pre = cutlass.Float32(1.0)",
            1,
        )
        a = s.index("                    if a_issue:\n")
        b = s.index("                    if b_issue:\n", a)
        # Each lane owns four contiguous uint32 words of one shared SF column.
        # Only row zero contributes; no tensor element outside the real input is read.
        source = "(_sf_col // 4) * 512" if swizzled else "_sf_col"
        body = """                    _sf_ptr = cute.recast_ptr(direct_input_sf.iterator, dtype=cutlass.Uint32).raw_ptr()
                    _sf_smem = cutlass.inttoptr(smem_sfa_list[0].data_ptr(sfa_smem_bytes * stage).toint(dtype=cutlass.Int32), 3, cutlass.Uint32)
                    _raw = cutlass.Int32(first_token_arr[cutlass.Int32(group_begin)])
                    for _col4 in cutlass.range_constexpr(4):
                        _sf_col = coord_k // 16 + _col4 * 4
                        _sf_val = cutlass.Uint32(0)
                        if lane == 0 and _raw >= 0 and _raw < num_experts:
                            _sf_val = (_sf_ptr + (SOURCE_EXPR) // 4).load().to(cutlass.Uint32)
                        for _row_group in cutlass.range_constexpr(4):
                            _word = cutlass.Uint32(0)
                            if _row_group == 0:
                                _word = _sf_val
                            (_sf_smem + _col4 * 128 + lane * 4 + _row_group).store(_word)
                    cute.arch.fence_view_async_shared()
                    nvvm.bar_warp_sync(0xFFFFFFFF)
                    if elect_one:
                        nvvm.mbarrier_complete_tx(sf_full_mbar_ptr.subview(stage), sfa_smem_bytes)
""".replace(
            "SOURCE_EXPR", source
        )
        s = s[:a] + body + s[b:]
    ast.parse(s)
    d = hashlib.sha256(s.encode()).hexdigest()
    p = root / "r27_direct" / f"{d}.py"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        assert p.read_text() == s
    else:
        p.write_text(s)
    return p, d


def _direct_t2(kernel, root, fc1, swizzled, tokens):
    p, _ = _slot_source(kernel, root)
    s = p.read_text()
    if fc1:
        assert "cta_group = 1" in s
        assert "sfa_smem_bytes = 2048" in s
        # One input row is real. TMA safely zero-fills the remaining rows.
        assert s.count("global_dims=[k_sym, m, 1]") == 1
        s = s.replace(
            "global_dims=[k_sym, m, 1]", f"global_dims=[k_sym, {tokens}, 1]", 1
        )
        needle = (
            "(coord_k, coord_m_desc + _a_off + _am * a_tma_box_m, cutlass.Int32(0))"
        )
        assert s.count(needle) == 1
        s = s.replace(
            needle,
            "(coord_k, cutlass.Int32(group_begin) // 8 + _a_off + _am * a_tma_box_m, cutlass.Int32(0))",
            1,
        )
        # Preserve host/kernel argument order before fused quantization arguments.
        needle = "    quant_output: cute.Tensor,\n"
        assert s.count(needle) == 2
        s = s.replace(needle, "    direct_input_sf: cute.Tensor,\n" + needle)
        needle = "        quant_output, quant_scales, quant_global,\n"
        assert s.count(needle) == 1
        s = s.replace(needle, "        direct_input_sf,\n" + needle, 1)
        needle = "    _fake_stream = make_fake_stream"
        s = s.replace(
            needle,
            "    fake_direct_input_sf = cute.runtime.make_fake_tensor(cutlass.Uint8, (cute.sym_int64(),), stride=(1,), assumed_align=16)\n"
            + needle,
            1,
        )
        assert s.count(", fake_quant_output,") == 1
        s = s.replace(
            ", fake_quant_output,", ", fake_direct_input_sf, fake_quant_output,", 1
        )
        assert s.count("_aux_scale_pre = (_aux_scale_ptr + 0).load()") == 1
        s = s.replace(
            "_aux_scale_pre = (_aux_scale_ptr + 0).load()",
            "_aux_scale_pre = cutlass.Float32(1.0)",
            1,
        )
        a = s.index("                    if a_issue:\n")
        b = s.index("                    if b_issue:\n", a)
        # Each lane owns four contiguous uint32 words of one shared SF column.
        # Only row zero contributes; no tensor element outside the real input is read.
        source = (
            "(_sf_col // 4) * 512 + _input_token * 16"
            if swizzled
            else "_input_token * (k // 16) + _sf_col"
        )
        body = """                    _input_token = cutlass.Int32(group_begin) // 8
                    _sf_ptr = cute.recast_ptr(direct_input_sf.iterator, dtype=cutlass.Uint32).raw_ptr()
                    _sf_smem = cutlass.inttoptr(smem_sfa_list[0].data_ptr(sfa_smem_bytes * stage).toint(dtype=cutlass.Int32), 3, cutlass.Uint32)
                    _raw = cutlass.Int32(first_token_arr[cutlass.Int32(group_begin)])
                    for _col4 in cutlass.range_constexpr(4):
                        _sf_col = coord_k // 16 + _col4 * 4
                        _sf_val = cutlass.Uint32(0)
                        if lane == 0 and _raw >= 0 and _raw < num_experts:
                            _sf_val = (_sf_ptr + (SOURCE_EXPR) // 4).load().to(cutlass.Uint32)
                        for _row_group in cutlass.range_constexpr(4):
                            _word = cutlass.Uint32(0)
                            if _row_group == 0:
                                _word = _sf_val
                            (_sf_smem + _col4 * 128 + lane * 4 + _row_group).store(_word)
                    cute.arch.fence_view_async_shared()
                    nvvm.bar_warp_sync(0xFFFFFFFF)
                    if elect_one:
                        nvvm.mbarrier_complete_tx(sf_full_mbar_ptr.subview(stage), sfa_smem_bytes)
""".replace("SOURCE_EXPR", source)
        s = s[:a] + body + s[b:]
    ast.parse(s)
    d = hashlib.sha256(s.encode()).hexdigest()
    p = root / "r27_direct" / f"{d}.py"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        assert p.read_text() == s
    else:
        p.write_text(s)
    return p, d


def _one_tile(path, root):
    s = path.read_text()
    assert "cta_group = 1" in s
    s = s.replace(
        "    grid_shape = (grid_num_clusters * cluster_m, cluster_n, 1)",
        "    grid_shape = (num_groups * ((n + cgrp_tile_mnk[1] - 1) // cgrp_tile_mnk[1]), 1, 1)",
        1,
    )
    needle = "    first_token_arr = cutlass.make_array_view(first_token_offset)\n"
    assert s.count(needle) == 1
    s = s.replace(
        needle,
        needle + """    _direct_slot = cutlass.Int32(bidx // clusters_along_n)
    _direct_col = cutlass.Int32(bidx % clusters_along_n)
    _direct_expert = cutlass.Int32(first_token_arr[_direct_slot])
    if _direct_expert < 0 or _direct_expert >= num_experts:
        _direct_expert = cutlass.Int32(0)
""",
        1,
    )
    vals = [
        "_direct_expert",
        "cutlass.Int32(0)",
        "_direct_col",
        "cutlass.Int32(1)",
        "_direct_slot",
        "_direct_slot + 1",
        "_direct_slot",
        "_direct_expert",
    ]
    pat = r"\((?:_?slot\.subview\((\d)\)|sched_storage\.subview\(sched_stage \* SCHED_SLOT_WORDS\)\.subview\((\d)\))\)\.load\(\)"
    s, n = re.subn(pat, lambda m: vals[int(m[1] or m[2])], s)
    assert n >= 20, n

    class Edit(ast.NodeTransformer):
        def visit_If(self, node):
            if ast.unparse(node.test) == "warp_idx == scheduler_warp_id":
                node.body = ast.parse(
                    "nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)"
                ).body
                return node
            if (
                len(node.body) == 1
                and isinstance(node.body[0], ast.Expr)
                and "nvvm.mbarrier_arrive(sched_empty_mbar_ptr"
                in ast.unparse(node.body[0])
            ):
                return None
            return self.generic_visit(node)

        def visit_While(self, node):
            test = ast.unparse(node.test)
            if "mbarrier_try_wait_parity" in test and "sched_full_mbar_ptr" in test:
                return None
            node = self.generic_visit(node)
            if test == "is_valid != 0":
                return ast.For(
                    target=ast.Name(id="_single_task", ctx=ast.Store()),
                    iter=ast.parse("cutlass.range_constexpr(1)", mode="eval").body,
                    body=node.body,
                    orelse=[],
                )
            return node

    s = ast.unparse(ast.fix_missing_locations(Edit().visit(ast.parse(s)))) + "\n"
    ast.parse(s)
    d = hashlib.sha256(s.encode()).hexdigest()
    p = root / "r27_one_tile" / f"{d}.py"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        assert p.read_text() == s
    else:
        p.write_text(s)
    return p, d


def _local_ab_completion(source):
    # Each CTA owns an independent GEMM pipeline. Enlarging the physical cluster
    # for final output reduction must not retain the logical 1-CTA completion
    # mask (bit zero), which would signal every pipeline release to rank zero.
    assert "cta_group = 1" in source
    assert "multicast_a = False" in source and "multicast_b = False" in source
    needle = "    _smem_sys_reserved ="
    assert source.count(needle) == 1
    return source.replace(
        needle,
        "    ab_empty_arrive_mask = cutlass.Int16(1) << cta_rank_in_cluster\n" + needle,
        1,
    )


def _cluster_t1(kernel, root, fc1, swizzled):
    p, _ = _direct_t1(kernel, root, fc1, swizzled)
    p, _ = _one_tile(p, root)
    if fc1:
        return p, hashlib.sha256(p.read_bytes()).hexdigest()
    s = _local_ab_completion(p.read_text())
    s = s.replace(
        "_direct_slot = cutlass.Int32(bidx // clusters_along_n)",
        "_direct_slot = cutlass.Int32(bidx % 8)",
        1,
    )
    s = s.replace(
        "_direct_col = cutlass.Int32(bidx % clusters_along_n)",
        "_direct_col = cutlass.Int32(bidx // 8)",
        1,
    )
    s = s.replace(
        "n_rank = cta_rank_in_cluster // cluster_m", "n_rank = cutlass.Int32(0)", 1
    )
    s = s.replace("cluster=cluster_shape_mnk", "cluster=(8, 1, 1)", 1)
    tree = ast.parse(s)
    kern = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name.startswith("frost_")
        and any(ast.unparse(d) == "cute.kernel" for d in n.decorator_list)
    )
    host = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_host"
    )
    for fn in [kern, host]:
        where = len(fn.args.args) if fn is kern else len(fn.args.args) - 1
        fn.args.args[where:where] = [
            ast.arg(
                arg="final_output",
                annotation=ast.parse("cute.Tensor", mode="eval").body,
            ),
            ast.arg(
                arg="route_scores",
                annotation=ast.parse("cute.Tensor", mode="eval").body,
            ),
        ]
    for node in ast.walk(host):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == kern.name
        ):
            node.args.extend(
                [
                    ast.Name(id="final_output", ctx=ast.Load()),
                    ast.Name(id="route_scores", ctx=ast.Load()),
                ]
            )
    # The final cluster barrier follows all native GEMM global stores and TMEM release.
    body = """
llvm.inline_asm(None, [], "fence.acq_rel.gpu; barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;", "~{memory}", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
if _direct_slot == 0:
    _column = _direct_col * cgrp_tile_mnk[1] + tidx
    if _column < N:
        _sum = cutlass.Float32(0)
        for _j in cutlass.range_constexpr(8):
            _id = cutlass.Int32(first_token_arr[cutlass.Int32(_j)])
            if _id >= 0 and _id < num_experts:
                _v = mC_tap_0[_j, _column, 0].to(cutlass.Float32)
                _w = route_scores[0, _j].to(cutlass.Float32)
                _sum = cutlass.Float32(llvm.inline_asm(cutlass.Float32(0).ir_value().type, [_v.ir_value(), _w.ir_value(), _sum.ir_value()], "{ .reg .f32 p; mul.rn.f32 p, $1, $2; add.rn.f32 $0, $3, p; }", "=f,f,f,f", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))
        final_output[0, _column] = _sum.to(cutlass.BFloat16)
"""
    kern.body.extend(ast.parse(body).body)
    s = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    needle = "    _fake_stream = make_fake_stream"
    s = s.replace(
        needle,
        "    fake_final_output = cute.runtime.make_fake_tensor(cutlass.BFloat16, (1, sym_n), stride=(sym_n, 1), assumed_align=16)\n    fake_route_scores = cute.runtime.make_fake_tensor(cutlass.Float32, (1, sym_g), stride=(sym_g, 1), assumed_align=4)\n"
        + needle,
        1,
    )
    needle = "fake_c_tap_0, fake_alpha, stream="
    assert s.count(needle) == 1
    s = s.replace(
        needle,
        "fake_c_tap_0, fake_alpha, fake_final_output, fake_route_scores, stream=",
        1,
    )
    ast.parse(s)
    d = hashlib.sha256(s.encode()).hexdigest()
    p = root / "r27_cluster" / f"{d}.py"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        assert p.read_text() == s
    else:
        p.write_text(s)
    return p, d


def _cluster_t2(kernel, root, fc1, swizzled, tokens):
    p, _ = _direct_t2(kernel, root, fc1, swizzled, tokens)
    p, _ = _one_tile(p, root)
    if fc1:
        return p, hashlib.sha256(p.read_bytes()).hexdigest()
    s = _local_ab_completion(p.read_text())
    s = s.replace(
        "_direct_slot = cutlass.Int32(bidx // clusters_along_n)",
        "_direct_slot = cutlass.Int32((bidx // (8 * clusters_along_n)) * 8 + bidx % 8)",
        1,
    )
    s = s.replace(
        "_direct_col = cutlass.Int32(bidx % clusters_along_n)",
        "_direct_col = cutlass.Int32((bidx // 8) % clusters_along_n)",
        1,
    )
    s = s.replace(
        "n_rank = cta_rank_in_cluster // cluster_m", "n_rank = cutlass.Int32(0)", 1
    )
    s = s.replace("cluster=cluster_shape_mnk", "cluster=(8, 1, 1)", 1)
    tree = ast.parse(s)
    kern = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name.startswith("frost_")
        and any(ast.unparse(d) == "cute.kernel" for d in n.decorator_list)
    )
    host = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_host"
    )
    for fn in [kern, host]:
        where = len(fn.args.args) if fn is kern else len(fn.args.args) - 1
        fn.args.args[where:where] = [
            ast.arg(
                arg="final_output",
                annotation=ast.parse("cute.Tensor", mode="eval").body,
            ),
            ast.arg(
                arg="route_scores",
                annotation=ast.parse("cute.Tensor", mode="eval").body,
            ),
        ]
    for node in ast.walk(host):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == kern.name
        ):
            node.args.extend(
                [
                    ast.Name(id="final_output", ctx=ast.Load()),
                    ast.Name(id="route_scores", ctx=ast.Load()),
                ]
            )
    # The final cluster barrier follows all native GEMM global stores and TMEM release.
    body = """
llvm.inline_asm(None, [], "fence.acq_rel.gpu; barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;", "~{memory}", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
if _direct_slot % 8 == 0:
    _column = _direct_col * cgrp_tile_mnk[1] + tidx
    if _column < N:
        _sum = cutlass.Float32(0)
        for _j in cutlass.range_constexpr(8):
            _id = cutlass.Int32(first_token_arr[cutlass.Int32(_direct_slot + _j)])
            if _id >= 0 and _id < num_experts:
                _v = mC_tap_0[_direct_slot + _j, _column, 0].to(cutlass.Float32)
                _w = route_scores[_direct_slot // 8, _j].to(cutlass.Float32)
                _sum = cutlass.Float32(llvm.inline_asm(cutlass.Float32(0).ir_value().type, [_v.ir_value(), _w.ir_value(), _sum.ir_value()], "{ .reg .f32 p; mul.rn.f32 p, $1, $2; add.rn.f32 $0, $3, p; }", "=f,f,f,f", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))
        final_output[_direct_slot // 8, _column] = _sum.to(cutlass.BFloat16)
"""
    kern.body.extend(ast.parse(body).body)
    s = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    needle = "    _fake_stream = make_fake_stream"
    s = s.replace(
        needle,
        "    fake_final_output = cute.runtime.make_fake_tensor(cutlass.BFloat16, (sym_g // 8, sym_n), stride=(sym_n, 1), assumed_align=16)\n    fake_route_scores = cute.runtime.make_fake_tensor(cutlass.Float32, (sym_g // 8, 8), stride=(8, 1), assumed_align=4)\n"
        + needle,
        1,
    )
    needle = "fake_c_tap_0, fake_alpha, stream="
    assert s.count(needle) == 1
    s = s.replace(
        needle,
        "fake_c_tap_0, fake_alpha, fake_final_output, fake_route_scores, stream=",
        1,
    )
    ast.parse(s)
    d = hashlib.sha256(s.encode()).hexdigest()
    p = root / "r27_cluster" / f"{d}.py"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        assert p.read_text() == s
    else:
        p.write_text(s)
    return p, d


def make_source(kernel, root, *, fc1, swizzled, tokens):
    assert tokens in (1, 2, 3) and not kernel.swap_ab
    if tokens == 1:
        return _cluster_t1(kernel, root, fc1, swizzled)
    return _cluster_t2(kernel, root, fc1, swizzled, tokens)
