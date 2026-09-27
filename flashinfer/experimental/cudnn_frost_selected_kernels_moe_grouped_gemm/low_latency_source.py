# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
import ast
import hashlib
import re
import os
import tempfile


def make_source(kernel, root, name):
    s = kernel.source_path.read_text()
    if "static" in name:
        a = s.index("            # Dynamic tile assignment:")
        z = s.index("            group_begin = cached_next_begin", a)
        s = (
            s[:a]
            + """            linear_idx = static_linear_idx
            static_linear_idx += gridx // cluster_m

"""
            + s[z:]
        )
        needle = "        while is_tile_valid != 0:"
        assert s.count(needle) == 1
        s = s.replace(
            needle,
            "        static_linear_idx = cutlass.Int32(bidx // cluster_m)\n" + needle,
            1,
        )
        a = s.index(
            "        # Every multi-CTA cluster emits one invalid scheduler record"
        )
        z = s.index("    if warp_idx == tma_warp_id:", a)
        s = s[:a] + s[z:]
        s, n = re.subn(
            r"^    _dynamic_scheduler_counter_initialization\([^\n]+\n",
            "",
            s,
            flags=re.M,
        )
        assert n == 1
        if "general" not in name:
            a = s.index("                cluster_tile_m, coord_n = _moe_swizzle_tile(")
            z = s.index("                coord_expert =", a)
            mapping = (
                "                cluster_tile_m = local_linear_idx\n                coord_n = cutlass.Int32(0)\n"
                if kernel.swap_ab
                else "                cluster_tile_m = cutlass.Int32(0)\n                coord_n = local_linear_idx\n"
            )
            s = s[:a] + mapping + s[z:]
    if "absolute" in name:
        a = s.index("        previous_group_begin = cutlass.Int32(-1)")
        z = s.index("                for k_tile_idx in range(num_k_tiles):", a)
        # Only the input descriptor uses absolute coordinates. In particular,
        # unaligned TMA output still needs its expert-boundary descriptor update.
        s = s[:a] + s[a:z].replace("moe_aligned_offsets", "True") + s[z:]
        # Input data coordinates are global; microscale rows retain their
        # separately padded expert-local address space.
        if kernel.swap_ab:
            needle = "                sfb_n_block = coord_n_desc // 128"
            assert s.count(needle) == 1
            s = s.replace(
                needle,
                "                sfb_n_block = start_sf_block_n + coord_n_group // 128",
            )
            needle = "    for _sfb_op in _sfb_operands:\n"
            assert s.count(needle) == 1
            s = s.replace(
                needle, needle + "        rest_n = _sfb_op.shape[0] // (512 * rest_k)\n"
            )
        else:
            needle = "                sfa_m_block = coord_m_desc // 128"
            assert s.count(needle) == 1
            s = s.replace(
                needle,
                "                sfa_m_block = start_sf_block_m + coord_m_group // 128",
            )
            needle = "    for _sfa_op in _sfa_operands:\n"
            assert s.count(needle) == 1
            s = s.replace(
                needle, needle + "        rest_m = _sfa_op.shape[0] // (512 * rest_k)\n"
            )
    if "plain" in name:
        functions = [
            node
            for node in ast.parse(s).body
            if isinstance(node, ast.FunctionDef) and node.name == "moe_swizzle_tile"
        ]
        if len(functions) != 1:
            raise ValueError("prepared Frost profile requires exactly one tile mapper")
        function = functions[0]
        lines = s.splitlines(keepends=True)
        lines[function.body[0].lineno - 1 : function.end_lineno] = [
            "    return t // nt_n, t % nt_n\n"
        ]
        s = "".join(lines)
    digest = hashlib.sha256(s.encode()).hexdigest()
    p = root / "low_latency_sources" / (digest + ".py")
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        fd, temporary = tempfile.mkstemp(dir=p.parent, suffix=".py.tmp")
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(s)
            os.replace(temporary, p)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return p, digest
