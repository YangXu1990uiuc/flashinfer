# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
import ast
import hashlib
import re
import os
import tempfile
import functools
from pathlib import Path
from types import SimpleNamespace

from . import runtime as common


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


def profiles(dtype, tokens, hidden, intermediate, experts, topk, activation):
    """Keep the original execution strategy alongside bounded prepared variants."""
    geometry = (experts, hidden, intermediate, topk)
    primary = geometry in ((64, 2048, 1408, 6), (12, 7168, 3072, 2))
    if activation != "swiglu" or tokens <= 0:
        return (None,)
    if primary and tokens * topk <= 128:
        return (
            (None, "static_absolute_all_v3", "static_absolute_all_v3_swap_quant")
            if dtype == "nvfp4"
            else (None, "static_absolute_all_v3")
        )
    if dtype == "nvfp4":
        if geometry == (128, 2048, 768, 8) and 8192 <= tokens <= 12288:
            return (
                (None, "input_fusion", "input_reuse")
                if tokens >= 9216
                else (None, "input_fusion")
            )
        if tokens <= 512 and (
            primary or geometry in ((128, 2048, 768, 8), (8, 4096, 14336, 2))
        ):
            return (
                (
                    None,
                    "static_general_plain_absolute",
                    "static_general_plain_absolute_swap_quant",
                )
                if tokens <= (512 if experts in (8, 12) else 128)
                else (None, "static_general_plain_absolute")
            )
        if primary and 8192 <= tokens <= 12288:
            return (None, "absolute")
    return (None,)


@functools.cache
def _source_identity():
    return "".join(
        common._digest(Path(__file__).parent / name)
        for name in (
            "prepared.py",
            "nvfp4/input_source.py",
            "nvfp4/reuse_source.py",
            "nvfp4/swap_fused_quant.py",
        )
    )


def plan_key(tag, first, second, profile):
    key = (tag, first.tactic, second.tactic)
    return (
        key
        if profile is None
        else key + ("prepared-v1", profile, common._tactic_digest(_source_identity()))
    )


def load(kernel, device, profile):
    if profile is None or (profile == "input_fusion" and not kernel.fc1):
        return common._load_kernel(kernel, device)
    from ...jit.env import FLASHINFER_GEN_SRC_DIR

    if profile in ("input_fusion", "input_reuse"):
        from .nvfp4.input_source import make_source as input_source
        from .nvfp4.reuse_source import make_source as reuse_source

        root = FLASHINFER_GEN_SRC_DIR / "cudnn_frost_prepared_profiles"
        if kernel.fc1:
            path, digest = input_source(kernel, root)
        if profile == "input_reuse":
            source = SimpleNamespace(source_path=path) if kernel.fc1 else kernel
            path, digest = reuse_source(source, root, "fc1" if kernel.fc1 else "fc2")
        return common._load_source(path, digest, kernel.arch, device.index)

    path, digest = make_source(
        kernel, FLASHINFER_GEN_SRC_DIR / "cudnn_frost_prepared_profiles", profile
    )
    if "swap_quant" in profile and kernel.fc1:
        from .nvfp4.swap_fused_quant import make_source as swap_source

        path, digest = swap_source(
            SimpleNamespace(source_path=path),
            FLASHINFER_GEN_SRC_DIR / "cudnn_frost_prepared_profiles",
            omit_output_descriptor=True,
        )
    return common._load_source(path, digest, kernel.arch, device.index)


def valid_pair(first, second, profile):
    if profile is not None and "swap_quant" in profile:
        return (
            first.swap_ab
            and first.tactic_metadata.get("store_mode") == "tma"
            and first.tactic_metadata.get("cta_tile", {}).get("m") == 128
        )
    if profile not in ("input_fusion", "input_reuse"):
        return True
    if (
        first.swap_ab
        or "output_scale" not in first.launch_tail
        or first.tactic_metadata.get("store_mode") != "stg"
    ):
        return False
    return profile != "input_reuse" or (
        "128x256x128_128x256x64_cluster2x1" in first.artifact_id
        and "256x256x128_128x256x64_cluster2x1" in second.artifact_id
        and second.swap_ab
    )
