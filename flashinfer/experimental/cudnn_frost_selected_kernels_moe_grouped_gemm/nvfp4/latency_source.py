# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Private bounded E128 decode experiments, prepared outside the run path."""
import ast
import hashlib
from pathlib import Path
from . import fused_quant

def normal_tma(kernel, root, omit):
    s=kernel.source_path.read_text()
    assert 'tma_c_m_major = (False,)' in s
    original=s
    extra='    quant_output: cute.Tensor,\n    quant_scales: cute.Tensor,\n    quant_global: cute.Tensor,\n'
    for needle in ('    c_0: cute.Tensor,\n','    tma_c_desc_0: cutlass.GridConstant[_tma.TensorMap],\n'):
        assert s.count(needle)==1
        s=s.replace(needle,needle+extra)
    needle='        tma_c_desc_list[0],\n    ).launch('
    assert s.count(needle)==1
    s=s.replace(needle,'        tma_c_desc_list[0],\n        quant_output, quant_scales, quant_global,\n    ).launch(')
    needle='from typing import Callable'
    s=s.replace(needle,needle+'\nfrom flashinfer.cute_dsl.fp4_common import cvt_e2m1x8_f32, cvt_f32_to_e4m3, cvt_e4m3_to_f32_via_f16, rcp_approx_ftz',1)
    s=s.replace('        # @@EPILOGUE_SETUP:END@@','        _quant_global_pre = quant_global.iterator.raw_ptr().load(is_invariant=True)\n        # @@EPILOGUE_SETUP:END@@',1)
    tree=ast.parse(Path(fused_quant.__file__).read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_fused_source')
    body=next(ast.literal_eval(n.value) for n in fn.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='body' for t in n.targets))
    body=body.replace('_global = quant_global.iterator.raw_ptr().load()', '_global = _quant_global_pre')
    body=body.replace('_qr = row - group_begin','_qr = (row - group_begin).to(cutlass.Uint32)').replace('_qc = col_j // 16','_qc = (col_j // 16).to(cutlass.Uint32)').replace('_cols = N // 16','_cols = (N // 16).to(cutlass.Uint32)')
    body=body.replace('(start_sf_block_m * 128 + _qr // 128 * 128)','(start_sf_block_m.to(cutlass.Uint64) * 128 + _qr // 128 * 128)')
    body=body.replace('                                    quant_output[row, col_j // 8, 0] = _lo\n                                    quant_output[row, col_j // 8 + 1, 0] = _hi','                                    _packed64 = cutlass.Uint64(_lo) | (cutlass.Uint64(_hi) << 32)\n                                    cute.make_tensor(cute.recast_ptr(quant_output.iterator, dtype=cutlass.Uint64), cute.make_layout(m * N // 16))[row * (N // 16) + col_j // 16] = _packed64')
    a=s.index('                        epi_stage_idx = (epi_stage_idx + 1)')
    b=s.index('                # The M-major TMA path',a)
    replacement='''                        if row_active and row < group_end:
                            for _qj in cutlass.range_constexpr(epi_n // 16):
                                col_j = col + _qj * 16
                                if col_j < N:
                                    _tap_0 = vec_out[_qj * 16 : _qj * 16 + 16]
'''+body+'\n\n'
    s=s[:a]+replacement+s[b:]
    s=fused_quant._bf16_block_maximum(s)
    needle='    _fake_stream = make_fake_stream'
    s=s.replace(needle,'    fake_quant_output = cute.runtime.make_fake_tensor(cutlass.Uint32, (sym_m, sym_n // 8, 1), stride=(sym_n // 8, 1, cute.sym_int64()), assumed_align=16)\n    fake_quant_scales = cute.runtime.make_fake_tensor(cutlass.Float8E4M3FN, (cute.sym_int64(), 1, 1), stride=(1, 1, 1), assumed_align=16)\n    fake_quant_global = cute.runtime.make_fake_tensor(cutlass.Float32, (1, 1, 1), stride=(1, 1, 1), assumed_align=4)\n'+needle,1)
    assert s.count('fake_scale, fake_c_0, stream=')==1
    s=s.replace('fake_scale, fake_c_0, stream=','fake_scale, fake_c_0, fake_quant_output, fake_quant_scales, fake_quant_global, stream=',1)
    if omit:
        setup=s.index('        # @@EPILOGUE_SETUP:BEGIN@@')
        a=s.index('        epi_block_linear = ',setup); b=s.index('        while is_valid != 0:',a)
        s=s[:a]+s[b:]
        a=s.index('                if warp_idx == 0 and cutlass.const_expr(not moe_aligned_offsets):',setup)
        b=s.index('                # @@EPILOGUE_DRAIN:BEGIN@@',a)
        assert '_replace_tensormap_global_dim_' in s[a:b]
        s=s[:a]+s[b:]
    ast.parse(s)
    digest=hashlib.sha256(s.encode()).hexdigest()
    out=root/'r26_normal_tma_sources'/f'{digest}.py';out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():assert out.read_text()==s
    else:out.write_text(s)
    return out,digest

def sparse_source(kernel, root):
    path,digest=kernel.source_path,hashlib.sha256(kernel.source_path.read_bytes()).hexdigest()
    if kernel.swap_ab:return path,digest
    s=path.read_text()
    constants={}
    for n in ast.parse(s).body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            try:constants[n.targets[0].id]=ast.literal_eval(n.value)
            except (ValueError,TypeError):pass
    assert constants['cgrp_tile_mnk'][0]>=128,constants['cgrp_tile_mnk']
    assert 'static_linear_idx' in s
    a=s.index('        cached_next_end = cutlass.Int32(0)',s.index('    if warp_idx == scheduler_warp_id:'))
    b=s.index('        static_linear_idx =',a)
    s=s[:a]+'''        _active_masks = []
        _active_counts = []
        _total_active = cutlass.Int32(0)
        for _batch in cutlass.range_constexpr(4):
            _expert = _batch * 32 + lane
            _begin = cutlass.Int32(first_token_arr[_expert])
            _end = gemm_s
            if _expert + 1 < 128:
                _end = cutlass.Int32(first_token_arr[_expert + 1])
            _mask = nvvm.vote_sync(full_warp_mask, _end > _begin, nvvm.VoteSync.BALLOT)
            _count = cute.arch.popc(_mask).to(cutlass.Int32)
            _active_masks.append(_mask)
            _active_counts.append(_count)
            _total_active += _count

'''+s[b:]
    a=s.index('            group_begin = cached_next_begin',a)
    b=s.index('            while not nvvm.mbarrier_try_wait_parity(\n                sched_empty_mbar_ptr',a)
    s=s[:a]+'''            _active_index = linear_idx // clusters_along_n
            is_tile_valid = (_active_index < _total_active).to(cutlass.Int32)
            group_begin = cutlass.Int32(0)
            group_end = cutlass.Int32(0)
            group_idx = cutlass.Int32(0)
            coord_expert = cutlass.Int32(0)
            cluster_tile_m = cutlass.Int32(0)
            coord_n = linear_idx % clusters_along_n
            start_sf_block_m = _active_index
            if is_tile_valid != 0:
                _prefix = cutlass.Int32(0)
                _chosen = cutlass.Uint32(0)
                _nth = cutlass.Int32(0)
                _base = cutlass.Int32(0)
                for _batch in cutlass.range_constexpr(4):
                    if _active_index >= _prefix and _active_index < _prefix + _active_counts[_batch]:
                        _chosen = _active_masks[_batch].to(cutlass.Uint32)
                        _nth = _active_index - _prefix
                        _base = cutlass.Int32(_batch * 32)
                    _prefix += _active_counts[_batch]
                for _shift in (16, 8, 4, 2, 1):
                    _low_count = cute.arch.popc(_chosen & cutlass.Uint32((1 << _shift) - 1)).to(cutlass.Int32)
                    if _nth >= _low_count:
                        _chosen = _chosen >> _shift
                        _nth -= _low_count
                        _base += _shift
                group_idx = _base
                coord_expert = _base
                group_begin = cutlass.Int32(first_token_arr[_base])
                group_end = gemm_s
                if _base + 1 < 128:
                    group_end = cutlass.Int32(first_token_arr[_base + 1])

'''+s[b:]
    ast.parse(s)
    digest=hashlib.sha256(s.encode()).hexdigest()
    out=root/'r26_sparse_sources'/f'{digest}.py';out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():assert out.read_text()==s
    else:out.write_text(s)
    return out,digest
