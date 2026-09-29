// Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
// Routing, block-scale packing, frozen Frost GEMMs, and weighted finalization.
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/extra/module.h>

#include <algorithm>
#include <array>
#include <limits>

#include "moe_finalize.cuh"
#include "tvm_ffi_utils.h"

using tvm::ffi::Array;
using tvm::ffi::Function;
using tvm::ffi::Module;
using tvm::ffi::Optional;

namespace {
constexpr DLDataType dl_fp4{kDLFloat4_e2m1fn, 4, 2};
size_t align128(size_t n) { return (n + 127) / 128 * 128; }
void checked(cudaError_t err) { TVM_FFI_ICHECK_EQ(err, cudaSuccess) << cudaGetErrorString(err); }

__global__ void histogram(const int32_t* ids, int32_t* counts, int rows, int experts) {
  for (int64_t r = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; r < rows;
       r += int64_t(blockDim.x) * gridDim.x) {
    int e = ids[r];
    // Keep dummy rows in expert zero so every expanded row has storage. Invalid
    // ids are masked in gather/finalize, never used as memory addresses.
    atomicAdd(counts + (e >= 0 && e < experts ? e : 0), 1);
  }
}

__global__ void histogram_local(const int32_t* ids, int32_t* counts, int rows, int experts) {
  if (experts > 128) {
    for (int64_t r = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; r < rows;
         r += int64_t(blockDim.x) * gridDim.x) {
      int e = ids[r];
      atomicAdd(counts + (e >= 0 && e < experts ? e : 0), 1);
    }
    return;
  }
  __shared__ int local[128];
  if (threadIdx.x < experts) local[threadIdx.x] = 0;
  __syncthreads();
  for (int64_t r = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; r < rows;
       r += int64_t(blockDim.x) * gridDim.x) {
    int e = ids[r];
    atomicAdd(local + (e >= 0 && e < experts ? e : 0), 1);
  }
  __syncthreads();
  if (threadIdx.x < experts && local[threadIdx.x])
    atomicAdd(counts + threadIdx.x, local[threadIdx.x]);
}

template <bool SplitColumns = false>
__global__ void finalize(const __nv_bfloat16* grouped, const int32_t* ids, const int32_t* mapping,
                         const float* scores, __nv_bfloat16* out, int tokens, int hidden, int topk,
                         int experts) {
  union Pack {
    int4 words;
    __nv_bfloat16 values[8];
  };
  // Vectorized data accesses and deterministic FP32 top-k reduction before
  // a single BF16 conversion. Small batches expose independent column tiles.
  const int tiles = SplitColumns ? (hidden / 8 + 127) / 128 : 1;
  for (int64_t task = blockIdx.x; task < int64_t(tokens) * tiles; task += gridDim.x) {
    const int64_t t = task / tiles;
    const int part = task % tiles;
    for (int h = part * blockDim.x + threadIdx.x; h < hidden / 8; h += blockDim.x * tiles) {
      float sum[8] = {};
      for (int k = 0; k < topk; ++k) {
        int64_t r = t * topk + k;
        if (ids[r] >= 0 && ids[r] < experts) {
          Pack value;
          value.words =
              reinterpret_cast<const int4*>(grouped)[int64_t(mapping[r]) * (hidden / 8) + h];
          float score = scores[r];
#pragma unroll
          for (int v = 0; v < 8; ++v) sum[v] += __bfloat162float(value.values[v]) * score;
        }
      }
      Pack result;
#pragma unroll
      for (int v = 0; v < 8; ++v) result.values[v] = __float2bfloat16(sum[v]);
      reinterpret_cast<int4*>(out)[t * (hidden / 8) + h] = result.words;
    }
  }
}

union alignas(16) Bf16x8 {
  uint4 bits;
  __nv_bfloat162 pairs[4];
};

template <int TOPK>
__global__ __launch_bounds__(256, 2) void finalize_vec8(
    const __nv_bfloat16* __restrict__ grouped, const int32_t* __restrict__ ids,
    const int32_t* __restrict__ mapping, const float* __restrict__ scores, int expert_count,
    __nv_bfloat16* __restrict__ output, int tokens, int hidden) {
  constexpr int kThreads = 256;
  constexpr int kElements = 8;
  const int vectors_per_row = hidden / kElements;
  const int tiles_per_row = (vectors_per_row + kThreads - 1) / kThreads;
  const int tile = static_cast<int>(blockIdx.x);
  const int token = tile / tiles_per_row;
  const int vector = (tile - token * tiles_per_row) * kThreads + threadIdx.x;
  if (token >= tokens || vector >= vectors_per_row) return;
  const unsigned mask = 0xffffffffu;
  const int lane = threadIdx.x & 31;
  float2 accum[4];
#pragma unroll
  for (int q = 0; q < 4; ++q) accum[q] = make_float2(0.0f, 0.0f);
#pragma unroll
  for (int j = 0; j < TOPK; ++j) {
    int expert = lane == j ? __ldg(ids + token * TOPK + j) : 0;
    int row = lane == j ? __ldg(mapping + token * TOPK + j) : 0;
    float weight = lane == j ? __ldg(scores + token * TOPK + j) : 0.0f;
    expert = __shfl_sync(mask, expert, j);
    row = __shfl_sync(mask, row, j);
    weight = __shfl_sync(mask, weight, j);
    if (expert >= 0 && expert < expert_count) {
      Bf16x8 value;
      value.bits =
          reinterpret_cast<const uint4*>(grouped + static_cast<int64_t>(row) * hidden)[vector];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float2 x = __bfloat1622float2(value.pairs[q]);
        accum[q].x = fmaf(x.x, weight, accum[q].x);
        accum[q].y = fmaf(x.y, weight, accum[q].y);
      }
    }
  }
  Bf16x8 result;
#pragma unroll
  for (int q = 0; q < 4; ++q) result.pairs[q] = __floats2bfloat162_rn(accum[q].x, accum[q].y);
  reinterpret_cast<uint4*>(output + static_cast<int64_t>(token) * hidden)[vector] = result.bits;
}

#define VEC 8  // bf16 elements per 16B vector load

__device__ __forceinline__ void accum_vec(float (&acc)[VEC], const uint4 r, const float w) {
  const unsigned int u[4] = {r.x, r.y, r.z, r.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    // bf16 -> f32 is a pure bit shift, no rounding.
    acc[2 * i + 0] = fmaf(__uint_as_float(u[i] << 16), w, acc[2 * i + 0]);
    acc[2 * i + 1] = fmaf(__uint_as_float(u[i] & 0xffff0000u), w, acc[2 * i + 1]);
  }
}

__device__ __forceinline__ uint4 pack_bf16(const float (&acc)[VEC]) {
  uint4 o;
  unsigned int* op = reinterpret_cast<unsigned int*>(&o);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    __nv_bfloat162 p = __floats2bfloat162_rn(acc[2 * i + 0], acc[2 * i + 1]);
    op[i] = *reinterpret_cast<unsigned int*>(&p);
  }
  return o;
}

// Vectorized kernel: blockIdx.y = token, blockIdx.x tiles the hidden dim.
// Each thread owns U independent 16B lanes so that U*K loads are in flight.
template <int K, int U, int BD, bool SkipInvalid = false>
__global__ void __launch_bounds__(BD)
    finalize_vec_kernel(const __nv_bfloat16* __restrict__ grouped, const int* __restrict__ ids,
                        const int* __restrict__ mapping, const float* __restrict__ scores,
                        const int E, __nv_bfloat16* __restrict__ output, const int H,
                        const int nvec) {
  const int t = blockIdx.y;

  const uint4* __restrict__ rows[K];
  float w[K];
  bool valid[K];
#pragma unroll
  for (int j = 0; j < K; ++j) {
    const int id = __ldg(ids + (size_t)t * K + j);
    const int m = __ldg(mapping + (size_t)t * K + j);
    valid[j] = id >= 0 && id < E;
    w[j] = valid[j] ? __ldg(scores + (size_t)t * K + j) : 0.0f;
    rows[j] = reinterpret_cast<const uint4*>(grouped + (size_t)(valid[j] ? m : 0) * H);
  }
  uint4* __restrict__ out = reinterpret_cast<uint4*>(output + (size_t)t * H);

  const int v0 = blockIdx.x * (BD * U) + threadIdx.x;

  if (v0 + (U - 1) * BD < nvec) {
    uint4 raw[U][K];
#pragma unroll
    for (int u = 0; u < U; ++u)
#pragma unroll
      for (int j = 0; j < K; ++j)
        raw[u][j] = valid[j] ? __ldg(rows[j] + (v0 + u * BD)) : make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int u = 0; u < U; ++u) {
      float acc[VEC];
#pragma unroll
      for (int i = 0; i < VEC; ++i) acc[i] = 0.0f;
#pragma unroll
      for (int j = 0; j < K; ++j)
        // Preserve the original tail path's signed-zero behavior when its
        // work is covered by a full span in this bounded configuration.
        if (!SkipInvalid || valid[j]) accum_vec(acc, raw[u][j], w[j]);
      out[v0 + u * BD] = pack_bf16(acc);
    }
  } else {
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int v = v0 + u * BD;
      if (v >= nvec) break;
      float acc[VEC];
#pragma unroll
      for (int i = 0; i < VEC; ++i) acc[i] = 0.0f;
#pragma unroll
      for (int j = 0; j < K; ++j)
        if (valid[j]) accum_vec(acc, __ldg(rows[j] + v), w[j]);
      out[v] = pack_bf16(acc);
    }
  }
}

#undef VEC
void tensor(TensorView t, DLDevice device, DLDataType dtype, std::initializer_list<int64_t> shape,
            size_t alignment = 16) {
  TVM_FFI_ICHECK(t.device().device_type == kDLCUDA && t.device().device_id == device.device_id);
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(t.dtype()), encode_dlpack_dtype(dtype));
  TVM_FFI_ICHECK(t.IsContiguous());
  TVM_FFI_ICHECK_EQ(t.ndim(), shape.size());
  int dim = 0;
  for (auto size : shape) TVM_FFI_ICHECK_EQ(t.size(dim++), size);
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(t.data_ptr()) % alignment, 0);
}

__device__ int64_t sf_index(int row, int col, int columns) {
  return int64_t(row / 128) * 128 * columns + (col / 4) * 512 + (row % 32) * 16 +
         ((row % 128) / 32) * 4 + col % 4;
}

__global__ void prefix(const int32_t* counts, int32_t* offsets, int32_t* cursors,
                       int32_t* sf_offsets, int experts, float* scale) {
  if (experts > 128) {
    if (threadIdx.x == 0) {
      int start = 0, sf_start = 0;
      for (int e = 0; e < experts; ++e) {
        offsets[e] = cursors[e] = start;
        sf_offsets[e] = sf_start;
        start += counts[e];
        sf_start += (counts[e] + 127) / 128 * 128;
      }
      scale[0] = 1.f;
      scale[1] = 4.f;
      scale[2] = 25.f;
    }
    return;
  }
  __shared__ int warp_counts[4], warp_scales[4];
  int e = threadIdx.x, lane = e % 32, warp = e / 32;
  int count = e < experts ? counts[e] : 0;
  int padded = (count + 127) / 128 * 128;
  int sum = count, sf_sum = padded;
#pragma unroll
  for (int delta = 1; delta < 32; delta *= 2) {
    int v = __shfl_up_sync(0xffffffff, sum, delta);
    int sv = __shfl_up_sync(0xffffffff, sf_sum, delta);
    if (lane >= delta) {
      sum += v;
      sf_sum += sv;
    }
  }
  if (lane == 31) {
    warp_counts[warp] = sum;
    warp_scales[warp] = sf_sum;
  }
  __syncthreads();
#pragma unroll
  for (int w = 0; w < 4; ++w)
    if (w < warp) {
      sum += warp_counts[w];
      sf_sum += warp_scales[w];
    }
  if (e < experts) {
    offsets[e] = cursors[e] = sum - count;
    sf_offsets[e] = sf_sum - padded;
  }
  if (e == 0) {
    scale[0] = 1.f;
    scale[1] = 4.f;
    scale[2] = 25.f;
  }
}

template <bool Reserve>
__global__ void assign_rows(const int32_t* ids, int32_t* cursors, int32_t* mapping,
                            int32_t* row_experts, int rows, int experts, int32_t* inverse,
                            int topk) {
  const int64_t r = int64_t(blockIdx.x) * 256 + threadIdx.x;
  const bool active = r < rows;
  int e = active ? ids[r] : 0;
  e = e >= 0 && e < experts ? e : 0;
  int dest = 0;
  if constexpr (Reserve) {
    __shared__ int counts[128], starts[128];
    if (threadIdx.x < 128) counts[threadIdx.x] = 0;
    __syncthreads();
    int rank = active ? atomicAdd(counts + e, 1) : 0;
    __syncthreads();
    if (threadIdx.x < experts && counts[threadIdx.x])
      starts[threadIdx.x] = atomicAdd(cursors + threadIdx.x, counts[threadIdx.x]);
    __syncthreads();
    if (active) dest = starts[e] + rank;
  } else {
    if (active) dest = atomicAdd(cursors + e, 1);
  }
  if (active) {
    mapping[r] = dest;
    row_experts[dest] = e;
    if (inverse) inverse[dest] = ids[r] >= 0 && ids[r] < experts ? r / topk : -1;
  }
}

__device__ __forceinline__ uint4 ld_cs16(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}
__device__ __forceinline__ void st_cs16(uint4* p, const uint4& v) {
  asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z),
               "r"(v.w)
               : "memory");
}
__device__ __forceinline__ uint32_t ld_nc32(const void* p) {
  uint32_t v;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ uint4 ld_nc16(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

// ---------------------------------------------------------------------------
// Kernel A: hidden-state gather (token-major, read-once) + inverse metadata.
// blockDim.x == V (16B vectors per row);  each block owns TPB source tokens.
// ---------------------------------------------------------------------------
template <int V, int K, int C, int TILE, bool SWZ>
__global__ __launch_bounds__(256) void gather_input_roles(
    const uint4* __restrict__ xv, uint4* __restrict__ pv, const uint8_t* __restrict__ isf,
    uint8_t* __restrict__ scales, const int32_t* __restrict__ ids,
    const int32_t* __restrict__ mapping, const int32_t* __restrict__ invmeta,
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ sf_offsets, int T, int R,
    int E, int NS) {
  constexpr int NMETA = (TILE * K > 128) ? TILE * K : 128;
  __shared__ int32_t smem[NMETA];
  const int tid = threadIdx.x;

  if (blockIdx.x < (unsigned)NS) {
    // ---------------- scale-tile role ----------------
    constexpr int G = C / 4;
    const int p = blockIdx.x;
    int lo = 0, hi = E - 1;
    while (lo < hi) {
      int mid = (lo + hi + 1) >> 1;
      if ((__ldg(sf_offsets + mid) >> 7) <= p)
        lo = mid;
      else
        hi = mid - 1;
    }
    const int e = lo;
    const int off = __ldg(offsets + e);
    const int cnt = ((e + 1 < E) ? __ldg(offsets + e + 1) : R) - off;
    const int b = p - (__ldg(sf_offsets + e) >> 7);
    if (b * 128 >= cnt) return;

    const int base_d = off + b * 128;
    uint8_t* out = scales + (size_t)(__ldg(sf_offsets + e) + b * 128) * C;

    if (tid < 128) {
      int d = base_d + tid;
      int t = (d < R) ? __ldg(invmeta + d) : -1;
      int sb = -1;
      if (t >= 0) {
        if constexpr (SWZ)
          sb = (t >> 7) * 128 * C + (t & 31) * 16 + ((t & 127) >> 5) * 4;
        else
          sb = t * C;
      }
      smem[tid] = sb;
    }
    __syncthreads();

    const int lane = tid & 31;
    const int warp = tid >> 5;
    int sb0 = smem[lane], sb1 = smem[lane + 32];
    int sb2 = smem[lane + 64], sb3 = smem[lane + 96];

    if constexpr (SWZ) {
      constexpr int gs = (G + 7) / 8;
      int g0 = warp * gs, g1 = min(G, g0 + gs);
      for (int g = g0; g < g1; ++g) {
        const int o = g * 512;
        uint4 r;
        r.x = (sb0 >= 0) ? ld_nc32(isf + sb0 + o) : 0u;
        r.y = (sb1 >= 0) ? ld_nc32(isf + sb1 + o) : 0u;
        r.z = (sb2 >= 0) ? ld_nc32(isf + sb2 + o) : 0u;
        r.w = (sb3 >= 0) ? ld_nc32(isf + sb3 + o) : 0u;
        st_cs16((uint4*)(out + o + lane * 16), r);
      }
    } else {
      constexpr int Gq = G >> 2;
      constexpr int gs = (Gq + 7) / 8;
      int q0 = warp * gs, q1 = min(Gq, q0 + gs);
      const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
      for (int q = q0; q < q1; ++q) {
        const int o16 = q * 16;
        uint4 a = (sb0 >= 0) ? ld_nc16(isf + sb0 + o16) : zero;
        uint4 bb = (sb1 >= 0) ? ld_nc16(isf + sb1 + o16) : zero;
        uint4 c = (sb2 >= 0) ? ld_nc16(isf + sb2 + o16) : zero;
        uint4 dd = (sb3 >= 0) ? ld_nc16(isf + sb3 + o16) : zero;
        uint8_t* ob = out + q * 2048 + lane * 16;
        st_cs16((uint4*)(ob), make_uint4(a.x, bb.x, c.x, dd.x));
        st_cs16((uint4*)(ob + 512), make_uint4(a.y, bb.y, c.y, dd.y));
        st_cs16((uint4*)(ob + 1024), make_uint4(a.z, bb.z, c.z, dd.z));
        st_cs16((uint4*)(ob + 1536), make_uint4(a.w, bb.w, c.w, dd.w));
      }
    }
    return;
  }

  // ---------------- hidden-state gather role ----------------
  constexpr int ILP = TILE * V / 256;
  const int tok0 = (blockIdx.x - NS) * TILE;

  for (int i = tid; i < TILE * K; i += 256) {
    int n = i / K;
    int t = tok0 + n;
    int v = 0x80000000;
    if (t < T) {
      int r = t * K + (i - n * K);
      int id = __ldg(ids + r);
      int d = __ldg(mapping + r);
      v = ((unsigned)id < (unsigned)E) ? d : ~d;
    }
    smem[i] = v;
  }

  uint4 val[ILP];
#pragma unroll
  for (int u = 0; u < ILP; ++u) {
    int L = u * 256 + tid;
    int n = L / V;
    int t = tok0 + n;
    if (t < T) val[u] = ld_cs16(xv + (size_t)t * V + (L - n * V));
  }
  __syncthreads();

  const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
  for (int u = 0; u < ILP; ++u) {
    int L = u * 256 + tid;
    int n = L / V;
    int t = tok0 + n;
    if (t >= T) continue;
    const int v = L - n * V;
#pragma unroll
    for (int j = 0; j < K; ++j) {
      int enc = smem[n * K + j];
      bool ok = enc >= 0;
      int d = ok ? enc : ~enc;
      st_cs16(pv + (size_t)d * V + v, ok ? val[u] : zero);
    }
  }
}

template <int V, int K, int C, int TILE, bool SWZ>
static void launch_roles_specialized(const void* x, void* packed, const uint8_t* isf,
                                     uint8_t* scales, const int32_t* ids, const int32_t* mapping,
                                     const int32_t* invmeta, const int32_t* offsets,
                                     const int32_t* sf_offsets, int T, int R, int E, int NS,
                                     cudaStream_t s) {
  int grid = NS + (T + TILE - 1) / TILE;
  gather_input_roles<V, K, C, TILE, SWZ><<<grid, 256, 0, s>>>((const uint4*)x, (uint4*)packed, isf,
                                                              scales, ids, mapping, invmeta,
                                                              offsets, sf_offsets, T, R, E, NS);
}

void launch_input_roles(const void* x, void* packed, const uint8_t* isf, uint8_t* scales,
                        const int32_t* ids, const int32_t* mapping, const int32_t* invmeta,
                        const int32_t* offsets, const int32_t* sf_offsets, int T, int V, int R,
                        int E, int K, int C, int NS, bool swizzled, cudaStream_t s) {
#define GO(VV, KK, CC, TT)                                                                     \
  do {                                                                                         \
    if (swizzled)                                                                              \
      launch_roles_specialized<VV, KK, CC, TT, true>(                                          \
          x, packed, isf, scales, ids, mapping, invmeta, offsets, sf_offsets, T, R, E, NS, s); \
    else                                                                                       \
      launch_roles_specialized<VV, KK, CC, TT, false>(                                         \
          x, packed, isf, scales, ids, mapping, invmeta, offsets, sf_offsets, T, R, E, NS, s); \
  } while (0)
  if (V == 64 && K == 6)
    GO(64, 6, 128, 16);
  else if (V == 64 && K == 8)
    GO(64, 8, 128, 16);
  else if (V == 128 && K == 2)
    GO(128, 2, 256, 8);
  else
    GO(224, 2, 448, 8);
#undef GO
}

template <bool PackedScale>
__global__ void gather_copy(const uint8_t* x, const uint8_t* input_sf, const int32_t* ids,
                            const int32_t* offsets, const int32_t* sf_offsets,
                            const int32_t* mapping, uint8_t* grouped, uint8_t* sf, int rows,
                            int hidden, int topk, int experts, bool swizzled) {
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  for (int64_t r = int64_t(blockIdx.x) * 4 + warp; r < rows; r += int64_t(gridDim.x) * 4) {
    int e = ids[r];
    bool valid = e >= 0 && e < experts;
    e = valid ? e : 0;
    const int dest = __ldg(mapping + r);
    auto source = reinterpret_cast<const int4*>(x) + (r / topk) * (hidden / 32);
    auto target = reinterpret_cast<int4*>(grouped) + int64_t(dest) * (hidden / 32);
    for (int h = lane; h < hidden / 32; h += 32)
      target[h] = valid ? source[h] : make_int4(0, 0, 0, 0);
    const int cols = hidden / 16;
    if constexpr (PackedScale) {
      for (int col = lane * 4; col < cols; col += 128) {
        int64_t src = swizzled ? sf_index(r / topk, col, cols) : (r / topk) * cols + col;
        int64_t dst = int64_t(sf_offsets[e]) * cols + sf_index(dest - offsets[e], col, cols);
        *reinterpret_cast<uint32_t*>(sf + dst) =
            valid ? *reinterpret_cast<const uint32_t*>(input_sf + src) : 0;
      }
    } else {
      for (int col = lane; col < cols; col += 32) {
        int64_t src = swizzled ? sf_index(r / topk, col, cols) : (r / topk) * cols + col;
        int64_t dst = int64_t(sf_offsets[e]) * cols + sf_index(dest - offsets[e], col, cols);
        sf[dst] = valid ? input_sf[src] : 0;
      }
    }
  }
}

__global__ void gather(const uint8_t* x, const uint8_t* input_sf, const int32_t* ids,
                       const int32_t* offsets, const int32_t* sf_offsets, int32_t* cursors,
                       int32_t* mapping, int32_t* row_experts, uint8_t* grouped, uint8_t* sf,
                       int rows, int hidden, int topk, int experts, bool swizzled) {
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  for (int64_t r = int64_t(blockIdx.x) * 4 + warp; r < rows; r += int64_t(gridDim.x) * 4) {
    int e = ids[r];
    bool valid = e >= 0 && e < experts;
    e = valid ? e : 0;
    int dest = 0;
    if (lane == 0) {
      dest = atomicAdd(cursors + e, 1);
      mapping[r] = dest;
      row_experts[dest] = e;
    }
    dest = __shfl_sync(0xffffffff, dest, 0);
    auto source = reinterpret_cast<const int4*>(x) + (r / topk) * (hidden / 32);
    auto target = reinterpret_cast<int4*>(grouped) + int64_t(dest) * (hidden / 32);
    for (int h = lane; h < hidden / 32; h += 32)
      target[h] = valid ? source[h] : make_int4(0, 0, 0, 0);
    int cols = hidden / 16;
    for (int col = lane; col < cols; col += 32) {
      int64_t src = swizzled ? sf_index(r / topk, col, cols) : (r / topk) * cols + col;
      int64_t dst = int64_t(sf_offsets[e]) * cols + sf_index(dest - offsets[e], col, cols);
      sf[dst] = valid ? input_sf[src] : 0;
    }
  }
}

__global__ void gather_original(const uint8_t* x, const uint8_t* input_sf, const int32_t* ids,
                                const int32_t* offsets, const int32_t* sf_offsets, int32_t* cursors,
                                int32_t* mapping, int32_t* row_experts, uint8_t* grouped,
                                uint8_t* sf, int rows, int hidden, int topk, int experts,
                                bool swizzled) {
  __shared__ int dest;
  for (int64_t r = blockIdx.x; r < rows; r += gridDim.x) {
    int e = ids[r];
    bool valid = e >= 0 && e < experts;
    e = valid ? e : 0;
    if (threadIdx.x == 0) {
      dest = atomicAdd(cursors + e, 1);
      mapping[r] = dest;
      row_experts[dest] = e;
    }
    __syncthreads();
    auto source = reinterpret_cast<const int4*>(x) + (r / topk) * (hidden / 32);
    auto target = reinterpret_cast<int4*>(grouped) + int64_t(dest) * (hidden / 32);
    for (int h = threadIdx.x; h < hidden / 32; h += blockDim.x)
      target[h] = valid ? source[h] : make_int4(0, 0, 0, 0);
    int cols = hidden / 16;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
      int64_t src = swizzled ? sf_index(r / topk, col, cols) : (r / topk) * cols + col;
      int64_t dst = int64_t(sf_offsets[e]) * cols + sf_index(dest - offsets[e], col, cols);
      sf[dst] = valid ? input_sf[src] : 0;
    }
    __syncthreads();
  }
}

// At most eight expanded rows: derive offsets from the row list in registers.
// Every CTA is independent, and invalid IDs retain expert-zero dummy storage.
__global__ void route_tiny(const uint8_t* x, const uint8_t* input_sf, const int32_t* ids,
                           int32_t* offsets, int32_t* sf_offsets, int32_t* mapping,
                           int32_t* row_experts, uint8_t* grouped, uint8_t* sf, float* scale,
                           int rows, int hidden, int topk, int experts, bool swizzled) {
  const int r = blockIdx.x, lane = threadIdx.x % 32;
  const bool active = r < rows;
  int dest = 0, begin = 0, sfbegin = 0, expert = 0, valid = 0;
  if (lane == 0) {
    int es[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int value = j < rows ? ids[j] : 0;
      es[j] = value >= 0 && value < experts ? value : 0;
    }
    if (active) {
      const int value = ids[r];
      valid = value >= 0 && value < experts;
      expert = es[r];
    }
    int off = 0, sfoff = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j)
      if (j < rows) {
        bool first = true;
#pragma unroll
        for (int q = 0; q < 8; ++q)
          if (q < j && es[q] == es[j]) first = false;
        off += es[j] < r;
        sfoff += (es[j] < r && first) * 128;
        begin += es[j] < expert;
        sfbegin += (es[j] < expert && first) * 128;
        dest += es[j] < expert || (es[j] == expert && j < r);
      }
    if (threadIdx.x == 0) {
      if (r < experts) {
        offsets[r] = off;
        sf_offsets[r] = sfoff;
      }
      if (active) {
        mapping[r] = dest;
        row_experts[dest] = expert;
      }
      if (r == 0) {
        scale[0] = 1.f;
        scale[1] = 4.f;
        scale[2] = 25.f;
      }
    }
  }
  dest = __shfl_sync(0xffffffff, dest, 0);
  begin = __shfl_sync(0xffffffff, begin, 0);
  sfbegin = __shfl_sync(0xffffffff, sfbegin, 0);
  valid = __shfl_sync(0xffffffff, valid, 0);
  if (active) {
    auto source = reinterpret_cast<const int4*>(x) + (r / topk) * (hidden / 32);
    auto target = reinterpret_cast<int4*>(grouped) + int64_t(dest) * (hidden / 32);
    for (int h = threadIdx.x; h < hidden / 32; h += blockDim.x)
      target[h] = valid ? source[h] : make_int4(0, 0, 0, 0);
    const int cols = hidden / 16;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
      const int64_t src = swizzled ? sf_index(r / topk, col, cols) : (r / topk) * cols + col;
      const int64_t dst = int64_t(sfbegin) * cols + sf_index(dest - begin, col, cols);
      sf[dst] = valid ? input_sf[src] : 0;
    }
  }
}

// <=32 expanded rows: warp match/ballot provides stable expert ranks and
// padded scale prefixes. No shared atomics, serial expert scan, or CTA sync.
__global__ void route_warp(const uint8_t* x, const uint8_t* input_sf, const int32_t* ids,
                           int32_t* offsets, int32_t* sf_offsets, int32_t* mapping,
                           int32_t* row_experts, uint8_t* grouped, uint8_t* sf, float* scale,
                           int rows, int hidden, int topk, int experts, bool swizzled) {
  const int r = blockIdx.x, lane = threadIdx.x & 31;
  const bool active = r < rows;
  const int raw = active ? ids[r] : 0;
  const bool valid = active && raw >= 0 && raw < experts;
  const int expert = valid ? raw : 0;
  int value = lane < rows ? ids[lane] : experts;
  if (lane < rows && (value < 0 || value >= experts)) value = 0;
  const unsigned peers = __match_any_sync(0xffffffffu, value);
  const bool first = lane == __ffs(peers) - 1;
  const int off = __popc(__ballot_sync(0xffffffffu, lane < rows && value < r));
  const int sf_off = __popc(__ballot_sync(0xffffffffu, lane < rows && first && value < r)) * 128;
  const int begin = __popc(__ballot_sync(0xffffffffu, lane < rows && value < expert));
  const int sf_begin =
      __popc(__ballot_sync(0xffffffffu, lane < rows && first && value < expert)) * 128;
  const int dest = __popc(
      __ballot_sync(0xffffffffu, lane < rows && (value < expert || (value == expert && lane < r))));
  if (threadIdx.x == 0) {
    if (r < experts) {
      offsets[r] = off;
      sf_offsets[r] = sf_off;
    }
    if (active) {
      mapping[r] = dest;
      row_experts[dest] = expert;
    }
    if (r == 0) {
      scale[0] = 1.f;
      scale[1] = 4.f;
      scale[2] = 25.f;
    }
  }
  if (active) {
    auto source = reinterpret_cast<const int4*>(x) + (r / topk) * (hidden / 32);
    auto target = reinterpret_cast<int4*>(grouped) + int64_t(dest) * (hidden / 32);
    for (int h = threadIdx.x; h < hidden / 32; h += blockDim.x)
      target[h] = valid ? source[h] : make_int4(0, 0, 0, 0);
    const int cols = hidden / 16;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
      const int64_t src = swizzled ? sf_index(r / topk, col, cols) : (r / topk) * cols + col;
      const int64_t dst = int64_t(sf_begin) * cols + sf_index(dest - begin, col, cols);
      sf[dst] = valid ? input_sf[src] : 0;
    }
  }
}

// Small batches compute stable routing and expert-local scale segments per CTA.
// No CTA consumes metadata written by another CTA in this launch.
__global__ void route_small(const uint8_t* x, const uint8_t* input_sf, const int32_t* ids,
                            int32_t* offsets, int32_t* sf_offsets, int32_t* mapping,
                            int32_t* row_experts, uint8_t* grouped, uint8_t* sf, float* scale,
                            int rows, int hidden, int topk, int experts, bool swizzled) {
  const int r = blockIdx.x;
  const bool active = r < rows;
  const int raw = active ? ids[r] : 0;
  const bool valid = active && raw >= 0 && raw < experts;
  const int expert = valid ? raw : 0;
  __shared__ int counts[256];
  __shared__ int partial[4];
  __shared__ int destination, row_begin, row_sf_begin;
  for (int e = threadIdx.x; e < experts; e += blockDim.x) counts[e] = 0;
  __syncthreads();
  int before = 0;
  for (int j = threadIdx.x; j < rows; j += blockDim.x) {
    const int value = ids[j];
    const int e = value >= 0 && value < experts ? value : 0;
    atomicAdd(counts + e, 1);
    before += e < expert || (e == expert && j < r);
  }
  before = __reduce_add_sync(0xffffffff, before);
  if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = before;
  __syncthreads();
  if (threadIdx.x == 0) {
    destination = partial[0] + partial[1] + partial[2] + partial[3];
    int begin = 0, sf_begin = 0;
    for (int e = 0; e < experts; ++e) {
      if (e == r) {
        offsets[r] = begin;
        sf_offsets[r] = sf_begin;
      }
      if (e == expert) {
        row_begin = begin;
        row_sf_begin = sf_begin;
      }
      begin += counts[e];
      sf_begin += (counts[e] + 127) / 128 * 128;
    }
    if (active) {
      mapping[r] = destination;
      row_experts[destination] = expert;
    }
    if (r == 0) {
      scale[0] = 1.f;
      scale[1] = 4.f;
      scale[2] = 25.f;
    }
  }
  __syncthreads();
  if (active) {
    auto source = reinterpret_cast<const int4*>(x) + (r / topk) * (hidden / 32);
    auto target = reinterpret_cast<int4*>(grouped) + int64_t(destination) * (hidden / 32);
    for (int h = threadIdx.x; h < hidden / 32; h += blockDim.x)
      target[h] = valid ? source[h] : make_int4(0, 0, 0, 0);
    const int cols = hidden / 16;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
      int64_t src = swizzled ? sf_index(r / topk, col, cols) : (r / topk) * cols + col;
      int64_t dst = int64_t(row_sf_begin) * cols + sf_index(destination - row_begin, col, cols);
      sf[dst] = valid ? input_sf[src] : 0;
    }
  }
}
__global__ void route_small_parallel(const uint8_t* x, const uint8_t* input_sf, const int32_t* ids,
                                     int32_t* offsets, int32_t* sf_offsets, int32_t* mapping,
                                     int32_t* row_experts, uint8_t* grouped, uint8_t* sf,
                                     float* scale, int rows, int hidden, int topk, int experts,
                                     bool swizzled) {
  const int r = blockIdx.x;
  const bool active = r < rows;
  const int raw = active ? ids[r] : 0;
  const bool valid = active && raw >= 0 && raw < experts;
  const int expert = valid ? raw : 0;
  __shared__ int counts[256];
  __shared__ int partial[4];
  __shared__ int destination, row_begin, row_sf_begin;
  for (int e = threadIdx.x; e < experts; e += blockDim.x) counts[e] = 0;
  __syncthreads();
  int before = 0;
  for (int j = threadIdx.x; j < rows; j += blockDim.x) {
    const int value = ids[j];
    const int e = value >= 0 && value < experts ? value : 0;
    atomicAdd(counts + e, 1);
    before += e < expert || (e == expert && j < r);
  }
  before = __reduce_add_sync(0xffffffff, before);
  if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = before;
  __syncthreads();
  if (threadIdx.x < 32) {
    int begin = 0, sf_begin = 0, off = 0, sf_off = 0;
    for (int e = threadIdx.x; e < experts; e += 32) {
      const int count = counts[e];
      const int padded = (count + 127) / 128 * 128;
      off += e < r ? count : 0;
      sf_off += e < r ? padded : 0;
      begin += e < expert ? count : 0;
      sf_begin += e < expert ? padded : 0;
    }
    off = __reduce_add_sync(0xffffffff, off);
    sf_off = __reduce_add_sync(0xffffffff, sf_off);
    begin = __reduce_add_sync(0xffffffff, begin);
    sf_begin = __reduce_add_sync(0xffffffff, sf_begin);
    if (threadIdx.x == 0) {
      destination = partial[0] + partial[1] + partial[2] + partial[3];
      if (r < experts) {
        offsets[r] = off;
        sf_offsets[r] = sf_off;
      }
      row_begin = begin;
      row_sf_begin = sf_begin;
      if (active) {
        mapping[r] = destination;
        row_experts[destination] = expert;
      }
      if (r == 0) {
        scale[0] = 1.f;
        scale[1] = 4.f;
        scale[2] = 25.f;
      }
    }
  }
  __syncthreads();
  if (active) {
    auto source = reinterpret_cast<const int4*>(x) + (r / topk) * (hidden / 32);
    auto target = reinterpret_cast<int4*>(grouped) + int64_t(destination) * (hidden / 32);
    for (int h = threadIdx.x; h < hidden / 32; h += blockDim.x)
      target[h] = valid ? source[h] : make_int4(0, 0, 0, 0);
    const int cols = hidden / 16;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
      int64_t src = swizzled ? sf_index(r / topk, col, cols) : (r / topk) * cols + col;
      int64_t dst = int64_t(row_sf_begin) * cols + sf_index(destination - row_begin, col, cols);
      sf[dst] = valid ? input_sf[src] : 0;
    }
  }
}

template <bool SplitColumns = false>
__global__ void requantize(const __nv_bfloat16* input, const int32_t* row_experts,
                           const int32_t* offsets, const int32_t* sf_offsets, uint8_t* output,
                           uint8_t* scales, const float* global_scale, int rows, int width) {
  union InputPack {
    int4 words;
    __nv_bfloat16 values[8];
  };
  union OutputPack {
    uint32_t words;
    uint8_t values[4];
  };
  // Column tiles contain whole 16-element blocks, preserving scale reductions.
  const int tiles = SplitColumns ? (width + 1023) / 1024 : 1;
  for (int64_t task = blockIdx.x; task < int64_t(rows) * tiles; task += gridDim.x) {
    const int64_t row = task / tiles;
    const int part = task % tiles;
    int e = row_experts[row];
    for (int col = (part * blockDim.x + threadIdx.x) * 8; col < width;
         col += blockDim.x * 8 * tiles) {
      InputPack in;
      in.words = reinterpret_cast<const int4*>(input)[(row * width + col) / 8];
      float values[8], maximum = 0.f;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        values[j] = __bfloat162float(in.values[j]);
        maximum = fmaxf(maximum, fabsf(values[j]));
      }
      // Two adjacent lanes own one 16-element NVFP4 block. FC1 already
      // rounded its activated output to BF16, matching CUTLASS's quantizer.
      auto mask = __activemask();
      maximum = fmaxf(maximum, __shfl_xor_sync(mask, maximum, 1));
      __nv_fp8_e4m3 sf(maximum * (1.f / 6.f) * global_scale[0]);
      float scale = static_cast<float>(sf);
      float inverse = maximum == 0.f ? 0.f : global_scale[0] / scale;
      if (threadIdx.x % 2 == 0) {
        int cols = width / 16;
        scales[int64_t(sf_offsets[e]) * cols + sf_index(row - offsets[e], col / 16, cols)] = sf.__x;
      }
      OutputPack out;
#pragma unroll
      for (int j = 0; j < 4; ++j)
        out.values[j] = __nv_cvt_float2_to_fp4x2(
            make_float2(values[2 * j] * inverse, values[2 * j + 1] * inverse), __NV_E2M1,
            cudaRoundNearest);
      reinterpret_cast<uint32_t*>(output)[(row * width + col) / 8] = out.words;
    }
  }
}

template <bool SplitColumns = false>
__global__ void requantize_warp(const __nv_bfloat16* input, const int32_t* row_experts,
                                const int32_t* offsets, const int32_t* sf_offsets, uint8_t* output,
                                uint8_t* scales, const float* global_scale, int rows, int width) {
  union InputPack {
    int4 words;
    __nv_bfloat16 values[8];
  };
  union OutputPack {
    uint32_t words;
    uint8_t values[4];
  };
  // Column tiles contain whole 16-element blocks, preserving scale reductions.
  const int tiles = 1;
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  for (int64_t task = int64_t(blockIdx.x) * 4 + warp; task < int64_t(rows);
       task += int64_t(gridDim.x) * 4) {
    const int64_t row = task / tiles;
    const int part = task % tiles;
    int e = row_experts[row];
    for (int col = lane * 8; col < width; col += 32 * 8) {
      InputPack in;
      in.words = reinterpret_cast<const int4*>(input)[(row * width + col) / 8];
      float values[8], maximum = 0.f;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        values[j] = __bfloat162float(in.values[j]);
        maximum = fmaxf(maximum, fabsf(values[j]));
      }
      // Two adjacent lanes own one 16-element NVFP4 block. FC1 already
      // rounded its activated output to BF16, matching CUTLASS's quantizer.
      auto mask = __activemask();
      maximum = fmaxf(maximum, __shfl_xor_sync(mask, maximum, 1));
      __nv_fp8_e4m3 sf(maximum * (1.f / 6.f) * global_scale[0]);
      float scale = static_cast<float>(sf);
      float inverse = maximum == 0.f ? 0.f : global_scale[0] / scale;
      if (threadIdx.x % 2 == 0) {
        int cols = width / 16;
        scales[int64_t(sf_offsets[e]) * cols + sf_index(row - offsets[e], col / 16, cols)] = sf.__x;
      }
      OutputPack out;
#pragma unroll
      for (int j = 0; j < 4; ++j)
        out.values[j] = __nv_cvt_float2_to_fp4x2(
            make_float2(values[2 * j] * inverse, values[2 * j + 1] * inverse), __NV_E2M1,
            cudaRoundNearest);
      reinterpret_cast<uint32_t*>(output)[(row * width + col) / 8] = out.words;
    }
  }
}

// Each thread owns one complete 16-element NVFP4 block. Preserve the
// producer's BF16 rounding and emit the existing expert-padded scale layout.
// The prepared dispatch below bounds the tested geometry and index range.
__global__ void prepare_input_rows(const int32_t* ids, const int32_t* mapping, int32_t* inverse,
                                   int rows, int topk, int experts) {
  for (int64_t r = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; r < rows;
       r += int64_t(blockDim.x) * gridDim.x)
    inverse[mapping[r]] = (ids[r] >= 0 && ids[r] < experts) ? r / topk : -1;
}

__global__ void linearize_input_scales(const uint8_t* input, uint8_t* output, int tokens,
                                       int hidden) {
  int words_per_row = hidden / 64, cols = hidden / 16;
  for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < int64_t(tokens) * words_per_row; index += int64_t(blockDim.x) * gridDim.x) {
    int row = index / words_per_row, col = (index % words_per_row) * 4;
    reinterpret_cast<uint32_t*>(output)[index] =
        *reinterpret_cast<const uint32_t*>(input + sf_index(row, col, cols));
  }
}

__device__ __forceinline__ float quant_fp8_to_float(uint8_t bits) {
  __nv_fp8_e4m3 v;
  *reinterpret_cast<uint8_t*>(&v) = bits;
  return static_cast<float>(v);
}

__device__ __forceinline__ uint8_t quant_float_to_fp8(float v) {
  __nv_fp8_e4m3 x(v);
  return *reinterpret_cast<uint8_t*>(&x);
}

__device__ __forceinline__ float quant_approx_div(float a, float b) {
  float out;
  asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
  return out;
}

template <int QCOLS>
__global__ __launch_bounds__(256, 2) void requantize_blockwise(
    const __nv_bfloat16* __restrict__ input, const int* __restrict__ row_experts,
    const int* __restrict__ offsets, const int* __restrict__ sf_offsets,
    const float* __restrict__ global_scale, uint8_t* __restrict__ packed,
    uint8_t* __restrict__ scales, int rows, int cols) {
  const int qcols = QCOLS == 0 ? (cols >> 4) : QCOLS;
  const int q = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (q >= rows * qcols) return;

  const int row = q / qcols;
  const int col = q - row * qcols;
  const uint4* src = reinterpret_cast<const uint4*>(input + static_cast<long long>(q) * 16);
  const uint4 a = src[0];
  const uint4 b = src[1];
  const uint32_t words[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
  float x[16];
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const __nv_bfloat162 p = *reinterpret_cast<const __nv_bfloat162*>(&words[i]);
    const float2 f = __bfloat1622float2(p);
    x[2 * i] = f.x;
    x[2 * i + 1] = f.y;
    amax = fmaxf(amax, fabsf(f.x));
    amax = fmaxf(amax, fabsf(f.y));
  }

  const float gs = __ldg(global_scale);
  float sf_value = (amax * 0.16666667163372039794921875f) * gs;
  sf_value = fminf(sf_value, 448.0f);
  const uint8_t sf_bits = quant_float_to_fp8(sf_value);
  const float inverse = (amax == 0.0f) ? 0.0f : quant_approx_div(gs, quant_fp8_to_float(sf_bits));

  uint64_t out = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const __nv_fp4x2_storage_t p = __nv_cvt_float2_to_fp4x2(
        make_float2(x[2 * i] * inverse, x[2 * i + 1] * inverse), __NV_E2M1, cudaRoundNearest);
    out |= static_cast<uint64_t>(*reinterpret_cast<const uint8_t*>(&p)) << (8 * i);
  }
  *reinterpret_cast<uint64_t*>(packed + static_cast<long long>(q) * 8) = out;

  const int expert = __ldg(row_experts + row);
  const int rr = row - __ldg(offsets + expert);
  const int scale_index = __ldg(sf_offsets + expert) * qcols + (rr >> 7) * (128 * qcols) +
                          (col >> 2) * 512 + (rr & 31) * 16 + ((rr & 127) >> 5) * 4 + (col & 3);
  scales[scale_index] = sf_bits;
}

class CudnnFrostNvfp4MoePlan final : public tvm::ffi::ModuleObj {
 public:
  CudnnFrostNvfp4MoePlan(Function fc1, Function fc2, int64_t tokens, int64_t hidden,
                         int64_t intermediate, int64_t experts, int64_t topk, int device,
                         size_t scratch1, size_t scratch2, bool gated, Array<int64_t> tail1,
                         Array<int64_t> tail2, bool swap1, bool swap2, bool swizzled, bool fma,
                         bool input_fused, bool swap_quantized)
      : fc1_(std::move(fc1)),
        fc2_(std::move(fc2)),
        t_(tokens),
        h_(hidden),
        i_(intermediate),
        e_(experts),
        k_(topk),
        s_(tokens * topk),
        device_{kDLCUDA, device},
        scratch1_(scratch1),
        scratch2_(scratch2),
        gated_(gated),
        tail1_(std::move(tail1)),
        tail2_(std::move(tail2)),
        quantized_(fma || swap_quantized ||
                   std::find(tail1_.begin(), tail1_.end(), 6) != tail1_.end()),
        swap1_(swap1),
        swap2_(swap2),
        swizzled_(swizzled),
        fma_(fma),
        input_fused_(input_fused),
        swap_quantized_(swap_quantized) {
    int64_t active = std::min(s_, e_);
    // FMA keeps routes in token/slot order, with one scale segment per route.
    sf_rows_ = fma_ ? 128 * s_ : 128 * (active + (s_ - active) / 128);
    size_t pos = 0;
    auto reserve = [&](size_t bytes) {
      size_t start = pos;
      pos += align128(bytes);
      return start;
    };
    // FC2 can overwrite the grouped input after FC1 has consumed it.
    x_pos_ = reserve(fma_ ? 0 : s_ * h_ * 2);
    mid_pos_ = reserve(quantized_ ? 0 : s_ * i_ * 2);
    qmid_pos_ = reserve(s_ * i_ / 2);
    sf1_pos_ = reserve(fma_ ? 0 : sf_rows_ * h_ / 16);
    sf2_pos_ = reserve(sf_rows_ * i_ / 16);
    counts_pos_ = reserve(fma_ ? 0 : e_ * 4);
    offsets_pos_ = reserve(fma_ ? 0 : e_ * 4);
    sf_offsets_pos_ = reserve(fma_ ? 0 : e_ * 4);
    cursors_pos_ = reserve(fma_ ? 0 : e_ * 4);
    mapping_pos_ = reserve(fma_ ? 0 : s_ * 4);
    row_experts_pos_ = reserve(fma_ ? 0 : s_ * 4);
    scale_pos_ = reserve(fma_ ? 0 : 3 * sizeof(float));
    scratch_pos_ = reserve(std::max(scratch1_, scratch2_));
    workspace_size_ = pos;
    if (fma_) return;
    auto problem = [&](int64_t n, int64_t k, bool swap, bool gated, bool quantized = false) {
      Array<int64_t> shape{swap ? n : s_, swap ? s_ : n, k, e_, e_};
      auto token = [&]() {
        shape.push_back(k / 2);
        shape.push_back(1);
        shape.push_back(s_ * k / 2);
      };
      auto weight = [&]() {
        shape.push_back(k / 2);
        shape.push_back(1);
        shape.push_back((gated ? 2 : 1) * n * k / 2);
      };
      if (!swap) token();
      weight();
      if (gated) weight();
      if (swap) token();
      shape.push_back(swap ? 1 : (quantized ? n / 2 : n));
      shape.push_back(swap ? n : 1);
      shape.push_back(s_ * n / (quantized ? 2 : 1));
      if (quantized) {
        shape.push_back(n / 16);
        shape.push_back(1);
        shape.push_back(sf_rows_ * n / 16);
      }
      return shape;
    };
    problem1_ = problem(i_, h_, swap1_, gated_, quantized_ && !swap_quantized_);
    problem2_ = problem(h_, i_, swap2_, false);
  }

  const char* kind() const final { return "cudnn_frost_nvfp4_moe_plan"; }
  Optional<Function> GetFunction(const tvm::ffi::String& name) final {
    if (name == "swap_quantized_enabled")
      return Function::FromTyped([this]() { return swap_quantized_; });
    if (name == "input_fused_enabled")
      return Function::FromTyped([this]() { return input_fused_; });
    if (name == "finalize_variant")
      return Function::FromTyped([this]() { return finalize_variant(); });
    if (name == "workspace_size")
      return Function::FromTyped([this]() { return int64_t(workspace_size_); });
    if (name == "stage_layout")
      return Function::FromTyped([this]() {
        TVM_FFI_ICHECK(!fma_ && !input_fused_)
            << "FMA and fused input plans do not materialize grouped stage inputs";
        return Array<int64_t>{int64_t(x_pos_),
                              quantized_ ? -1 : int64_t(mid_pos_),
                              int64_t(qmid_pos_),
                              int64_t(sf1_pos_),
                              int64_t(sf2_pos_),
                              int64_t(offsets_pos_),
                              sf_rows_};
      });
    if (name == "run" || name == "prepare_stages") {
      bool stages = name == "prepare_stages";
      return Function::FromTyped(
          [this, stages](TensorView out, TensorView x, TensorView ids, TensorView scores,
                         TensorView w1, TensorView w2, TensorView sf1, TensorView sf2,
                         TensorView xsf, TensorView global1, TensorView alpha1, TensorView global2,
                         TensorView alpha2, TensorView workspace) {
            run(out, x, ids, scores, w1, w2, sf1, sf2, xsf, global1, alpha1, global2, alpha2,
                workspace, stages);
          });
    }
    return Function(nullptr);
  }

 private:
  int64_t finalize_variant() const {
    if (frost_moe_finalize::large_supported(t_, h_, i_, e_, k_)) return 3;
    if (frost_moe_finalize::small_supported(t_, h_, i_, e_, k_)) return 1;
    if (frost_moe_finalize::shape_supported(t_, h_, i_, e_, k_)) return 2;
    return 0;
  }
  bool use_tiled_input_gather() const {
    // Use the validated input path after physical row assignment.
    return !input_fused_ && s_ >= 8192 && t_ <= 32768 && (t_ <= 12288 || e_ == 12 || e_ == 64) &&
           ((e_ == 64 && h_ == 2048 && i_ == 1408 && k_ == 6) ||
            (e_ == 12 && h_ == 7168 && i_ == 3072 && k_ == 2) ||
            (e_ == 8 && h_ == 4096 && i_ == 14336 && k_ == 2));
  }

  void run(TensorView out, TensorView x, TensorView ids, TensorView scores, TensorView w1,
           TensorView w2, TensorView sf1, TensorView sf2, TensorView xsf, TensorView global1,
           TensorView alpha1, TensorView global2, TensorView alpha2, TensorView workspace,
           bool stages) const {
    TVM_FFI_ICHECK(!stages || !input_fused_)
        << "Fused input plans do not materialize grouped stage inputs";
    tensor(out, device_, dl_bfloat16, {t_, h_});
    tensor(x, device_, dl_uint8, {t_, h_ / 2});
    tensor(ids, device_, dl_int32, {t_, k_}, 4);
    tensor(scores, device_, dl_float32, {t_, k_}, 4);
    tensor(w1, device_, dl_uint8, {e_, (gated_ ? 2 : 1) * i_, h_ / 2});
    tensor(w2, device_, dl_uint8, {e_, h_, i_ / 2});
    tensor(sf1, device_, dl_uint8, {(gated_ ? 2 : 1), e_, i_, h_ / 16});
    tensor(sf2, device_, dl_uint8, {e_, h_, i_ / 16});
    tensor(global1, device_, dl_float32, {1}, 4);
    tensor(global2, device_, dl_float32, {1}, 4);
    tensor(alpha1, device_, dl_float32, {e_}, 4);
    tensor(alpha2, device_, dl_float32, {e_}, 4);
    if (swizzled_)
      tensor(xsf, device_, dl_uint8, {(t_ + 127) / 128 * 128 * h_ / 16});
    else
      tensor(xsf, device_, dl_uint8, {t_, h_ / 16});
    tensor(workspace, device_, dl_uint8, {workspace.numel()}, 128);
    TVM_FFI_ICHECK_GE(workspace.numel(), workspace_size_);
    ffi::CUDADeviceGuard guard(device_.device_id);
    auto stream = get_stream(device_);
    auto base = static_cast<char*>(workspace.data_ptr());
    if (fma_) {
      TVM_FFI_ICHECK(!stages) << "FMA plans do not materialize grouped stage inputs";
      int64_t qshape[]{s_ * i_ / 2}, sfshape[]{sf_rows_ * i_ / 16}, unit[]{1};
      int64_t xshape[]{swizzled_ ? 1 : t_, swizzled_ ? 128 * h_ / 16 : h_ / 16};
      int64_t xstride[]{xshape[1], 1};
      DLTensor qm{base + qmid_pos_, device_, 1, dl_uint8, qshape, unit, 0};
      DLTensor sfm{base + sf2_pos_, device_, 1, dl_uint8, sfshape, unit, 0};
      DLTensor sx{xsf.data_ptr(), device_, 2, dl_uint8, xshape, xstride, 0};
      // Both functions consume current bindings. The plan owns their compiled
      // modules, and graph retention keeps the plan and scratch alive on replay.
      fc1_(x, TensorView(&sx), w1, sf1, alpha1, ids, TensorView(&qm), TensorView(&sfm), global2,
           static_cast<void*>(stream));
      fc2_(TensorView(&qm), TensorView(&sfm), w2, sf2, alpha2, ids, scores, out,
           static_cast<void*>(stream));
      checked(cudaGetLastError());
      return;
    }
    auto gx = reinterpret_cast<uint8_t*>(base + x_pos_);
    auto mid = reinterpret_cast<__nv_bfloat16*>(base + mid_pos_);
    auto qm = reinterpret_cast<uint8_t*>(base + qmid_pos_);
    auto gy = reinterpret_cast<__nv_bfloat16*>(base + x_pos_);
    auto sfx = reinterpret_cast<uint8_t*>(base + sf1_pos_);
    auto sfm = reinterpret_cast<uint8_t*>(base + sf2_pos_);
    auto counts = reinterpret_cast<int32_t*>(base + counts_pos_);
    auto offsets = reinterpret_cast<int32_t*>(base + offsets_pos_);
    auto sf_offsets = reinterpret_cast<int32_t*>(base + sf_offsets_pos_);
    auto cursors = reinterpret_cast<int32_t*>(base + cursors_pos_);
    auto mapping = reinterpret_cast<int32_t*>(base + mapping_pos_);
    auto row_experts = reinterpret_cast<int32_t*>(base + row_experts_pos_);
    auto scale = reinterpret_cast<float*>(base + scale_pos_);
    auto scratch = reinterpret_cast<int64_t*>(base + scratch_pos_);
    auto expert_ids = static_cast<int32_t*>(ids.data_ptr());
    if (s_ <= 32 && ((e_ == 128 && h_ == 2048 && i_ == 768 && k_ == 8) ||
                     (e_ == 64 && h_ == 2048 && i_ == 1408 && k_ == 6 && s_ > 8))) {
      route_warp<<<std::max(s_, e_), 128, 0, stream>>>(
          static_cast<const uint8_t*>(x.data_ptr()), static_cast<const uint8_t*>(xsf.data_ptr()),
          expert_ids, offsets, sf_offsets, mapping, row_experts, gx, sfx, scale, s_, h_, k_, e_,
          swizzled_);
    } else if (s_ <= 8 && e_ == 64) {
      route_tiny<<<std::max(s_, e_), 128, 0, stream>>>(
          static_cast<const uint8_t*>(x.data_ptr()), static_cast<const uint8_t*>(xsf.data_ptr()),
          expert_ids, offsets, sf_offsets, mapping, row_experts, gx, sfx, scale, s_, h_, k_, e_,
          swizzled_);
    } else if (s_ <= 512 && ((e_ == 128 && h_ == 2048 && i_ == 768 && k_ == 8) ||
                             (e_ == 64 && h_ == 2048 && i_ == 1408 && k_ == 6 && s_ > 8))) {
      route_small_parallel<<<std::max(s_, e_), 128, 0, stream>>>(
          static_cast<const uint8_t*>(x.data_ptr()), static_cast<const uint8_t*>(xsf.data_ptr()),
          expert_ids, offsets, sf_offsets, mapping, row_experts, gx, sfx, scale, s_, h_, k_, e_,
          swizzled_);
    } else if (s_ <= 512 && e_ <= 256) {
      route_small<<<std::max(s_, e_), 128, 0, stream>>>(
          static_cast<const uint8_t*>(x.data_ptr()), static_cast<const uint8_t*>(xsf.data_ptr()),
          expert_ids, offsets, sf_offsets, mapping, row_experts, gx, sfx, scale, s_, h_, k_, e_,
          swizzled_);
    } else {
      checked(cudaMemsetAsync(counts, 0, e_ * 4, stream));
      if (s_ >= 16384 && e_ >= 32) {
        histogram_local<<<std::min<int64_t>((s_ + 255) / 256, 1024), 256, 0, stream>>>(
            expert_ids, counts, s_, e_);
      } else {
        histogram<<<std::min<int64_t>((s_ + 255) / 256, 1024), 256, 0, stream>>>(expert_ids, counts,
                                                                                 s_, e_);
      }
      prefix<<<1, 128, 0, stream>>>(counts, offsets, cursors, sf_offsets, e_, scale);
      if (s_ >= 8192 && e_ <= 128) {
        auto inverse = input_fused_
                           ? reinterpret_cast<int32_t*>(gx)
                           : (use_tiled_input_gather() ? reinterpret_cast<int32_t*>(qm) : nullptr);
        assign_rows<true><<<(s_ + 255) / 256, 256, 0, stream>>>(expert_ids, cursors, mapping,
                                                                row_experts, s_, e_, inverse, k_);
        if (input_fused_) {
          if (swizzled_)
            linearize_input_scales<<<std::min<int64_t>((t_ * (h_ / 64) + 255) / 256, 4096), 256, 0,
                                     stream>>>(static_cast<const uint8_t*>(xsf.data_ptr()), sfx, t_,
                                               h_);
        } else if (use_tiled_input_gather()) {
          // FC1 has not produced quantized output yet. Its buffer can hold the
          // inverse map even when fused FC1 removes the BF16 intermediate.
          launch_input_roles(x.data_ptr(), gx, static_cast<const uint8_t*>(xsf.data_ptr()), sfx,
                             expert_ids, mapping, inverse, offsets, sf_offsets, t_, h_ / 32, s_, e_,
                             k_, h_ / 16, (s_ + 127) / 128 + e_, swizzled_, stream);
        } else {
          gather_copy<true><<<std::min<int64_t>((s_ + 3) / 4, 4096), 128, 0, stream>>>(
              static_cast<uint8_t*>(x.data_ptr()), static_cast<uint8_t*>(xsf.data_ptr()),
              expert_ids, offsets, sf_offsets, mapping, gx, sfx, s_, h_, k_, e_, swizzled_);
        }
      } else if (s_ >= 8192) {
        gather<<<std::min<int64_t>((s_ + 3) / 4, 4096), 128, 0, stream>>>(
            static_cast<uint8_t*>(x.data_ptr()), static_cast<uint8_t*>(xsf.data_ptr()), expert_ids,
            offsets, sf_offsets, cursors, mapping, row_experts, gx, sfx, s_, h_, k_, e_, swizzled_);
      } else {
        gather_original<<<std::min<int64_t>(s_, 4096), 128, 0, stream>>>(
            static_cast<uint8_t*>(x.data_ptr()), static_cast<uint8_t*>(xsf.data_ptr()), expert_ids,
            offsets, sf_offsets, cursors, mapping, row_experts, gx, sfx, s_, h_, k_, e_, swizzled_);
      }
    }
    checked(cudaGetLastError());

    int64_t xshape[]{s_, h_ / 2, 1}, qshape[]{s_, i_ / 2, 1};
    int64_t xstride[]{h_ / 2, 1, s_ * h_ / 2}, qstride[]{i_ / 2, 1, s_ * i_ / 2};
    int64_t mshape[]{s_, i_, 1}, mstride[]{i_, 1, s_ * i_};
    int64_t yshape[]{s_, h_, 1}, ystride[]{h_, 1, s_ * h_};
    int64_t w1shape[]{i_, h_ / 2, e_}, w1stride[]{h_ / 2, 1, (gated_ ? 2 : 1) * i_ * h_ / 2};
    int64_t w2shape[]{h_, i_ / 2, e_}, w2stride[]{i_ / 2, 1, h_ * i_ / 2};
    int64_t sf1shape[]{i_ * h_ / 16, 1, e_}, sf1stride[]{1, 1, i_ * h_ / 16};
    int64_t sf2shape[]{h_ * i_ / 16, 1, e_}, sf2stride[]{1, 1, h_ * i_ / 16};
    int64_t sfxshape[]{sf_rows_ * h_ / 16, 1, 1}, sfxstride[]{1, 1, 1};
    int64_t sfmshape[]{sf_rows_ * i_ / 16, 1, 1};
    int64_t eshape[]{e_}, dshape[]{int64_t(scratch1_ / 8)}, unit[]{1};
    int64_t scalar_shape[]{1, 1, 1}, scalar_stride[]{1, 1, 1};
    DLTensor tx{input_fused_ ? x.data_ptr() : gx, device_, 3, dl_fp4, xshape, xstride, 0};
    DLTensor tm{mid, device_, 3, dl_bfloat16, mshape, mstride, 0};
    DLTensor tqm{qm, device_, 3, dl_fp4, qshape, qstride, 0};
    DLTensor quantized_mid{qm, device_, 3, dl_int8, qshape, qstride, 0};
    DLTensor ty{gy, device_, 3, dl_bfloat16, yshape, ystride, 0};
    DLTensor up{w1.data_ptr(), device_, 3, dl_fp4, w1shape, w1stride, 0};
    DLTensor gate = up;
    gate.data = static_cast<uint8_t*>(w1.data_ptr()) + (gated_ ? i_ * h_ / 2 : 0);
    DLTensor down{w2.data_ptr(), device_, 3, dl_fp4, w2shape, w2stride, 0};
    DLTensor sf_up{sf1.data_ptr(), device_, 3, dl_float8_e4m3fn, sf1shape, sf1stride, 0};
    DLTensor sf_gate = sf_up;
    sf_gate.data = static_cast<uint8_t*>(sf1.data_ptr()) + (gated_ ? e_ * i_ * h_ / 16 : 0);
    DLTensor sf_down{sf2.data_ptr(), device_, 3, dl_float8_e4m3fn, sf2shape, sf2stride, 0};
    DLTensor sf_x{sfx, device_, 3, dl_float8_e4m3fn, sfxshape, sfxstride, 0};
    DLTensor sf_mid{sfm, device_, 3, dl_float8_e4m3fn, sfmshape, sfxstride, 0};
    int64_t sf_out_shape[]{sf_rows_, i_ / 16, 1};
    int64_t sf_out_stride[]{i_ / 16, 1, sf_rows_ * i_ / 16};
    DLTensor sf_out{sfm, device_, 3, dl_float8_e4m3fn, sf_out_shape, sf_out_stride, 0};
    DLTensor quant_scale{global2.data_ptr(), device_,       3, dl_float32,
                         scalar_shape,       scalar_stride, 0};
    DLTensor first{offsets, device_, 1, dl_int32, eshape, unit, 0};
    DLTensor desc{scratch, device_, 1, dl_int64, dshape, unit, 0};
    DLTensor factors[3];
    for (int j = 0; j < 3; ++j)
      factors[j] = DLTensor{scale + j, device_, 3, dl_float32, scalar_shape, scalar_stride, 0};
    int64_t group_shape[]{e_, 1, 1};
    DLTensor alpha_first{alpha1.data_ptr(), device_, 3, dl_float32, group_shape, scalar_stride, 0};
    DLTensor alpha_second{alpha2.data_ptr(), device_, 3, dl_float32, group_shape, scalar_stride, 0};
    int64_t mshape_sw[]{i_, s_, 1}, mstride_sw[]{1, i_, s_ * i_};
    int64_t yshape_sw[]{h_, s_, 1}, ystride_sw[]{1, h_, s_ * h_};
    DLTensor tm_sw{mid, device_, 3, dl_bfloat16, mshape_sw, mstride_sw, 0};
    DLTensor ty_sw{gy, device_, 3, dl_bfloat16, yshape_sw, ystride_sw, 0};
    // The frozen host resets its scheduler counter on every invocation, and
    // the kernel initializes operand/scale/output descriptors before use.
    // Own the TensorView descriptors until the borrowed AnyView arguments return.
    std::array<TensorView, 16> tensors{
        TensorView(&first),
        TensorView(&desc),
        TensorView(swap1_ ? &gate : &tx),
        TensorView(swap1_ ? (gated_ ? &up : &tx) : &gate),
        TensorView(swap1_ ? &tx : &up),
        TensorView(swap1_ ? &sf_gate : &sf_x),
        TensorView(swap1_ ? (gated_ ? &sf_up : &sf_x) : &sf_gate),
        TensorView(swap1_ ? &sf_x : &sf_up),
        TensorView(quantized_ && !swap_quantized_ ? &quantized_mid : (swap1_ ? &tm_sw : &tm)),
        TensorView(&factors[0]),
        TensorView(&factors[1]),
        TensorView(&factors[2]),
        TensorView(&alpha_first),
        TensorView(&alpha_first),
        TensorView(&sf_out),
        TensorView(&quant_scale)};
    int64_t input_rows_shape[]{s_}, original_sf_shape[]{t_ * h_ / 16};
    DLTensor input_rows_tensor{gx, device_, 1, dl_int32, input_rows_shape, unit, 0};
    DLTensor original_sf_tensor{
        swizzled_ ? sfx : xsf.data_ptr(), device_, 1, dl_uint8, original_sf_shape, unit, 0};
    TensorView input_rows_view(&input_rows_tensor), original_sf_view(&original_sf_tensor);
    tvm::ffi::AnyView args[23];
    int argc = 0;
    args[argc++] = problem1_;
    args[argc++] = tensors[0];
    args[argc++] = tensors[1];
    for (int j = 0; j < (gated_ ? 3 : 2); ++j) args[argc++] = tensors[j + 2];
    for (int j = 0; j < (gated_ ? 3 : 2); ++j) args[argc++] = tensors[j + 5];
    for (auto slot : tail1_) {
      args[argc++] = tensors[slot + 8];
      if (input_fused_ && slot == 1) {
        args[argc++] = input_rows_view;
        args[argc++] = int32_t(t_);
        args[argc++] = original_sf_view;
        args[argc++] = int32_t(0);
      }
    }
    int64_t swap_qshape[]{s_, i_ / 8, 1}, swap_qstride[]{i_ / 8, 1, s_ * i_ / 8};
    DLTensor swap_q{qm, device_, 3, DLDataType{kDLUInt, 32, 1}, swap_qshape, swap_qstride, 0};
    TensorView quantized_mid_view(&swap_q), swap_sf_view(&sf_mid);
    if (swap_quantized_) {
      args[argc++] = quantized_mid_view;
      args[argc++] = swap_sf_view;
      args[argc++] = tensors[15];
    }
    args[argc++] = static_cast<void*>(stream);
    tvm::ffi::Any result;
    fc1_.CallPacked(args, argc, &result);
    if (!quantized_) {
      if (s_ <= 8) {
        requantize<true><<<std::min<int64_t>(s_ * ((i_ + 1023) / 1024), 4096), 128, 0, stream>>>(
            mid, row_experts, offsets, sf_offsets, qm, sfm, static_cast<float*>(global2.data_ptr()),
            s_, i_);
      } else {
        if (s_ >= 8192) {
          if (gated_ && t_ <= 12288 && e_ == 128 && h_ == 2048 && i_ == 768 && k_ == 8) {
            requantize_blockwise<48><<<(s_ * (i_ / 16) + 255) / 256, 256, 0, stream>>>(
                mid, row_experts, offsets, sf_offsets, static_cast<float*>(global2.data_ptr()), qm,
                sfm, s_, i_);
          } else if (gated_ && t_ <= 12288 && e_ == 64 && h_ == 2048 && i_ == 1408 && k_ == 6) {
            requantize_blockwise<88><<<(s_ * (i_ / 16) + 255) / 256, 256, 0, stream>>>(
                mid, row_experts, offsets, sf_offsets, static_cast<float*>(global2.data_ptr()), qm,
                sfm, s_, i_);
          } else if (gated_ && t_ <= 12288 && e_ == 12 && h_ == 7168 && i_ == 3072 && k_ == 2) {
            requantize_blockwise<192><<<(s_ * (i_ / 16) + 255) / 256, 256, 0, stream>>>(
                mid, row_experts, offsets, sf_offsets, static_cast<float*>(global2.data_ptr()), qm,
                sfm, s_, i_);
          } else {
            requantize_warp<false><<<std::min<int64_t>((s_ + 3) / 4, 4096), 128, 0, stream>>>(
                mid, row_experts, offsets, sf_offsets, qm, sfm,
                static_cast<float*>(global2.data_ptr()), s_, i_);
          }
        } else {
          requantize<false><<<std::min<int64_t>(s_, 4096), 128, 0, stream>>>(
              mid, row_experts, offsets, sf_offsets, qm, sfm,
              static_cast<float*>(global2.data_ptr()), s_, i_);
        }
      }
    }
    checked(cudaGetLastError());
    if (stages) return;
    dshape[0] = scratch2_ / 8;
    std::array<TensorView, 8> second{TensorView(&first),
                                     TensorView(&desc),
                                     TensorView(swap2_ ? &down : &tqm),
                                     TensorView(swap2_ ? &tqm : &down),
                                     TensorView(swap2_ ? &sf_down : &sf_mid),
                                     TensorView(swap2_ ? &sf_mid : &sf_down),
                                     TensorView(swap2_ ? &ty_sw : &ty),
                                     TensorView(&alpha_second)};
    argc = 0;
    args[argc++] = problem2_;
    for (int j = 0; j < 6; ++j) args[argc++] = second[j];
    for (auto slot : tail2_) args[argc++] = second[slot == 0 ? 6 : 7];
    args[argc++] = static_cast<void*>(stream);
    fc2_.CallPacked(args, argc, &result);
    const auto finalize_path = finalize_variant();
    if (finalize_path == 3) {
      frost_moe_finalize::launch_large(
          gy, expert_ids, mapping, static_cast<float*>(scores.data_ptr()),
          static_cast<__nv_bfloat16*>(out.data_ptr()), t_, e_, k_, stream);
    } else if (finalize_path == 1) {
      frost_moe_finalize::launch_small(
          gy, expert_ids, mapping, static_cast<float*>(scores.data_ptr()),
          static_cast<__nv_bfloat16*>(out.data_ptr()), t_, h_, e_, k_, stream);
    } else if (finalize_path == 2) {
      const int vectors = h_ / 8;
      dim3 grid((vectors + 511) / 512, t_);
      finalize_vec_kernel<2, 4, 128, true><<<grid, 128, 0, stream>>>(
          gy, expert_ids, mapping, static_cast<float*>(scores.data_ptr()), e_,
          static_cast<__nv_bfloat16*>(out.data_ptr()), h_, vectors);
    } else if (t_ > 8 && t_ <= 512 && (k_ == 2 || k_ == 6 || k_ == 8) && h_ % 256 == 0) {
      const int grid = t_ * ((h_ / 8 + 255) / 256);
      if (k_ == 2) {
        finalize_vec8<2><<<grid, 256, 0, stream>>>(
            gy, expert_ids, mapping, static_cast<float*>(scores.data_ptr()), e_,
            static_cast<__nv_bfloat16*>(out.data_ptr()), t_, h_);
      } else if (k_ == 6) {
        finalize_vec8<6><<<grid, 256, 0, stream>>>(
            gy, expert_ids, mapping, static_cast<float*>(scores.data_ptr()), e_,
            static_cast<__nv_bfloat16*>(out.data_ptr()), t_, h_);
      } else {
        finalize_vec8<8><<<grid, 256, 0, stream>>>(
            gy, expert_ids, mapping, static_cast<float*>(scores.data_ptr()), e_,
            static_cast<__nv_bfloat16*>(out.data_ptr()), t_, h_);
      }
    } else if (t_ > 512 && t_ <= 65535 && (k_ == 2 || k_ == 6 || k_ == 8) && h_ % 256 == 0) {
      const int vectors = h_ / 8;
      if (k_ == 2) {
        dim3 grid((vectors + 895) / 896, t_);
        finalize_vec_kernel<2, 7, 128><<<grid, 128, 0, stream>>>(
            gy, expert_ids, mapping, static_cast<float*>(scores.data_ptr()), e_,
            static_cast<__nv_bfloat16*>(out.data_ptr()), h_, vectors);
      } else if (k_ == 6) {
        dim3 grid((vectors + 255) / 256, t_);
        finalize_vec_kernel<6, 2, 128><<<grid, 128, 0, stream>>>(
            gy, expert_ids, mapping, static_cast<float*>(scores.data_ptr()), e_,
            static_cast<__nv_bfloat16*>(out.data_ptr()), h_, vectors);
      } else {
        dim3 grid((vectors + 255) / 256, t_);
        finalize_vec_kernel<8, 2, 128><<<grid, 128, 0, stream>>>(
            gy, expert_ids, mapping, static_cast<float*>(scores.data_ptr()), e_,
            static_cast<__nv_bfloat16*>(out.data_ptr()), h_, vectors);
      }
    } else if (t_ <= 8 && (k_ == 2 || k_ == 6 || k_ == 8) && h_ % 128 == 0) {
      if (k_ == 2) {
        dim3 grid((h_ / 8 + 63) / 64, t_);
        finalize_vec_kernel<2, 1, 64><<<grid, 64, 0, stream>>>(
            gy, expert_ids, mapping, static_cast<float*>(scores.data_ptr()), e_,
            static_cast<__nv_bfloat16*>(out.data_ptr()), h_, h_ / 8);
      } else if (k_ == 6) {
        dim3 grid((h_ / 8 + 31) / 32, t_);
        finalize_vec_kernel<6, 1, 32><<<grid, 32, 0, stream>>>(
            gy, expert_ids, mapping, static_cast<float*>(scores.data_ptr()), e_,
            static_cast<__nv_bfloat16*>(out.data_ptr()), h_, h_ / 8);
      } else {
        dim3 grid((h_ / 8 + 31) / 32, t_);
        finalize_vec_kernel<8, 1, 32><<<grid, 32, 0, stream>>>(
            gy, expert_ids, mapping, static_cast<float*>(scores.data_ptr()), e_,
            static_cast<__nv_bfloat16*>(out.data_ptr()), h_, h_ / 8);
      }
    } else if (t_ <= 8) {
      finalize<true><<<std::min<int64_t>(t_ * ((h_ / 8 + 127) / 128), 4096), 128, 0, stream>>>(
          gy, expert_ids, mapping, static_cast<float*>(scores.data_ptr()),
          static_cast<__nv_bfloat16*>(out.data_ptr()), t_, h_, k_, e_);
    } else {
      finalize<false><<<std::min<int64_t>(t_, 4096), 128, 0, stream>>>(
          gy, expert_ids, mapping, static_cast<float*>(scores.data_ptr()),
          static_cast<__nv_bfloat16*>(out.data_ptr()), t_, h_, k_, e_);
    }
    checked(cudaGetLastError());
  }

  Function fc1_, fc2_;
  int64_t t_, h_, i_, e_, k_, s_, sf_rows_;
  DLDevice device_;
  size_t scratch1_, scratch2_;
  bool gated_;
  Array<int64_t> tail1_, tail2_;
  bool quantized_, swap1_, swap2_, swizzled_, fma_, input_fused_, swap_quantized_;
  Array<int64_t> problem1_, problem2_;
  size_t x_pos_, mid_pos_, qmid_pos_, sf1_pos_, sf2_pos_, counts_pos_, offsets_pos_,
      sf_offsets_pos_, cursors_pos_, mapping_pos_, row_experts_pos_, scale_pos_, scratch_pos_,
      workspace_size_;
};

Module make_plan(Function fc1, Function fc2, int64_t tokens, int64_t hidden, int64_t intermediate,
                 int64_t experts, int64_t topk, int64_t device, int64_t scratch1, int64_t scratch2,
                 bool gated, Array<int64_t> tail1, Array<int64_t> tail2, bool swap1, bool swap2,
                 bool swizzled, bool fma, bool input_fused, bool swap_quantized) {
  constexpr int64_t max_dim = 1 << 20;
  TVM_FFI_ICHECK(tokens > 0 && tokens <= max_dim && hidden > 0 && hidden <= max_dim &&
                 intermediate > 0 && intermediate <= max_dim && experts > 0 && experts <= 1024 &&
                 topk > 0 && topk <= experts);
  TVM_FFI_ICHECK_LT(tokens * topk, std::numeric_limits<int32_t>::max() - 128 * experts);
  TVM_FFI_ICHECK_EQ(hidden % 128, 0);
  TVM_FFI_ICHECK_EQ(intermediate % 128, 0);
  if (swap_quantized) {
    TVM_FFI_ICHECK(!fma && !input_fused && swap1 && gated &&
                   tokens <= ((experts == 8 || experts == 12) ? 512 : 128));
    TVM_FFI_ICHECK((experts == 64 && hidden == 2048 && intermediate == 1408 && topk == 6) ||
                   (experts == 12 && hidden == 7168 && intermediate == 3072 && topk == 2) ||
                   (experts == 128 && hidden == 2048 && intermediate == 768 && topk == 8) ||
                   (experts == 8 && hidden == 4096 && intermediate == 14336 && topk == 2));
    TVM_FFI_ICHECK(std::find(tail1.begin(), tail1.end(), 6) == tail1.end());
  }
  if (input_fused) {
    TVM_FFI_ICHECK(!fma && !swap1 && gated && experts == 128 && hidden == 2048 &&
                   intermediate == 768 && topk == 8 && tokens >= 8192 && tokens <= 12288);
    TVM_FFI_ICHECK(std::find(tail1.begin(), tail1.end(), 6) != tail1.end());
  }
  if (fma) {
    TVM_FFI_ICHECK(!swap1 && !swap2 && scratch1 == 0 && scratch2 == 0 && tail1.empty() &&
                   tail2.empty());
    TVM_FFI_ICHECK(
        (experts == 128 && hidden == 2048 && intermediate == 768 && topk == 8 && tokens <= 4) ||
        (experts == 64 && hidden == 2048 && intermediate == 1408 && topk == 6 && tokens <= 4) ||
        (experts == 12 && hidden == 7168 && intermediate == 3072 && topk == 2 && tokens == 1));
    return Module(tvm::ffi::make_object<CudnnFrostNvfp4MoePlan>(
        std::move(fc1), std::move(fc2), tokens, hidden, intermediate, experts, topk, device,
        scratch1, scratch2, gated, std::move(tail1), std::move(tail2), swap1, swap2, swizzled, true,
        false, false));
  }
  TVM_FFI_ICHECK(scratch1 > 0 && scratch1 % 128 == 0 && scratch2 > 0 && scratch2 % 128 == 0);
  int mask = 0;
  for (auto slot : tail1) {
    TVM_FFI_ICHECK(slot >= 0 && slot <= 7);
    TVM_FFI_ICHECK_EQ(mask & (1 << slot), 0);
    mask |= 1 << slot;
  }
  int required = 3 | (1 << 4) | (gated ? (1 << 5) : 0);
  if (mask & (1 << 6)) {
    required |= (1 << 6) | (1 << 7);
    TVM_FFI_ICHECK(!swap1);
  }
  TVM_FFI_ICHECK(mask == required || mask == (required | 12));
  TVM_FFI_ICHECK(tail2.size() == 2 &&
                 ((tail2[0] == 0 && tail2[1] == 4) || (tail2[0] == 4 && tail2[1] == 0)));
  return Module(tvm::ffi::make_object<CudnnFrostNvfp4MoePlan>(
      std::move(fc1), std::move(fc2), tokens, hidden, intermediate, experts, topk, device, scratch1,
      scratch2, gated, std::move(tail1), std::move(tail2), swap1, swap2, swizzled, false,
      input_fused, swap_quantized));
}
}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(make_plan, make_plan);
