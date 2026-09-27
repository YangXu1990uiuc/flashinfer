// Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
// Routing, block-scale packing, frozen Frost GEMMs, and weighted finalization.
#include <cuda_bf16.h>
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
constexpr DLDataType dl_e8m0{kDLFloat8_e8m0fnu, 8, 1};
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
template <int K, int U, int BD>
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
      for (int j = 0; j < K; ++j) accum_vec(acc, raw[u][j], w[j]);
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
                            int32_t* row_experts, int rows, int experts) {
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
  }
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
    auto source = reinterpret_cast<const int4*>(x) + (r / topk) * (hidden / 16);
    auto target = reinterpret_cast<int4*>(grouped) + int64_t(dest) * (hidden / 16);
    for (int h = lane; h < hidden / 16; h += 32)
      target[h] = valid ? source[h] : make_int4(0, 0, 0, 0);
    const int cols = hidden / 32;
    if constexpr (PackedScale) {
      for (int col = lane * 4; col < cols; col += 128) {
        int64_t src = swizzled ? sf_index(r / topk, col, cols) : (r / topk) * cols + col;
        int64_t dst = int64_t(sf_offsets[e]) * cols + sf_index(dest - offsets[e], col, cols);
        *reinterpret_cast<uint32_t*>(sf + dst) =
            valid ? *reinterpret_cast<const uint32_t*>(input_sf + src) : 0x7f7f7f7fu;
      }
    } else {
      for (int col = lane; col < cols; col += 32) {
        int64_t src = swizzled ? sf_index(r / topk, col, cols) : (r / topk) * cols + col;
        int64_t dst = int64_t(sf_offsets[e]) * cols + sf_index(dest - offsets[e], col, cols);
        sf[dst] = valid ? input_sf[src] : 127;
      }
    }
  }
}

__global__ void gather(const uint8_t* x, const uint8_t* input_sf, const int32_t* ids,
                       const int32_t* offsets, const int32_t* sf_offsets, int32_t* cursors,
                       int32_t* mapping, int32_t* row_experts, uint8_t* grouped, uint8_t* sf,
                       int rows, int hidden, int topk, int experts, bool swizzled) {
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
    auto source = reinterpret_cast<const int4*>(x) + (r / topk) * (hidden / 16);
    auto target = reinterpret_cast<int4*>(grouped) + int64_t(dest) * (hidden / 16);
    for (int h = threadIdx.x; h < hidden / 16; h += blockDim.x)
      target[h] = valid ? source[h] : make_int4(0, 0, 0, 0);
    int cols = hidden / 32;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
      int64_t src = swizzled ? sf_index(r / topk, col, cols) : (r / topk) * cols + col;
      int64_t dst = int64_t(sf_offsets[e]) * cols + sf_index(dest - offsets[e], col, cols);
      sf[dst] = valid ? input_sf[src] : 127;
    }
    __syncthreads();
  }
}

__global__ void gather_warp(const uint8_t* x, const uint8_t* input_sf, const int32_t* ids,
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
    auto source = reinterpret_cast<const int4*>(x) + (r / topk) * (hidden / 16);
    auto target = reinterpret_cast<int4*>(grouped) + int64_t(dest) * (hidden / 16);
    for (int h = lane; h < hidden / 16; h += 32)
      target[h] = valid ? source[h] : make_int4(0, 0, 0, 0);
    int cols = hidden / 32;
    for (int col = lane; col < cols; col += 32) {
      int64_t src = swizzled ? sf_index(r / topk, col, cols) : (r / topk) * cols + col;
      int64_t dst = int64_t(sf_offsets[e]) * cols + sf_index(dest - offsets[e], col, cols);
      sf[dst] = valid ? input_sf[src] : 127;
    }
  }
}

// Small batches compute stable routing and expert-local scale segments per CTA.
// No CTA consumes metadata written by another CTA in this launch.
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
    auto source = reinterpret_cast<const int4*>(x) + (r / topk) * (hidden / 16);
    auto target = reinterpret_cast<int4*>(grouped) + int64_t(dest) * (hidden / 16);
    for (int h = threadIdx.x; h < hidden / 16; h += blockDim.x)
      target[h] = valid ? source[h] : make_int4(0, 0, 0, 0);
    const int cols = hidden / 32;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
      const int64_t src = swizzled ? sf_index(r / topk, col, cols) : (r / topk) * cols + col;
      const int64_t dst = int64_t(sfbegin) * cols + sf_index(dest - begin, col, cols);
      sf[dst] = valid ? input_sf[src] : 127;
    }
  }
}

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
    auto source = reinterpret_cast<const int4*>(x) + (r / topk) * (hidden / 16);
    auto target = reinterpret_cast<int4*>(grouped) + int64_t(destination) * (hidden / 16);
    for (int h = threadIdx.x; h < hidden / 16; h += blockDim.x)
      target[h] = valid ? source[h] : make_int4(0, 0, 0, 0);
    const int cols = hidden / 32;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
      int64_t src = swizzled ? sf_index(r / topk, col, cols) : (r / topk) * cols + col;
      int64_t dst = int64_t(row_sf_begin) * cols + sf_index(destination - row_begin, col, cols);
      sf[dst] = valid ? input_sf[src] : 127;
    }
  }
}

template <bool SplitColumns = false>
__global__ void requantize(const __nv_bfloat16* input, const int32_t* row_experts,
                           const int32_t* offsets, const int32_t* sf_offsets, uint8_t* output,
                           uint8_t* scales, int rows, int width) {
  union InputPack {
    int4 words;
    __nv_bfloat16 values[8];
  };
  union OutputPack {
    uint64_t words;
    uint8_t values[8];
  };
  // Column tiles contain whole 32-element blocks, preserving lane reductions
  // and the original quantization rounding for small batches.
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
      // Four adjacent lanes own one 32-element microscaling block.
      auto mask = __activemask();
      maximum = fmaxf(maximum, __shfl_xor_sync(mask, maximum, 1));
      maximum = fmaxf(maximum, __shfl_xor_sync(mask, maximum, 2));
      __nv_fp8_e8m0 sf;
      sf.__x = __nv_cvt_float_to_e8m0(maximum / 448.f, __NV_SATFINITE, cudaRoundPosInf);
      float scale = static_cast<float>(sf);
      float inverse = maximum == 0.f || scale == 0.f ? 0.f : 1.f / scale;
      if (threadIdx.x % 4 == 0) {
        int cols = width / 32;
        scales[int64_t(sf_offsets[e]) * cols + sf_index(row - offsets[e], col / 32, cols)] = sf.__x;
      }
      OutputPack out;
#pragma unroll
      for (int j = 0; j < 8; ++j)
        out.values[j] = __nv_cvt_float_to_fp8(values[j] * inverse, __NV_SATFINITE, __NV_E4M3);
      reinterpret_cast<uint64_t*>(output)[(row * width + col) / 8] = out.words;
    }
  }
}

template <bool SplitColumns = false>
__global__ void requantize_warp(const __nv_bfloat16* input, const int32_t* row_experts,
                                const int32_t* offsets, const int32_t* sf_offsets, uint8_t* output,
                                uint8_t* scales, int rows, int width) {
  union InputPack {
    int4 words;
    __nv_bfloat16 values[8];
  };
  union OutputPack {
    uint64_t words;
    uint8_t values[8];
  };
  // Column tiles contain whole 32-element blocks, preserving lane reductions
  // and the original quantization rounding for small batches.
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
      // Four adjacent lanes own one 32-element microscaling block.
      auto mask = __activemask();
      maximum = fmaxf(maximum, __shfl_xor_sync(mask, maximum, 1));
      maximum = fmaxf(maximum, __shfl_xor_sync(mask, maximum, 2));
      __nv_fp8_e8m0 sf;
      sf.__x = __nv_cvt_float_to_e8m0(maximum / 448.f, __NV_SATFINITE, cudaRoundPosInf);
      float scale = static_cast<float>(sf);
      float inverse = maximum == 0.f || scale == 0.f ? 0.f : 1.f / scale;
      if (threadIdx.x % 4 == 0) {
        int cols = width / 32;
        scales[int64_t(sf_offsets[e]) * cols + sf_index(row - offsets[e], col / 32, cols)] = sf.__x;
      }
      OutputPack out;
#pragma unroll
      for (int j = 0; j < 8; ++j)
        out.values[j] = __nv_cvt_float_to_fp8(values[j] * inverse, __NV_SATFINITE, __NV_E4M3);
      reinterpret_cast<uint64_t*>(output)[(row * width + col) / 8] = out.words;
    }
  }
}

class CudnnFrostMxfp8Mxfp4MoePlan final : public tvm::ffi::ModuleObj {
 public:
  CudnnFrostMxfp8Mxfp4MoePlan(Function fc1, Function fc2, int64_t tokens, int64_t hidden,
                              int64_t intermediate, int64_t experts, int64_t topk, int device,
                              size_t scratch1, size_t scratch2, bool gated, Array<int64_t> tail,
                              bool swap1, bool swap2, bool swizzled)
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
        tail_(std::move(tail)),
        swap1_(swap1),
        swap2_(swap2),
        swizzled_(swizzled) {
    int64_t active = std::min(s_, e_);
    sf_rows_ = 128 * (active + (s_ - active) / 128);
    size_t pos = 0;
    auto reserve = [&](size_t bytes) {
      size_t start = pos;
      pos += align128(bytes);
      return start;
    };
    // FC2 can overwrite the grouped input after FC1 has consumed it.
    x_pos_ = reserve(s_ * h_ * 2);
    mid_pos_ = reserve(s_ * i_ * 2);
    qmid_pos_ = reserve(s_ * i_);
    sf1_pos_ = reserve(sf_rows_ * h_ / 32);
    sf2_pos_ = reserve(sf_rows_ * i_ / 32);
    counts_pos_ = reserve(e_ * 4);
    offsets_pos_ = reserve(e_ * 4);
    sf_offsets_pos_ = reserve(e_ * 4);
    cursors_pos_ = reserve(e_ * 4);
    mapping_pos_ = reserve(s_ * 4);
    row_experts_pos_ = reserve(s_ * 4);
    scale_pos_ = reserve(3 * sizeof(float));
    scratch_pos_ = reserve(std::max(scratch1_, scratch2_));
    workspace_size_ = pos;
    auto problem = [&](int64_t n, int64_t k, bool swap, bool gated) {
      Array<int64_t> shape{swap ? n : s_, swap ? s_ : n, k, e_, e_};
      auto token = [&]() {
        shape.push_back(k);
        shape.push_back(1);
        shape.push_back(s_ * k);
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
      shape.push_back(swap ? 1 : n);
      shape.push_back(swap ? n : 1);
      shape.push_back(s_ * n);
      return shape;
    };
    problem1_ = problem(i_, h_, swap1_, gated_);
    problem2_ = problem(h_, i_, swap2_, false);
  }

  const char* kind() const final { return "cudnn_frost_mxfp8_mxfp4_moe_plan"; }
  Optional<Function> GetFunction(const tvm::ffi::String& name) final {
    if (name == "finalize_variant")
      return Function::FromTyped([this]() { return finalize_variant(); });
    if (name == "workspace_size")
      return Function::FromTyped([this]() { return int64_t(workspace_size_); });
    if (name == "stage_layout")
      return Function::FromTyped([this]() {
        return Array<int64_t>{int64_t(x_pos_),   int64_t(mid_pos_), int64_t(qmid_pos_),
                              int64_t(sf1_pos_), int64_t(sf2_pos_), int64_t(offsets_pos_),
                              sf_rows_};
      });
    if (name == "run" || name == "prepare_stages") {
      bool stages = name == "prepare_stages";
      return Function::FromTyped([this, stages](TensorView out, TensorView x, TensorView ids,
                                                TensorView scores, TensorView w1, TensorView w2,
                                                TensorView sf1, TensorView sf2, TensorView xsf,
                                                TensorView workspace) {
        run(out, x, ids, scores, w1, w2, sf1, sf2, xsf, workspace, stages);
      });
    }
    return Function(nullptr);
  }

 private:
  int64_t finalize_variant() const {
    if (frost_moe_finalize::small_supported(t_, h_, i_, e_, k_)) return 1;
    return 0;
  }
  void run(TensorView out, TensorView x, TensorView ids, TensorView scores, TensorView w1,
           TensorView w2, TensorView sf1, TensorView sf2, TensorView xsf, TensorView workspace,
           bool stages) const {
    tensor(out, device_, dl_bfloat16, {t_, h_});
    tensor(x, device_, dl_float8_e4m3fn, {t_, h_});
    tensor(ids, device_, dl_int32, {t_, k_}, 4);
    tensor(scores, device_, dl_float32, {t_, k_}, 4);
    tensor(w1, device_, dl_uint8, {e_, (gated_ ? 2 : 1) * i_, h_ / 2});
    tensor(w2, device_, dl_uint8, {e_, h_, i_ / 2});
    tensor(sf1, device_, dl_uint8, {(gated_ ? 2 : 1), e_, i_, h_ / 32});
    tensor(sf2, device_, dl_uint8, {e_, h_, i_ / 32});
    if (swizzled_)
      tensor(xsf, device_, dl_uint8, {(t_ + 127) / 128 * 128 * h_ / 32});
    else
      tensor(xsf, device_, dl_uint8, {t_, h_ / 32});
    tensor(workspace, device_, dl_uint8, {workspace.numel()}, 128);
    TVM_FFI_ICHECK_GE(workspace.numel(), workspace_size_);
    ffi::CUDADeviceGuard guard(device_.device_id);
    auto stream = get_stream(device_);
    auto base = static_cast<char*>(workspace.data_ptr());
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
    if (s_ <= 8 && e_ == 64) {
      route_tiny<<<std::max(s_, e_), 128, 0, stream>>>(
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
      if (s_ >= 8192 && e_ >= 32 && e_ <= 128) {
        assign_rows<true><<<(s_ + 255) / 256, 256, 0, stream>>>(expert_ids, cursors, mapping,
                                                                row_experts, s_, e_);
        gather_copy<true><<<std::min<int64_t>((s_ + 3) / 4, 4096), 128, 0, stream>>>(
            static_cast<uint8_t*>(x.data_ptr()), static_cast<uint8_t*>(xsf.data_ptr()), expert_ids,
            offsets, sf_offsets, mapping, gx, sfx, s_, h_, k_, e_, swizzled_);
      } else if (s_ >= 8192) {
        gather_warp<<<std::min<int64_t>((s_ + 3) / 4, 4096), 128, 0, stream>>>(
            static_cast<uint8_t*>(x.data_ptr()), static_cast<uint8_t*>(xsf.data_ptr()), expert_ids,
            offsets, sf_offsets, cursors, mapping, row_experts, gx, sfx, s_, h_, k_, e_, swizzled_);
      } else {
        gather<<<std::min<int64_t>(s_, 4096), 128, 0, stream>>>(
            static_cast<uint8_t*>(x.data_ptr()), static_cast<uint8_t*>(xsf.data_ptr()), expert_ids,
            offsets, sf_offsets, cursors, mapping, row_experts, gx, sfx, s_, h_, k_, e_, swizzled_);
      }
    }
    checked(cudaGetLastError());

    int64_t xshape[]{s_, h_, 1}, mshape[]{s_, i_, 1};
    int64_t xstride[]{h_, 1, s_ * h_}, mstride[]{i_, 1, s_ * i_};
    int64_t w1shape[]{i_, h_ / 2, e_}, w1stride[]{h_ / 2, 1, (gated_ ? 2 : 1) * i_ * h_ / 2};
    int64_t w2shape[]{h_, i_ / 2, e_}, w2stride[]{i_ / 2, 1, h_ * i_ / 2};
    int64_t sf1shape[]{i_ * h_ / 32, 1, e_}, sf1stride[]{1, 1, i_ * h_ / 32};
    int64_t sf2shape[]{h_ * i_ / 32, 1, e_}, sf2stride[]{1, 1, h_ * i_ / 32};
    int64_t sfxshape[]{sf_rows_ * h_ / 32, 1, 1}, sfxstride[]{1, 1, 1};
    int64_t sfmshape[]{sf_rows_ * i_ / 32, 1, 1};
    int64_t eshape[]{e_}, dshape[]{int64_t(scratch1_ / 8)}, unit[]{1};
    int64_t scalar_shape[]{1, 1, 1}, scalar_stride[]{1, 1, 1};
    DLTensor tx{gx, device_, 3, dl_float8_e4m3fn, xshape, xstride, 0};
    DLTensor tm{mid, device_, 3, dl_bfloat16, mshape, mstride, 0};
    DLTensor tqm{qm, device_, 3, dl_float8_e4m3fn, mshape, mstride, 0};
    DLTensor ty{gy, device_, 3, dl_bfloat16, xshape, xstride, 0};
    DLTensor up{w1.data_ptr(), device_, 3, dl_fp4, w1shape, w1stride, 0};
    DLTensor gate = up;
    gate.data = static_cast<uint8_t*>(w1.data_ptr()) + (gated_ ? i_ * h_ / 2 : 0);
    DLTensor down{w2.data_ptr(), device_, 3, dl_fp4, w2shape, w2stride, 0};
    DLTensor sf_up{sf1.data_ptr(), device_, 3, dl_e8m0, sf1shape, sf1stride, 0};
    DLTensor sf_gate = sf_up;
    sf_gate.data = static_cast<uint8_t*>(sf1.data_ptr()) + (gated_ ? e_ * i_ * h_ / 32 : 0);
    DLTensor sf_down{sf2.data_ptr(), device_, 3, dl_e8m0, sf2shape, sf2stride, 0};
    DLTensor sf_x{sfx, device_, 3, dl_e8m0, sfxshape, sfxstride, 0};
    DLTensor sf_mid{sfm, device_, 3, dl_e8m0, sfmshape, sfxstride, 0};
    DLTensor first{offsets, device_, 1, dl_int32, eshape, unit, 0};
    DLTensor desc{scratch, device_, 1, dl_int64, dshape, unit, 0};
    DLTensor factors[3];
    for (int j = 0; j < 3; ++j)
      factors[j] = DLTensor{scale + j, device_, 3, dl_float32, scalar_shape, scalar_stride, 0};
    int64_t mshape_sw[]{i_, s_, 1}, mstride_sw[]{1, i_, s_ * i_};
    int64_t yshape_sw[]{h_, s_, 1}, ystride_sw[]{1, h_, s_ * h_};
    DLTensor tm_sw{mid, device_, 3, dl_bfloat16, mshape_sw, mstride_sw, 0};
    DLTensor ty_sw{gy, device_, 3, dl_bfloat16, yshape_sw, ystride_sw, 0};
    // The frozen host resets its scheduler counter on every invocation.
    // Its kernels populate descriptor storage before consuming it.
    // Own the TensorView descriptors until the borrowed AnyView arguments return.
    std::array<TensorView, 12> tensors{TensorView(&first),
                                       TensorView(&desc),
                                       TensorView(swap1_ ? &gate : &tx),
                                       TensorView(swap1_ ? (gated_ ? &up : &tx) : &gate),
                                       TensorView(swap1_ ? &tx : &up),
                                       TensorView(swap1_ ? &sf_gate : &sf_x),
                                       TensorView(swap1_ ? (gated_ ? &sf_up : &sf_x) : &sf_gate),
                                       TensorView(swap1_ ? &sf_x : &sf_up),
                                       TensorView(swap1_ ? &tm_sw : &tm),
                                       TensorView(&factors[0]),
                                       TensorView(&factors[1]),
                                       TensorView(&factors[2])};
    tvm::ffi::AnyView args[15];
    int argc = 0;
    args[argc++] = problem1_;
    args[argc++] = tensors[0];
    args[argc++] = tensors[1];
    for (int j = 0; j < (gated_ ? 3 : 2); ++j) args[argc++] = tensors[j + 2];
    for (int j = 0; j < (gated_ ? 3 : 2); ++j) args[argc++] = tensors[j + 5];
    for (auto slot : tail_) args[argc++] = tensors[slot + 9];
    args[argc++] = static_cast<void*>(stream);
    tvm::ffi::Any result;
    fc1_.CallPacked(args, argc, &result);
    if (s_ <= 8) {
      requantize<true><<<std::min<int64_t>(s_ * ((i_ + 1023) / 1024), 4096), 128, 0, stream>>>(
          mid, row_experts, offsets, sf_offsets, qm, sfm, s_, i_);
    } else {
      if (s_ >= 8192) {
        requantize_warp<false><<<std::min<int64_t>((s_ + 3) / 4, 4096), 128, 0, stream>>>(
            mid, row_experts, offsets, sf_offsets, qm, sfm, s_, i_);
      } else {
        requantize<false><<<std::min<int64_t>(s_, 4096), 128, 0, stream>>>(
            mid, row_experts, offsets, sf_offsets, qm, sfm, s_, i_);
      }
    }
    checked(cudaGetLastError());
    if (stages) return;
    dshape[0] = scratch2_ / 8;
    fc2_(problem2_, TensorView(&first), TensorView(&desc), TensorView(swap2_ ? &down : &tqm),
         TensorView(swap2_ ? &tqm : &down), TensorView(swap2_ ? &sf_down : &sf_mid),
         TensorView(swap2_ ? &sf_mid : &sf_down), TensorView(swap2_ ? &ty_sw : &ty),
         static_cast<void*>(stream));
    const auto finalize_path = finalize_variant();
    if (finalize_path == 1) {
      frost_moe_finalize::launch_small(
          gy, expert_ids, mapping, static_cast<float*>(scores.data_ptr()),
          static_cast<__nv_bfloat16*>(out.data_ptr()), t_, h_, e_, k_, stream);
    } else if (t_ > 8 && t_ <= 512 && (k_ == 2 || k_ == 6) && h_ % 256 == 0) {
      const int grid = t_ * ((h_ / 8 + 255) / 256);
      if (k_ == 2) {
        finalize_vec8<2><<<grid, 256, 0, stream>>>(
            gy, expert_ids, mapping, static_cast<float*>(scores.data_ptr()), e_,
            static_cast<__nv_bfloat16*>(out.data_ptr()), t_, h_);
      } else {
        finalize_vec8<6><<<grid, 256, 0, stream>>>(
            gy, expert_ids, mapping, static_cast<float*>(scores.data_ptr()), e_,
            static_cast<__nv_bfloat16*>(out.data_ptr()), t_, h_);
      }
    } else if (t_ > 512 && t_ <= 65535 && (k_ == 2 || k_ == 6) && h_ % 256 == 0) {
      const int vectors = h_ / 8;
      if (k_ == 2) {
        dim3 grid((vectors + 895) / 896, t_);
        finalize_vec_kernel<2, 7, 128><<<grid, 128, 0, stream>>>(
            gy, expert_ids, mapping, static_cast<float*>(scores.data_ptr()), e_,
            static_cast<__nv_bfloat16*>(out.data_ptr()), h_, vectors);
      } else {
        dim3 grid((vectors + 255) / 256, t_);
        finalize_vec_kernel<6, 2, 128><<<grid, 128, 0, stream>>>(
            gy, expert_ids, mapping, static_cast<float*>(scores.data_ptr()), e_,
            static_cast<__nv_bfloat16*>(out.data_ptr()), h_, vectors);
      }
    } else if (t_ <= 8 && (k_ == 2 || k_ == 6) && h_ % 128 == 0) {
      if (k_ == 2) {
        dim3 grid((h_ / 8 + 63) / 64, t_);
        finalize_vec_kernel<2, 1, 64><<<grid, 64, 0, stream>>>(
            gy, expert_ids, mapping, static_cast<float*>(scores.data_ptr()), e_,
            static_cast<__nv_bfloat16*>(out.data_ptr()), h_, h_ / 8);
      } else {
        dim3 grid((h_ / 8 + 31) / 32, t_);
        finalize_vec_kernel<6, 1, 32><<<grid, 32, 0, stream>>>(
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
  Array<int64_t> tail_;
  bool swap1_, swap2_, swizzled_;
  Array<int64_t> problem1_, problem2_;
  size_t x_pos_, mid_pos_, qmid_pos_, sf1_pos_, sf2_pos_, counts_pos_, offsets_pos_,
      sf_offsets_pos_, cursors_pos_, mapping_pos_, row_experts_pos_, scale_pos_, scratch_pos_,
      workspace_size_;
};

Module make_plan(Function fc1, Function fc2, int64_t tokens, int64_t hidden, int64_t intermediate,
                 int64_t experts, int64_t topk, int64_t device, int64_t scratch1, int64_t scratch2,
                 bool gated, Array<int64_t> tail, bool swap1, bool swap2, bool swizzled) {
  constexpr int64_t max_dim = 1 << 20;
  TVM_FFI_ICHECK(tokens > 0 && tokens <= max_dim && hidden > 0 && hidden <= max_dim &&
                 intermediate > 0 && intermediate <= max_dim && experts > 0 && experts <= 1024 &&
                 topk > 0 && topk <= experts);
  TVM_FFI_ICHECK_LT(tokens * topk, std::numeric_limits<int32_t>::max() - 128 * experts);
  TVM_FFI_ICHECK_EQ(hidden % 128, 0);
  TVM_FFI_ICHECK_EQ(intermediate % 128, 0);
  TVM_FFI_ICHECK(scratch1 > 0 && scratch1 % 128 == 0 && scratch2 > 0 && scratch2 % 128 == 0);
  TVM_FFI_ICHECK(tail.size() == 2 || tail.size() == 4);
  int mask = 0;
  for (auto slot : tail) {
    TVM_FFI_ICHECK(slot >= -1 && slot <= 2);
    TVM_FFI_ICHECK_EQ(mask & (1 << (slot + 1)), 0);
    mask |= 1 << (slot + 1);
  }
  TVM_FFI_ICHECK(mask == 3 || mask == 15);
  return Module(tvm::ffi::make_object<CudnnFrostMxfp8Mxfp4MoePlan>(
      std::move(fc1), std::move(fc2), tokens, hidden, intermediate, experts, topk, device, scratch1,
      scratch2, gated, std::move(tail), swap1, swap2, swizzled));
}
}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(make_plan, make_plan);
