// Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace frost_moe_finalize {

__device__ __forceinline__ float bf16_bits_to_float(uint32_t x) { return __uint_as_float(x << 16); }

__device__ __forceinline__ uint4 load_nc_16(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

__device__ __forceinline__ void store_cs_16(void* p, uint4 v) {
  asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z),
               "r"(v.w)
               : "memory");
}

__device__ __forceinline__ void ordered_fma8(float* a, uint4 v, float w) {
  const uint32_t q[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    a[2 * i] = __fmaf_rn(bf16_bits_to_float(q[i] & 0xffffu), w, a[2 * i]);
    a[2 * i + 1] = __fmaf_rn(bf16_bits_to_float(q[i] >> 16), w, a[2 * i + 1]);
  }
}

__device__ __forceinline__ uint4 pack_bf16_8(const float* a) {
  uint4 o;
  uint32_t* q = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const __nv_bfloat162 p = __floats2bfloat162_rn(a[2 * i], a[2 * i + 1]);
    q[i] = *reinterpret_cast<const uint32_t*>(&p);
  }
  return o;
}

template <int K>
__global__ __launch_bounds__(256) void dense_vec2048_kernel(const uint4* __restrict__ grouped,
                                                            const int32_t* __restrict__ ids,
                                                            const int32_t* __restrict__ mapping,
                                                            const float* __restrict__ scores,
                                                            uint4* __restrict__ output, int experts,
                                                            int rows) {
  const int token = blockIdx.x;
  const int lane = threadIdx.x & 31;
  int e = -1, r = 0;
  float w = 0.0f;
  if (lane < K) {
    const int m = token * K + lane;
    e = ids[m];
    r = mapping[m];
    w = scores[m];
    if (static_cast<unsigned>(e) >= static_cast<unsigned>(experts) ||
        static_cast<unsigned>(r) >= static_cast<unsigned>(rows)) {
      r = 0;
      w = 0.0f;
    }
  }
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
  for (int j = 0; j < K; ++j) {
    const int row = __shfl_sync(0xffffffffu, r, j);
    const float weight = __shfl_sync(0xffffffffu, w, j);
    const int valid_e = __shfl_sync(0xffffffffu, e, j);
    if (static_cast<unsigned>(valid_e) < static_cast<unsigned>(experts)) {
      const uint4 v = __ldcs(grouped + static_cast<int64_t>(row) * 256 + threadIdx.x);
      ordered_fma8(acc, v, weight);
    }
  }
  output[static_cast<int64_t>(token) * 256 + threadIdx.x] = pack_bf16_8(acc);
}

template <int U>
__global__ __launch_bounds__(128) void prefetch_k2_kernel(const __nv_bfloat16* __restrict__ grouped,
                                                          const int32_t* __restrict__ ids,
                                                          const int32_t* __restrict__ mapping,
                                                          const float* __restrict__ scores,
                                                          __nv_bfloat16* __restrict__ output,
                                                          int hidden, int experts, int nchunk) {
  constexpr int CHUNK = 128 * 8;
  const int token = blockIdx.y;
  const int c0 = blockIdx.x * U;
  const int h0 = c0 * CHUNK + threadIdx.x * 8;
  const int lane = threadIdx.x & 31;
  int e = -1, r = 0;
  float w = 0.0f;
  if (lane < 2) {
    const int m = token * 2 + lane;
    e = ids[m];
    r = mapping[m];
    w = scores[m];
    if (static_cast<unsigned>(e) >= static_cast<unsigned>(experts)) {
      r = 0;
      w = 0.0f;
    }
  }
  int row[2], valid[2];
  float weight[2];
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    row[j] = __shfl_sync(0xffffffffu, r, j);
    weight[j] = __shfl_sync(0xffffffffu, w, j);
    valid[j] =
        static_cast<unsigned>(__shfl_sync(0xffffffffu, e, j)) < static_cast<unsigned>(experts);
  }
  uint4 values[U][2];
  bool live[U];
#pragma unroll
  for (int u = 0; u < U; ++u) live[u] = c0 + u < nchunk;
#pragma unroll
  for (int u = 0; u < U; ++u) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      if (live[u] && valid[j]) {
        values[u][j] = load_nc_16(grouped + static_cast<int64_t>(row[j]) * hidden + h0 + u * CHUNK);
      }
    }
  }
#pragma unroll
  for (int u = 0; u < U; ++u) {
    if (!live[u]) continue;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int j = 0; j < 2; ++j)
      if (valid[j]) ordered_fma8(acc, values[u][j], weight[j]);
    store_cs_16(output + static_cast<int64_t>(token) * hidden + h0 + u * CHUNK, pack_bf16_8(acc));
  }
}

inline bool measured_geometry(int64_t h, int64_t i, int64_t e, int64_t k) {
  return (e == 64 && h == 2048 && i == 1408 && k == 6) ||
         (e == 12 && h == 7168 && i == 3072 && k == 2) ||
         (e == 128 && h == 2048 && i == 768 && k == 8);
}
inline bool small_supported(int64_t t, int64_t h, int64_t i, int64_t e, int64_t k) {
  return t > 8 && t <= 512 && measured_geometry(h, i, e, k);
}
inline bool shape_supported(int64_t t, int64_t h, int64_t i, int64_t e, int64_t k) {
  return t > 512 && t <= 12288 && e == 8 && h == 4096 && i == 14336 && k == 2;
}
inline void launch_small(const __nv_bfloat16* grouped, const int32_t* ids, const int32_t* mapping,
                         const float* scores, __nv_bfloat16* output, int tokens, int hidden,
                         int experts, int topk, cudaStream_t stream) {
  if (topk == 2) {
    const int nchunk = hidden / 1024;
    prefetch_k2_kernel<4><<<dim3((nchunk + 3) / 4, tokens), 128, 0, stream>>>(
        grouped, ids, mapping, scores, output, hidden, experts, nchunk);
  } else if (topk == 6) {
    dense_vec2048_kernel<6>
        <<<tokens, 256, 0, stream>>>(reinterpret_cast<const uint4*>(grouped), ids, mapping, scores,
                                     reinterpret_cast<uint4*>(output), experts, tokens * topk);
  } else {
    dense_vec2048_kernel<8>
        <<<tokens, 256, 0, stream>>>(reinterpret_cast<const uint4*>(grouped), ids, mapping, scores,
                                     reinterpret_cast<uint4*>(output), experts, tokens * topk);
  }
}
}  // namespace frost_moe_finalize
