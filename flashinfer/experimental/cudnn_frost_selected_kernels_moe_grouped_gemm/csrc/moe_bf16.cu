// Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
// Standalone routing -> exported cuDNN Frost FC1/activation -> exported cuDNN Frost FC2 ->
// finalize. No CUTLASS runner, headers, workspace, or callbacks participate in this path.
#include <cuda_bf16.h>
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

__global__ void prefix(const int32_t* counts, int32_t* offsets, int32_t* cursors, int experts,
                       float* scale) {
  int start = 0;
  for (int e = 0; e < experts; ++e) {
    offsets[e] = cursors[e] = start;
    start += counts[e];
  }
  scale[0] = 1.f;
  scale[1] = 4.f;
  scale[2] = 25.f;
}

__global__ void prefix_parallel(const int32_t* counts, int32_t* offsets, int32_t* cursors,
                                int experts, float* scale) {
  if (experts > 128) {
    if (threadIdx.x == 0) {
      int sum = 0;
      for (int e = 0; e < experts; e++) {
        offsets[e] = cursors[e] = sum;
        sum += counts[e];
      }
      scale[0] = 1.f;
      scale[1] = 4.f;
      scale[2] = 25.f;
    }
    return;
  }
  int t = threadIdx.x, lane = t % 32, warp = t / 32;
  int value = t < experts ? counts[t] : 0, scan = value;
#pragma unroll
  for (int d = 1; d < 32; d *= 2) {
    int v = __shfl_up_sync(0xffffffff, scan, d);
    if (lane >= d) scan += v;
  }
  __shared__ int sums[4];
  if (lane == 31) sums[warp] = scan;
  __syncthreads();
  int before = scan - value;
  for (int w = 0; w < warp; w++) before += sums[w];
  if (t < experts) offsets[t] = cursors[t] = before;
  if (t == 0) {
    scale[0] = 1.f;
    scale[1] = 4.f;
    scale[2] = 25.f;
  }
}

__global__ void gather(const __nv_bfloat16* x, const int32_t* ids, int32_t* cursors,
                       int32_t* mapping, __nv_bfloat16* grouped, int rows, int hidden, int topk,
                       int experts) {
  __shared__ int dest;
  for (int64_t r = blockIdx.x; r < rows; r += gridDim.x) {
    int e = ids[r];
    bool valid = e >= 0 && e < experts;
    if (threadIdx.x == 0) {
      dest = atomicAdd(cursors + (valid ? e : 0), 1);
      mapping[r] = dest;
    }
    __syncthreads();
    auto source = reinterpret_cast<const int4*>(x) + (r / topk) * (hidden / 8);
    auto target = reinterpret_cast<int4*>(grouped) + int64_t(dest) * (hidden / 8);
    for (int h = threadIdx.x; h < hidden / 8; h += blockDim.x)
      target[h] = valid ? source[h] : make_int4(0, 0, 0, 0);
    __syncthreads();
  }
}

__global__ void gather_warp(const __nv_bfloat16* x, const int32_t* ids, int32_t* cursors,
                            int32_t* mapping, __nv_bfloat16* grouped, int rows, int hidden,
                            int topk, int experts) {
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  for (int64_t r = int64_t(blockIdx.x) * 4 + warp; r < rows; r += int64_t(gridDim.x) * 4) {
    int e = ids[r];
    bool valid = e >= 0 && e < experts;
    int dest = 0;
    if (lane == 0) {
      dest = atomicAdd(cursors + (valid ? e : 0), 1);
      mapping[r] = dest;
    }
    dest = __shfl_sync(0xffffffff, dest, 0);
    auto source = reinterpret_cast<const int4*>(x) + (r / topk) * (hidden / 8);
    auto target = reinterpret_cast<int4*>(grouped) + int64_t(dest) * (hidden / 8);
    for (int h = lane; h < hidden / 8; h += 32)
      target[h] = valid ? source[h] : make_int4(0, 0, 0, 0);
  }
}

// Small batches prepare group metadata and materialize rows in one launch.
// Each CTA computes a stable destination, independently of CTA scheduling.
__global__ void route_small(const __nv_bfloat16* x, const int32_t* ids, int32_t* offsets,
                            int32_t* mapping, __nv_bfloat16* grouped, float* scale, int rows,
                            int hidden, int topk, int experts) {
  const int r = blockIdx.x;
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const bool active = r < rows;
  const int raw = active ? ids[r] : 0;
  const bool valid = active && raw >= 0 && raw < experts;
  const int expert = valid ? raw : 0;
  int before = 0, group_begin = 0;
  for (int j = threadIdx.x; j < rows; j += blockDim.x) {
    const int value = ids[j];
    const int e = value >= 0 && value < experts ? value : 0;
    before += (e < expert || (e == expert && j < r));
    group_begin += e < r;
  }
  before = __reduce_add_sync(0xffffffff, before);
  group_begin = __reduce_add_sync(0xffffffff, group_begin);
  __shared__ int partial[2][4];
  __shared__ int destination;
  if (lane == 0) {
    partial[0][warp] = before;
    partial[1][warp] = group_begin;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    destination = partial[0][0] + partial[0][1] + partial[0][2] + partial[0][3];
    if (active) mapping[r] = destination;
    if (r < experts) offsets[r] = partial[1][0] + partial[1][1] + partial[1][2] + partial[1][3];
    if (r == 0) {
      scale[0] = 1.f;
      scale[1] = 4.f;
      scale[2] = 25.f;
    }
  }
  __syncthreads();
  if (active) {
    auto source = reinterpret_cast<const int4*>(x) + (r / topk) * (hidden / 8);
    auto target = reinterpret_cast<int4*>(grouped) + int64_t(destination) * (hidden / 8);
    for (int h = threadIdx.x; h < hidden / 8; h += blockDim.x)
      target[h] = valid ? source[h] : make_int4(0, 0, 0, 0);
  }
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

class CudnnFrostMoePlan final : public tvm::ffi::ModuleObj {
 public:
  CudnnFrostMoePlan(Function fc1, Function fc2, int64_t tokens, int64_t hidden,
                    int64_t intermediate, int64_t experts, int64_t topk, int device,
                    size_t scratch1, size_t scratch2, bool gated, Array<int64_t> tail, bool swap1,
                    bool swap2, bool fma)
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
        fma_(fma),
        problem1_{s_, i_,          h_, e_, e_,          h_, 1, s_ * h_, h_,
                  1,  2 * i_ * h_, h_, 1,  2 * i_ * h_, i_, 1, s_ * i_},
        problem2_{s_, h_, i_, e_, e_, i_, 1, s_ * i_, i_, 1, h_ * i_, h_, 1, s_ * h_} {
    if (swap1_)
      problem1_ = Array<int64_t>{i_, s_,          h_, e_, e_,      h_, 1,  2 * i_ * h_, h_,
                                 1,  2 * i_ * h_, h_, 1,  s_ * h_, 1,  i_, s_ * i_};
    if (!gated_) {
      problem1_ =
          swap1_
              ? Array<int64_t>{i_, s_, h_, e_, e_, h_, 1, i_ * h_, h_, 1, s_ * h_, 1, i_, s_ * i_}
              : Array<int64_t>{s_, i_, h_, e_, e_, h_, 1, s_ * h_, h_, 1, i_ * h_, i_, 1, s_ * i_};
    }
    if (swap2_)
      problem2_ =
          Array<int64_t>{h_, s_, i_, e_, e_, i_, 1, h_ * i_, i_, 1, s_ * i_, 1, h_, s_ * h_};
    size_t pos = 0;
    auto reserve = [&](size_t bytes) {
      size_t start = pos;
      pos += align128(bytes);
      return start;
    };
    x_pos_ = reserve(fma_ ? 0 : s_ * h_ * 2);
    mid_pos_ = reserve(s_ * i_ * 2);
    // FC1 has finished consuming grouped tokens before FC2 writes its output.
    y_pos_ = x_pos_;
    counts_pos_ = reserve(fma_ ? 0 : e_ * 4);
    offsets_pos_ = reserve(fma_ ? 0 : e_ * 4);
    cursors_pos_ = reserve(fma_ ? 0 : e_ * 4);
    mapping_pos_ = reserve(fma_ ? 0 : s_ * 4);
    scale_pos_ = reserve(fma_ ? 0 : 3 * sizeof(float));
    scratch_pos_ = reserve(std::max(scratch1_, scratch2_));
    workspace_size_ = pos;
  }

  const char* kind() const final { return "cudnn_frost_bf16_moe_plan"; }
  Optional<Function> GetFunction(const tvm::ffi::String& name) final {
    if (name == "finalize_variant")
      return Function::FromTyped([this]() { return finalize_variant(); });
    if (name == "workspace_size")
      return Function::FromTyped([this]() { return int64_t(workspace_size_); });
    if (name == "run")
      return Function::FromTyped(
          [this](TensorView out, TensorView x, TensorView ids, TensorView scores, TensorView w1,
                 TensorView w2,
                 TensorView workspace) { run(out, x, ids, scores, w1, w2, workspace); });
    return Function(nullptr);
  }

 private:
  int64_t finalize_variant() const {
    if (frost_moe_finalize::small_supported(t_, h_, i_, e_, k_)) return 1;
    return 0;
  }
  void run(TensorView out, TensorView x, TensorView ids, TensorView scores, TensorView w1,
           TensorView w2, TensorView workspace) const {
    tensor(out, device_, dl_bfloat16, {t_, h_});
    tensor(x, device_, dl_bfloat16, {t_, h_});
    tensor(ids, device_, dl_int32, {t_, k_}, 4);
    tensor(scores, device_, dl_float32, {t_, k_}, 4);
    tensor(w1, device_, dl_bfloat16, {e_, (gated_ ? 2 : 1) * i_, h_});
    tensor(w2, device_, dl_bfloat16, {e_, h_, i_});
    tensor(workspace, device_, dl_uint8, {workspace.numel()}, 128);
    TVM_FFI_ICHECK_GE(workspace.numel(), workspace_size_);
    ffi::CUDADeviceGuard guard(device_.device_id);
    auto stream = get_stream(device_);
    auto base = static_cast<char*>(workspace.data_ptr());
    if (fma_) {
      int64_t shape[]{s_, i_}, stride[]{i_, 1};
      DLTensor mid{base + mid_pos_, device_, 2, dl_bfloat16, shape, stride, 0};
      fc1_(x, w1, ids, TensorView(&mid), static_cast<void*>(stream));
      fc2_(TensorView(&mid), w2, ids, scores, out, static_cast<void*>(stream));
      checked(cudaGetLastError());
      return;
    }
    auto gx = reinterpret_cast<__nv_bfloat16*>(base + x_pos_);
    auto mid = reinterpret_cast<__nv_bfloat16*>(base + mid_pos_);
    auto gy = reinterpret_cast<__nv_bfloat16*>(base + y_pos_);
    auto counts = reinterpret_cast<int32_t*>(base + counts_pos_);
    auto offsets = reinterpret_cast<int32_t*>(base + offsets_pos_);
    auto cursors = reinterpret_cast<int32_t*>(base + cursors_pos_);
    auto mapping = reinterpret_cast<int32_t*>(base + mapping_pos_);
    auto scale = reinterpret_cast<float*>(base + scale_pos_);
    auto scratch = reinterpret_cast<int64_t*>(base + scratch_pos_);
    auto expert_ids = static_cast<int32_t*>(ids.data_ptr());
    if (s_ <= 512 && e_ <= 256) {
      route_small<<<std::max(s_, e_), 128, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(x.data_ptr()), expert_ids, offsets, mapping, gx, scale,
          s_, h_, k_, e_);
    } else {
      checked(cudaMemsetAsync(counts, 0, e_ * 4, stream));
      if (s_ >= 16384 && e_ >= 32) {
        histogram_local<<<std::min<int64_t>((s_ + 255) / 256, 1024), 256, 0, stream>>>(
            expert_ids, counts, s_, e_);
      } else {
        histogram<<<std::min<int64_t>((s_ + 255) / 256, 1024), 256, 0, stream>>>(expert_ids, counts,
                                                                                 s_, e_);
      }
      prefix_parallel<<<1, 128, 0, stream>>>(counts, offsets, cursors, e_, scale);
      if (s_ >= 8192) {
        gather_warp<<<std::min<int64_t>((s_ + 3) / 4, 4096), 128, 0, stream>>>(
            static_cast<__nv_bfloat16*>(x.data_ptr()), expert_ids, cursors, mapping, gx, s_, h_, k_,
            e_);
      } else {
        gather<<<std::min<int64_t>(s_, 4096), 128, 0, stream>>>(
            static_cast<__nv_bfloat16*>(x.data_ptr()), expert_ids, cursors, mapping, gx, s_, h_, k_,
            e_);
      }
    }
    checked(cudaGetLastError());

    int64_t xshape[]{s_, h_, 1}, mshape[]{s_, i_, 1};
    int64_t xstride[]{h_, 1, s_ * h_}, mstride[]{i_, 1, s_ * i_};
    int64_t w1shape[]{i_, h_, e_}, w1stride[]{h_, 1, (gated_ ? 2 : 1) * i_ * h_};
    int64_t w2shape[]{h_, i_, e_}, w2stride[]{i_, 1, h_ * i_};
    int64_t eshape[]{e_}, dshape[]{int64_t(scratch1_ / 8)}, unit[]{1};
    int64_t scale_shape[]{1, 1, 1}, scale_stride[]{1, 1, 1};
    DLTensor tx{gx, device_, 3, dl_bfloat16, xshape, xstride, 0};
    DLTensor tm{mid, device_, 3, dl_bfloat16, mshape, mstride, 0};
    DLTensor ty{gy, device_, 3, dl_bfloat16, xshape, xstride, 0};
    DLTensor up{w1.data_ptr(), device_, 3, dl_bfloat16, w1shape, w1stride, 0};
    DLTensor gate = up;
    gate.data = static_cast<__nv_bfloat16*>(w1.data_ptr()) + (gated_ ? i_ * h_ : 0);
    DLTensor down{w2.data_ptr(), device_, 3, dl_bfloat16, w2shape, w2stride, 0};
    DLTensor first{offsets, device_, 1, dl_int32, eshape, unit, 0};
    DLTensor desc{scratch, device_, 1, dl_int64, dshape, unit, 0};
    DLTensor factors[3];
    for (int j = 0; j < 3; ++j)
      factors[j] = DLTensor{scale + j, device_, 3, dl_float32, scale_shape, scale_stride, 0};
    int64_t mshape_sw[]{i_, s_, 1}, mstride_sw[]{1, i_, s_ * i_};
    int64_t yshape_sw[]{h_, s_, 1}, ystride_sw[]{1, h_, s_ * h_};
    DLTensor tm_sw{mid, device_, 3, dl_bfloat16, mshape_sw, mstride_sw, 0};
    DLTensor ty_sw{gy, device_, 3, dl_bfloat16, yshape_sw, ystride_sw, 0};
    auto fc1_out = swap1_ ? &tm_sw : &tm;
    // The frozen host initializes its scheduler counter on every invocation.
    // Descriptor storage is populated by the kernel before it is consumed.
    // AnyView borrows the TensorView descriptor; keep every descriptor alive
    // until CallPacked returns (temporary TensorViews would dangle).
    std::array<TensorView, 9> tensors{TensorView(&first),
                                      TensorView(&desc),
                                      TensorView(swap1_ ? &gate : &tx),
                                      TensorView(swap1_ ? (gated_ ? &up : &tx) : &gate),
                                      TensorView(swap1_ ? &tx : &up),
                                      TensorView(fc1_out),
                                      TensorView(&factors[0]),
                                      TensorView(&factors[1]),
                                      TensorView(&factors[2])};
    tvm::ffi::AnyView args[12];
    int argc = 0;
    args[argc++] = problem1_;
    for (int j = 0; j < (gated_ ? 5 : 4); ++j) args[argc++] = tensors[j];
    for (auto slot : tail_) args[argc++] = tensors[slot + 6];
    args[argc++] = static_cast<void*>(stream);
    tvm::ffi::Any result;
    fc1_.CallPacked(args, argc, &result);
    dshape[0] = scratch2_ / 8;
    fc2_(problem2_, TensorView(&first), TensorView(&desc), TensorView(swap2_ ? &down : &tm),
         TensorView(swap2_ ? &tm : &down), TensorView(swap2_ ? &ty_sw : &ty),
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
        dim3 grid((vectors + 511) / 512, t_);
        finalize_vec_kernel<2, 2, 256><<<grid, 256, 0, stream>>>(
            gy, expert_ids, mapping, static_cast<float*>(scores.data_ptr()), e_,
            static_cast<__nv_bfloat16*>(out.data_ptr()), h_, vectors);
      } else {
        dim3 grid((vectors + 255) / 256, t_);
        finalize_vec_kernel<6, 1, 256><<<grid, 256, 0, stream>>>(
            gy, expert_ids, mapping, static_cast<float*>(scores.data_ptr()), e_,
            static_cast<__nv_bfloat16*>(out.data_ptr()), h_, vectors);
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
  int64_t t_, h_, i_, e_, k_, s_;
  DLDevice device_;
  size_t scratch1_, scratch2_;
  bool gated_;
  Array<int64_t> tail_;
  bool swap1_, swap2_, fma_;
  Array<int64_t> problem1_, problem2_;
  size_t x_pos_, mid_pos_, y_pos_, counts_pos_, offsets_pos_, cursors_pos_, mapping_pos_,
      scale_pos_, scratch_pos_, workspace_size_;
};

Module make_plan(Function fc1, Function fc2, int64_t tokens, int64_t hidden, int64_t intermediate,
                 int64_t experts, int64_t topk, int64_t device, int64_t scratch1, int64_t scratch2,
                 bool gated, Array<int64_t> tail, bool swap1, bool swap2, bool fma) {
  // Limit the private ABI to practical int32-indexed dimensions and avoid
  // overflow in the workspace/launch descriptors even for malformed callers.
  constexpr int64_t max_dim = 1 << 20;
  TVM_FFI_ICHECK(tokens > 0 && tokens <= max_dim && hidden > 0 && hidden <= max_dim &&
                 intermediate > 0 && intermediate <= max_dim && experts > 0 && experts <= 1024 &&
                 topk > 0 && topk <= experts);
  TVM_FFI_ICHECK_LE(tokens * topk, std::numeric_limits<int32_t>::max());
  TVM_FFI_ICHECK_EQ(hidden % 8, 0) << "cuDNN Frost MoE routing uses 128-bit BF16 packs";
  TVM_FFI_ICHECK(scratch1 >= 0 && scratch1 % 128 == 0 && scratch2 >= 0 && scratch2 % 128 == 0);
  if (fma) {
    TVM_FFI_ICHECK(!swap1 && !swap2 && scratch1 == 0 && scratch2 == 0 && tail.empty());
    TVM_FFI_ICHECK(
        (experts == 64 && hidden == 2048 && intermediate == 1408 && topk == 6 && tokens <= 4) ||
        (experts == 12 && hidden == 7168 && intermediate == 3072 && topk == 2 && tokens == 1));
  } else {
    TVM_FFI_ICHECK(tail.size() == 2 || tail.size() == 4);
    int mask = 0;
    for (auto slot : tail) {
      TVM_FFI_ICHECK(slot >= -1 && slot <= 2);
      TVM_FFI_ICHECK_EQ(mask & (1 << (slot + 1)), 0);
      mask |= 1 << (slot + 1);
    }
    TVM_FFI_ICHECK(mask == 3 || mask == 15);
  }
  return Module(tvm::ffi::make_object<CudnnFrostMoePlan>(
      std::move(fc1), std::move(fc2), tokens, hidden, intermediate, experts, topk, device, scratch1,
      scratch2, gated, std::move(tail), swap1, swap2, fma));
}
}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(make_plan, make_plan);
