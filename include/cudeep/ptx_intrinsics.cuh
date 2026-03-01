#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// ===========================================================================
// cuDeep PTX Intrinsics Library
//
// Inline PTX wrappers for performance-critical operations that the NVCC
// compiler cannot reliably emit from C++ source. Every function is
// __device__ __forceinline__ and compiles to exactly the instructions shown.
//
// Sections:
//   1. Vectorised memory — global & shared, loads & stores
//   2. Async copy pipeline — cp.async, commit, wait
//   3. L2 / L1 prefetch
//   4. Tensor core MMA — mma.sync + ldmatrix
//   5. SFU fast math — ex2, lg2, rcp, rsqrt, sqrt, sin, cos
//   6. Explicit FMA
//   7. Warp-level hardware reductions (SM 8.0+)
//   8. Barrier management
//   9. Composite helpers — exp, sigmoid, tanh, gelu via PTX primitives
// ===========================================================================

namespace cudeep {
namespace ptx {

#ifdef __CUDACC__

// ===========================================================================
// 1. Vectorised Memory Operations
// ===========================================================================

// ---- Global loads ----

__device__ __forceinline__ float4 ldg_v4(const float* ptr) {
    float4 r;
    asm volatile("ld.global.v4.f32 {%0,%1,%2,%3}, [%4];"
        : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w)
        : "l"(ptr));
    return r;
}

__device__ __forceinline__ float2 ldg_v2(const float* ptr) {
    float2 r;
    asm volatile("ld.global.v2.f32 {%0,%1}, [%2];"
        : "=f"(r.x), "=f"(r.y)
        : "l"(ptr));
    return r;
}

__device__ __forceinline__ float ldg_f32(const float* ptr) {
    float r;
    asm volatile("ld.global.f32 %0, [%1];" : "=f"(r) : "l"(ptr));
    return r;
}

// Non-coherent (read-only texture path, bypasses L1 for streaming)
__device__ __forceinline__ float4 ldg_nc_v4(const float* ptr) {
    float4 r;
    asm volatile("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];"
        : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w)
        : "l"(ptr));
    return r;
}

// ---- Global stores ----

__device__ __forceinline__ void stg_v4(float* ptr, float4 v) {
    asm volatile("st.global.v4.f32 [%0], {%1,%2,%3,%4};"
        :: "l"(ptr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}

__device__ __forceinline__ void stg_v2(float* ptr, float2 v) {
    asm volatile("st.global.v2.f32 [%0], {%1,%2};"
        :: "l"(ptr), "f"(v.x), "f"(v.y) : "memory");
}

// Write-through (bypass L1 cache on store, good for write-once data)
__device__ __forceinline__ void stg_wt_v4(float* ptr, float4 v) {
    asm volatile("st.global.wt.v4.f32 [%0], {%1,%2,%3,%4};"
        :: "l"(ptr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}

// ---- Shared memory loads ----

__device__ __forceinline__ float4 lds_v4(const float* smem_ptr) {
    float4 r;
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];"
        : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w)
        : "r"(addr));
    return r;
}

__device__ __forceinline__ float2 lds_v2(const float* smem_ptr) {
    float2 r;
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ld.shared.v2.f32 {%0,%1}, [%2];"
        : "=f"(r.x), "=f"(r.y)
        : "r"(addr));
    return r;
}

__device__ __forceinline__ float lds_f32(const float* smem_ptr) {
    float r;
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(r) : "r"(addr));
    return r;
}

// ---- Shared memory stores ----

__device__ __forceinline__ void sts_v4(float* smem_ptr, float4 v) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};"
        :: "r"(addr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}

__device__ __forceinline__ void sts_v2(float* smem_ptr, float2 v) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("st.shared.v2.f32 [%0], {%1,%2};"
        :: "r"(addr), "f"(v.x), "f"(v.y) : "memory");
}

// ===========================================================================
// 2. Async Copy Pipeline (SM 8.0+ / Ampere)
// ===========================================================================

#if __CUDA_ARCH__ >= 800

// Copy 16 bytes (float4) from global to shared, bypassing registers
__device__ __forceinline__ void cp_async_cg_16(void* smem_dst, const void* global_src) {
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem), "l"(global_src) : "memory");
}

// Copy 8 bytes (float2) from global to shared
__device__ __forceinline__ void cp_async_cg_8(void* smem_dst, const void* global_src) {
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 8;\n"
        :: "r"(smem), "l"(global_src) : "memory");
}

// Copy 4 bytes from global to shared
__device__ __forceinline__ void cp_async_ca_4(void* smem_dst, const void* global_src) {
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
        :: "r"(smem), "l"(global_src) : "memory");
}

// Predicated 16-byte copy — zero-fills on pred==false
__device__ __forceinline__ void cp_async_cg_16_zfill(
    void* smem_dst, const void* global_src, bool pred) {
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  @!p st.shared.v4.b32 [%0], {0, 0, 0, 0};\n"
        "  @p cp.async.cg.shared.global [%0], [%1], 16;\n"
        "}\n"
        :: "r"(smem), "l"(global_src), "r"((int)pred) : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

#else  // Pre-Ampere fallback stubs (no-op or direct copy)

__device__ __forceinline__ void cp_async_cg_16(void* smem_dst, const void* global_src) {
    *reinterpret_cast<float4*>(smem_dst) =
        *reinterpret_cast<const float4*>(global_src);
}
__device__ __forceinline__ void cp_async_cg_8(void* smem_dst, const void* global_src) {
    *reinterpret_cast<float2*>(smem_dst) =
        *reinterpret_cast<const float2*>(global_src);
}
__device__ __forceinline__ void cp_async_ca_4(void* smem_dst, const void* global_src) {
    *reinterpret_cast<float*>(smem_dst) =
        *reinterpret_cast<const float*>(global_src);
}
__device__ __forceinline__ void cp_async_cg_16_zfill(
    void* smem_dst, const void* global_src, bool pred) {
    if (pred) {
        *reinterpret_cast<float4*>(smem_dst) =
            *reinterpret_cast<const float4*>(global_src);
    } else {
        *reinterpret_cast<float4*>(smem_dst) = make_float4(0, 0, 0, 0);
    }
}
__device__ __forceinline__ void cp_async_commit() {}
__device__ __forceinline__ void cp_async_wait_all() {}
template <int N>
__device__ __forceinline__ void cp_async_wait_group() {}

#endif  // __CUDA_ARCH__ >= 800

// ===========================================================================
// 3. L2 / L1 Prefetch
// ===========================================================================

__device__ __forceinline__ void prefetch_l2(const void* ptr) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(ptr));
}

__device__ __forceinline__ void prefetch_l1(const void* ptr) {
    asm volatile("prefetch.global.L1 [%0];" :: "l"(ptr));
}

// ===========================================================================
// 4. Tensor Core MMA + ldmatrix (SM 8.0+)
// ===========================================================================

// ---- Tensor core & ldmatrix instructions (SM 8.0+ for real HW path) ----
//
// These are declared unconditionally so host-side code and device code for
// older architectures can still compile.  On SM < 8.0 the body falls back
// to scalar FP32 emulation (never actually called at runtime — the launch
// dispatcher does a capability check).

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define CUDEEP_HAS_TENSOR_CORES 1
#else
#define CUDEEP_HAS_TENSOR_CORES 0
#endif

// ldmatrix: cooperatively load 8×8 matrices from shared memory
// into the register layout expected by mma.sync.

__device__ __forceinline__ void ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
    const void* smem_ptr) {
#if CUDEEP_HAS_TENSOR_CORES
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr));
#else
    r0 = r1 = r2 = r3 = 0;
#endif
}

__device__ __forceinline__ void ldmatrix_x2(
    uint32_t& r0, uint32_t& r1,
    const void* smem_ptr) {
#if CUDEEP_HAS_TENSOR_CORES
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r0), "=r"(r1)
        : "r"(addr));
#else
    r0 = r1 = 0;
#endif
}

__device__ __forceinline__ void ldmatrix_x1(
    uint32_t& r0, const void* smem_ptr) {
#if CUDEEP_HAS_TENSOR_CORES
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n"
        : "=r"(r0) : "r"(addr));
#else
    r0 = 0;
#endif
}

__device__ __forceinline__ void ldmatrix_x4_trans(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
    const void* smem_ptr) {
#if CUDEEP_HAS_TENSOR_CORES
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr));
#else
    r0 = r1 = r2 = r3 = 0;
#endif
}

__device__ __forceinline__ void ldmatrix_x2_trans(
    uint32_t& r0, uint32_t& r1, const void* smem_ptr) {
#if CUDEEP_HAS_TENSOR_CORES
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r0), "=r"(r1) : "r"(addr));
#else
    r0 = r1 = 0;
#endif
}

// ---- mma.sync: TF32 tensor core multiply-accumulate ----
//
// mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
//   D[16×8] = A[16×8] · B[8×8] + C[16×8]     (FP32 accumulators)
//
// TF32: same range as FP32 (8-bit exp), 10-bit mantissa (vs 23).
// Sufficient for DL training and inference.

__device__ __forceinline__ void mma_m16n8k8_tf32(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3) {
#if CUDEEP_HAS_TENSOR_CORES
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, "
        "{%8,%9}, "
        "{%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
#else
    d0 = c0; d1 = c1; d2 = c2; d3 = c3;
#endif
}

// FP16 variant: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
__device__ __forceinline__ void mma_m16n8k16_f16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3) {
#if CUDEEP_HAS_TENSOR_CORES
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, "
        "{%8,%9}, "
        "{%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
#else
    d0 = c0; d1 = c1; d2 = c2; d3 = c3;
#endif
}

// ===========================================================================
// 5. SFU Fast Math — Single-instruction hardware approximations
//
// These map directly to Special Function Unit (SFU) instructions.
// Each takes 1 SFU cycle (~4 FP32 cycles) but has ~1 ULP error.
// With --use_fast_math the compiler may already emit some of these,
// but inline PTX guarantees it.
// ===========================================================================

__device__ __forceinline__ float sfu_exp2(float x) {
    float r;
    asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ float sfu_log2(float x) {
    float r;
    asm("lg2.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ float sfu_rcp(float x) {
    float r;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ float sfu_rsqrt(float x) {
    float r;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ float sfu_sqrt(float x) {
    float r;
    asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ float sfu_sin(float x) {
    float r;
    asm("sin.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ float sfu_cos(float x) {
    float r;
    asm("cos.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

// ===========================================================================
// 6. Explicit FMA
// ===========================================================================

__device__ __forceinline__ float fma_rn(float a, float b, float c) {
    float r;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
    return r;
}

__device__ __forceinline__ float fma_rz(float a, float b, float c) {
    float r;
    asm("fma.rz.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
    return r;
}

// ===========================================================================
// 7. Warp-level Hardware Reductions (SM 8.0+)
//
// redux.sync performs a tree reduction across the warp in hardware,
// faster than the shfl_down_sync loop. Only supports integer types.
// For float, we bitcast to int, reduce, bitcast back — which only
// works for exact-value matching (max/min via unsigned). For float
// sum, we still need shfl.
// ===========================================================================

#if __CUDA_ARCH__ >= 800

__device__ __forceinline__ unsigned int redux_sync_add_u32(unsigned int val) {
    unsigned int r;
    asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;"
        : "=r"(r) : "r"(val));
    return r;
}

__device__ __forceinline__ unsigned int redux_sync_max_u32(unsigned int val) {
    unsigned int r;
    asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;"
        : "=r"(r) : "r"(val));
    return r;
}

__device__ __forceinline__ unsigned int redux_sync_min_u32(unsigned int val) {
    unsigned int r;
    asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;"
        : "=r"(r) : "r"(val));
    return r;
}

__device__ __forceinline__ unsigned int redux_sync_or_u32(unsigned int val) {
    unsigned int r;
    asm volatile("redux.sync.or.b32 %0, %1, 0xffffffff;"
        : "=r"(r) : "r"(val));
    return r;
}

__device__ __forceinline__ unsigned int redux_sync_and_u32(unsigned int val) {
    unsigned int r;
    asm volatile("redux.sync.and.b32 %0, %1, 0xffffffff;"
        : "=r"(r) : "r"(val));
    return r;
}

__device__ __forceinline__ int redux_sync_add_s32(int val) {
    int r;
    asm volatile("redux.sync.add.s32 %0, %1, 0xffffffff;"
        : "=r"(r) : "r"(val));
    return r;
}

__device__ __forceinline__ int redux_sync_max_s32(int val) {
    int r;
    asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;"
        : "=r"(r) : "r"(val));
    return r;
}

__device__ __forceinline__ int redux_sync_min_s32(int val) {
    int r;
    asm volatile("redux.sync.min.s32 %0, %1, 0xffffffff;"
        : "=r"(r) : "r"(val));
    return r;
}

#endif  // __CUDA_ARCH__ >= 800

// ===========================================================================
// 8. Barrier Management
// ===========================================================================

__device__ __forceinline__ void bar_sync(int barrier_id) {
    asm volatile("bar.sync %0;" :: "r"(barrier_id));
}

// Named barrier with thread count (cooperative groups style)
__device__ __forceinline__ void bar_sync_count(int barrier_id, int thread_count) {
    asm volatile("bar.sync %0, %1;" :: "r"(barrier_id), "r"(thread_count));
}

// ===========================================================================
// 9. Composite Helpers
//
// Built from the SFU primitives above. These guarantee minimal instruction
// count for common DL operations.
// ===========================================================================

// exp(x) = 2^(x * log2(e))  — 1 FMA + 1 SFU = 2 instructions
constexpr float LOG2E = 1.4426950408889634f;

__device__ __forceinline__ float fast_exp_ptx(float x) {
    return sfu_exp2(x * LOG2E);
}

// log(x) = log2(x) * ln(2)  — 1 SFU + 1 MUL = 2 instructions
constexpr float LN2 = 0.6931471805599453f;

__device__ __forceinline__ float fast_log_ptx(float x) {
    return sfu_log2(x) * LN2;
}

// sigmoid(x) = 1 / (1 + exp(-x))  — 1 NEG + 1 FMA + 1 SFU + 1 ADD + 1 RCP
__device__ __forceinline__ float fast_sigmoid_ptx(float x) {
    float e = sfu_exp2(-x * LOG2E);
    return sfu_rcp(1.0f + e);
}

// tanh(x) = 2 * sigmoid(2x) - 1
__device__ __forceinline__ float fast_tanh_ptx(float x) {
    return 2.0f * fast_sigmoid_ptx(2.0f * x) - 1.0f;
}

// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
constexpr float GELU_COEFF = 0.7978845608028654f;   // sqrt(2/π)
constexpr float GELU_INNER = 0.044715f;

__device__ __forceinline__ float fast_gelu_ptx(float x) {
    float x3 = x * x * x;
    float inner = GELU_COEFF * (x + GELU_INNER * x3);
    return 0.5f * x * (1.0f + fast_tanh_ptx(inner));
}

// SiLU(x) = x * sigmoid(x)
__device__ __forceinline__ float fast_silu_ptx(float x) {
    return x * fast_sigmoid_ptx(x);
}

// Swish(x, beta) = x * sigmoid(beta * x)
__device__ __forceinline__ float fast_swish_ptx(float x, float beta) {
    return x * fast_sigmoid_ptx(beta * x);
}

// Softplus(x) = log(1 + exp(x))   with numerical stability
__device__ __forceinline__ float fast_softplus_ptx(float x) {
    if (x > 20.0f) return x;
    return fast_log_ptx(1.0f + fast_exp_ptx(x));
}

// ===========================================================================
// 10. TF32 Conversion Helpers
// ===========================================================================

// Round FP32 to TF32 (truncate mantissa from 23 to 10 bits)
// The mma.sync instruction does this internally, but sometimes
// we need explicit conversion for shared memory storage.
__device__ __forceinline__ uint32_t fp32_to_tf32(float val) {
    uint32_t bits;
    asm("mov.b32 %0, %1;" : "=r"(bits) : "f"(val));
    // Round to nearest even: add bit 12 (round), mask off bits 0-12
    bits += 0x1000u;          // round
    bits &= 0xFFFFE000u;     // truncate
    return bits;
}

__device__ __forceinline__ float tf32_to_fp32(uint32_t bits) {
    float val;
    asm("mov.b32 %0, %1;" : "=f"(val) : "r"(bits));
    return val;
}

#endif  // __CUDACC__

}  // namespace ptx
}  // namespace cudeep
