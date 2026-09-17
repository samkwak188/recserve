#pragma once
// SIMD kernels and runtime CPU feature detection.
//
// References:
//   - Jegou, Douze, Schmid, "Product Quantization for Nearest Neighbor Search",
//     IEEE TPAMI 33(1), 2011. Scalar (per-vector) quantization used here is the
//     cheap cousin: asymmetric distance is avoided by quantizing the query too.
//   - Guo et al., "Accelerating Large-Scale Inference with Anisotropic Vector
//     Quantization", ICML 2020 (ScaNN). Motivates keeping the accumulator in
//     int32 and applying scales once at the end.
//   - Arm, "Neon Intrinsics Reference": SDOT (vdotq_s32) is optional in Armv8.2
//     and mandatory from Armv8.4. Detected at runtime, never assumed.
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(_M_ARM64) || defined(__aarch64__)
#define RECSERVE_ARM64 1
#if defined(_MSC_VER)
#include <arm64_neon.h>
#else
#include <arm_neon.h>
#endif
#endif

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#ifndef PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE
#define PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE 43
#endif
#elif defined(__linux__) && defined(RECSERVE_ARM64)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif

// SDOT is emitted unconditionally by MSVC on ARM64; on GCC/Clang it needs
// -march=...+dotprod, so only compile it when the feature macro is present.
#if defined(RECSERVE_ARM64) && (defined(_MSC_VER) || defined(__ARM_FEATURE_DOTPROD))
#define RECSERVE_HAS_SDOT 1
#endif

namespace recserve {

// ---------------------------------------------------------------- features --

inline bool cpu_has_dotprod() {
#if defined(RECSERVE_HAS_SDOT) && defined(_WIN32)
  static const bool v = IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE) != 0;
  return v;
#elif defined(RECSERVE_HAS_SDOT) && defined(__linux__)
  static const bool v = (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
  return v;
#elif defined(RECSERVE_HAS_SDOT)
  return true;  // __ARM_FEATURE_DOTPROD was set by the compiler flags
#else
  return false;
#endif
}

inline const char* simd_isa() {
#if defined(__AVX2__)
  return "avx2";
#elif defined(RECSERVE_ARM64)
  return cpu_has_dotprod() ? "neon+sdot" : "neon";
#else
  return "scalar";
#endif
}

// ------------------------------------------------------------- f32 kernels --

inline float dot_scalar(const float* a, const float* b, int dim) {
  float s = 0.f;
  for (int i = 0; i < dim; ++i) s += a[i] * b[i];
  return s;
}

inline float dot_simd(const float* a, const float* b, int dim) {
#if defined(__AVX2__)
  __m256 acc = _mm256_setzero_ps();
  int i = 0;
  for (; i + 8 <= dim; i += 8) {
    acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
  }
  alignas(32) float tmp[8];
  _mm256_store_ps(tmp, acc);
  float s = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
  for (; i < dim; ++i) s += a[i] * b[i];
  return s;
#elif defined(RECSERVE_ARM64)
  float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f);
  int i = 0;
  for (; i + 8 <= dim; i += 8) {
    a0 = vmlaq_f32(a0, vld1q_f32(a + i), vld1q_f32(b + i));
    a1 = vmlaq_f32(a1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
  }
  for (; i + 4 <= dim; i += 4) {
    a0 = vmlaq_f32(a0, vld1q_f32(a + i), vld1q_f32(b + i));
  }
  float s = vaddvq_f32(vaddq_f32(a0, a1));
  for (; i < dim; ++i) s += a[i] * b[i];
  return s;
#else
  return dot_scalar(a, b, dim);
#endif
}

// ------------------------------------------------------------ int8 kernels --

// Symmetric per-vector scalar quantization: x ~= scale * q, q in [-127, 127].
struct QuantizedQuery {
  std::vector<std::int8_t> q;
  float scale = 1.f;
  int dim = 0;

  void set(const float* v, int d) {
    dim = d;
    q.resize(static_cast<std::size_t>(d));
    float maxa = 0.f;
    for (int i = 0; i < d; ++i) maxa = std::max(maxa, std::fabs(v[i]));
    scale = maxa > 0.f ? maxa / 127.f : 1.f;
    const float inv = 1.f / scale;
    for (int i = 0; i < d; ++i) {
      int t = static_cast<int>(std::lround(v[i] * inv));
      q[static_cast<std::size_t>(i)] = static_cast<std::int8_t>(t > 127 ? 127 : (t < -127 ? -127 : t));
    }
  }
};

inline int dot_i8_scalar(const std::int8_t* a, const std::int8_t* b, int dim) {
  int s = 0;
  for (int i = 0; i < dim; ++i) s += static_cast<int>(a[i]) * static_cast<int>(b[i]);
  return s;
}

#if defined(RECSERVE_HAS_SDOT)
// 16 int8 lanes per instruction: 4 SDOTs cover a 64-dim vector.
inline int dot_i8_sdot(const std::int8_t* a, const std::int8_t* b, int dim) {
  int32x4_t acc = vdupq_n_s32(0);
  int i = 0;
  for (; i + 16 <= dim; i += 16) {
    acc = vdotq_s32(acc, vld1q_s8(a + i), vld1q_s8(b + i));
  }
  int s = vaddvq_s32(acc);
  for (; i < dim; ++i) s += static_cast<int>(a[i]) * static_cast<int>(b[i]);
  return s;
}
#endif

#if defined(RECSERVE_ARM64)
// Armv8.0 baseline: widening multiply + pairwise accumulate. Always legal.
inline int dot_i8_widen(const std::int8_t* a, const std::int8_t* b, int dim) {
  int32x4_t acc = vdupq_n_s32(0);
  int i = 0;
  for (; i + 8 <= dim; i += 8) {
    acc = vpadalq_s16(acc, vmull_s8(vld1_s8(a + i), vld1_s8(b + i)));
  }
  int s = vaddvq_s32(acc);
  for (; i < dim; ++i) s += static_cast<int>(a[i]) * static_cast<int>(b[i]);
  return s;
}
#endif

#if defined(__AVX2__)
// Sign-extend to int16 then PMADDWD. Avoids the unsigned-operand trap in
// _mm256_maddubs_epi16, which would misread negative query bytes.
inline int dot_i8_avx2(const std::int8_t* a, const std::int8_t* b, int dim) {
  __m256i acc = _mm256_setzero_si256();
  int i = 0;
  for (; i + 16 <= dim; i += 16) {
    __m256i va = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i)));
    __m256i vb = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i)));
    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(va, vb));
  }
  __m128i lo = _mm256_castsi256_si128(acc);
  __m128i hi = _mm256_extracti128_si256(acc, 1);
  __m128i sum = _mm_add_epi32(lo, hi);
  sum = _mm_hadd_epi32(sum, sum);
  sum = _mm_hadd_epi32(sum, sum);
  int s = _mm_cvtsi128_si32(sum);
  for (; i < dim; ++i) s += static_cast<int>(a[i]) * static_cast<int>(b[i]);
  return s;
}
#endif

inline int dot_i8(const std::int8_t* a, const std::int8_t* b, int dim) {
#if defined(__AVX2__)
  return dot_i8_avx2(a, b, dim);
#elif defined(RECSERVE_HAS_SDOT)
  return cpu_has_dotprod() ? dot_i8_sdot(a, b, dim) : dot_i8_widen(a, b, dim);
#elif defined(RECSERVE_ARM64)
  return dot_i8_widen(a, b, dim);
#else
  return dot_i8_scalar(a, b, dim);
#endif
}

// --------------------------------------------------------- blocked kernels --

// AoSoA block width. Items are grouped in blocks of kBlock; inside a block the
// layout is dimension-major, so scoring a whole block is one sequential sweep
// and the vector unit works ACROSS items instead of across dimensions.
inline constexpr int kBlock = 8;

// base points at blk[(block * dim) * kBlock]; writes kBlock scores.
inline void score_block_f32(const float* q, const float* base, int dim, float* out) {
#if defined(__AVX2__)
  __m256 acc = _mm256_setzero_ps();
  for (int d = 0; d < dim; ++d) {
    acc = _mm256_fmadd_ps(_mm256_set1_ps(q[d]), _mm256_loadu_ps(base + d * kBlock), acc);
  }
  _mm256_storeu_ps(out, acc);
#elif defined(RECSERVE_ARM64)
  float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f);
  for (int d = 0; d < dim; ++d) {
    float32x4_t qd = vdupq_n_f32(q[d]);
    a0 = vmlaq_f32(a0, qd, vld1q_f32(base + d * kBlock));
    a1 = vmlaq_f32(a1, qd, vld1q_f32(base + d * kBlock + 4));
  }
  vst1q_f32(out, a0);
  vst1q_f32(out + 4, a1);
#else
  for (int j = 0; j < kBlock; ++j) out[j] = 0.f;
  for (int d = 0; d < dim; ++d) {
    for (int j = 0; j < kBlock; ++j) out[j] += q[d] * base[d * kBlock + j];
  }
#endif
}

}  // namespace recserve
