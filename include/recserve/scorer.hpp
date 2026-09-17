#pragma once
#include "types.hpp"
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

#if defined(__AVX2__)
#include <immintrin.h>
#endif
#if defined(_M_ARM64) || defined(__aarch64__)
#if defined(_MSC_VER)
#include <arm64_neon.h>
#else
#include <arm_neon.h>
#endif
#endif

namespace recserve {

inline float l2_normalize(float* v, int dim) {
  double s = 0;
  for (int i = 0; i < dim; ++i) s += static_cast<double>(v[i]) * v[i];
  float n = static_cast<float>(std::sqrt(s));
  if (n <= 1e-12f) return 0.f;
  for (int i = 0; i < dim; ++i) v[i] /= n;
  return n;
}

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
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    acc = _mm256_fmadd_ps(va, vb, acc);
  }
  alignas(32) float tmp[8];
  _mm256_store_ps(tmp, acc);
  float s = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
  for (; i < dim; ++i) s += a[i] * b[i];
  return s;
#elif defined(_M_ARM64) || defined(__aarch64__)
  float32x4_t acc = vdupq_n_f32(0.f);
  int i = 0;
  for (; i + 4 <= dim; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vld1q_f32(b + i);
    acc = vmlaq_f32(acc, va, vb);
  }
  float s = vgetq_lane_f32(acc, 0) + vgetq_lane_f32(acc, 1) + vgetq_lane_f32(acc, 2) +
            vgetq_lane_f32(acc, 3);
  for (; i < dim; ++i) s += a[i] * b[i];
  return s;
#else
  return dot_scalar(a, b, dim);
#endif
}

struct Catalog {
  int dim = 0;
  int n = 0;
  DType dtype = DType::Float32;
  // AoS: n * dim floats
  std::vector<float> aos;
  // SoA: dim * n floats, row d holds all items' d-th component
  std::vector<float> soa;
  std::vector<std::int8_t> aos_i8;
  std::vector<float> i8_scale;  // per-item scale: f32 ~= i8 * scale

  void resize(int n_items, int d) {
    n = n_items;
    dim = d;
    aos.assign(static_cast<std::size_t>(n) * dim, 0.f);
    soa.assign(static_cast<std::size_t>(n) * dim, 0.f);
    aos_i8.clear();
    i8_scale.clear();
  }

  float* aos_item(int i) { return aos.data() + static_cast<std::size_t>(i) * dim; }
  const float* aos_item(int i) const { return aos.data() + static_cast<std::size_t>(i) * dim; }

  void rebuild_soa() {
    soa.resize(static_cast<std::size_t>(n) * dim);
    for (int i = 0; i < n; ++i) {
      for (int d = 0; d < dim; ++d) {
        soa[static_cast<std::size_t>(d) * n + i] = aos_item(i)[d];
      }
    }
  }

  void quantize_i8() {
    aos_i8.resize(static_cast<std::size_t>(n) * dim);
    i8_scale.resize(n);
    for (int i = 0; i < n; ++i) {
      const float* v = aos_item(i);
      float maxa = 0.f;
      for (int d = 0; d < dim; ++d) maxa = std::max(maxa, std::fabs(v[d]));
      float scale = maxa > 0.f ? maxa / 127.f : 1.f;
      i8_scale[i] = scale;
      for (int d = 0; d < dim; ++d) {
        int q = static_cast<int>(std::lround(v[d] / scale));
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        aos_i8[static_cast<std::size_t>(i) * dim + d] = static_cast<std::int8_t>(q);
      }
    }
    dtype = DType::Int8;
  }

  float dot_item(const float* query, int item, bool simd, bool soa_layout) const {
    if (dtype == DType::Int8) {
      const std::int8_t* q8 = aos_i8.data() + static_cast<std::size_t>(item) * dim;
      int acc = 0;
      // Query stays f32; dequant on the fly is slower, so convert query once outside.
      // Here: i8 * (query/scale) would be wrong. Use i8 * query then * scale.
      float s = 0.f;
      for (int d = 0; d < dim; ++d) s += static_cast<float>(q8[d]) * query[d];
      (void)acc;
      return s * i8_scale[item];
    }
    if (soa_layout) {
      float s = 0.f;
      for (int d = 0; d < dim; ++d) s += query[d] * soa[static_cast<std::size_t>(d) * n + item];
      return s;
    }
    return simd ? dot_simd(query, aos_item(item), dim) : dot_scalar(query, aos_item(item), dim);
  }
};

// Linear ranker: embedding dot + a few explicit feature weights.
inline float rank_score(float dot, float user_ctr, float item_ctr, float recency,
                        const float w[4]) {
  return w[0] * dot + w[1] * user_ctr + w[2] * item_ctr + w[3] * recency;
}

}  // namespace recserve
