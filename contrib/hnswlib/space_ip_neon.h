#pragma once
#if defined(_MSC_VER)
#include <arm64_neon.h>
#else
#include <arm_neon.h>
#endif

// Drop-in InnerProduct body for hnswlib/space_ip.h on ARM64.
// Proposed upstream; see README.md in this directory.
inline float recserve_hnswlib_inner_product_neon(const float* a, const float* b, unsigned qty) {
  float32x4_t acc = vdupq_n_f32(0.f);
  unsigned i = 0;
  for (; i + 4 <= qty; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vld1q_f32(b + i);
    acc = vmlaq_f32(acc, va, vb);
  }
  float res = vgetq_lane_f32(acc, 0) + vgetq_lane_f32(acc, 1) + vgetq_lane_f32(acc, 2) +
              vgetq_lane_f32(acc, 3);
  for (; i < qty; ++i) res += a[i] * b[i];
  return res;
}
