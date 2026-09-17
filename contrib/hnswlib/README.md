# ARM64 NEON inner-product / L2 helper for hnswlib

Upstream: https://github.com/nmslib/hnswlib (header-only C++ ANN).
RecServe hits the same inner-product kernel on ARM64 Windows/Linux. hnswlib's
`space_l2.h` / `space_ip.h` are AVX/AVX512-heavy. This patch adds a NEON path
so `InnerProduct` is defined on Apple Silicon / Windows ARM64 without falling
through a scalar-only gap that some build flags accidentally skip.

This is a **proposed** upstream PR, not a merged commit. Do not claim
"production C++ at hnswlib" until it is opened/merged under your GitHub
account.

## Why this is a real patch, not theater

- RecServe's scorer already uses the same `vmlaq_f32` loop on `_M_ARM64`.
- hnswlib contributors regularly accept SIMD backends (AVX, AVX512, SVE).
- The change is localized to `hnswlib/space_ip.h` (and the L2 analogue).

## Apply against a clone

```bash
git clone https://github.com/nmslib/hnswlib.git
cd hnswlib
git apply ../recserve/contrib/hnswlib/0001-arm64-neon-inner-product.patch
```

Open the PR from a fork after a green local test of `python bindings` or the
C++ examples. Link this RecServe repo as the motivating consumer.
