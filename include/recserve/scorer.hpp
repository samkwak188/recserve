#pragma once
#include "types.hpp"
#include "simd.hpp"
#include <cmath>
#include <cstring>
#include <string>
#include <fstream>
#include <vector>
#include <algorithm>

namespace recserve {

inline float l2_normalize(float* v, int dim) {
  double s = 0;
  for (int i = 0; i < dim; ++i) s += static_cast<double>(v[i]) * v[i];
  float n = static_cast<float>(std::sqrt(s));
  if (n <= 1e-12f) return 0.f;
  for (int i = 0; i < dim; ++i) v[i] /= n;
  return n;
}

struct Catalog {
  int dim = 0;
  int n = 0;
  std::vector<float> aos;             // n * dim, item-major
  std::vector<float> soa;             // dim * n, dimension-major (strided per item)
  std::vector<float> blk;             // AoSoA: [(block*dim + d)*kBlock + j]
  std::vector<std::int8_t> aos_i8;    // n * dim, item-major int8
  std::vector<float> i8_scale;        // per-item dequant scale

  int n_blocks() const { return (n + kBlock - 1) / kBlock; }

  void resize(int n_items, int d) {
    n = n_items;
    dim = d;
    aos.assign(static_cast<std::size_t>(n) * dim, 0.f);
    soa.clear();
    blk.clear();
    aos_i8.clear();
    i8_scale.clear();
  }

  float* aos_item(int i) { return aos.data() + static_cast<std::size_t>(i) * dim; }
  const float* aos_item(int i) const { return aos.data() + static_cast<std::size_t>(i) * dim; }
  const std::int8_t* i8_item(int i) const {
    return aos_i8.data() + static_cast<std::size_t>(i) * dim;
  }

  void rebuild_soa() {
    soa.assign(static_cast<std::size_t>(n) * dim, 0.f);
    for (int i = 0; i < n; ++i) {
      for (int d = 0; d < dim; ++d) {
        soa[static_cast<std::size_t>(d) * n + i] = aos_item(i)[d];
      }
    }
  }

  // Pad the tail block with zeros so a full block can always be scored.
  void rebuild_blocked() {
    const std::size_t nb = static_cast<std::size_t>(n_blocks());
    blk.assign(nb * static_cast<std::size_t>(dim) * kBlock, 0.f);
    for (int i = 0; i < n; ++i) {
      const int b = i / kBlock, j = i % kBlock;
      const float* v = aos_item(i);
      for (int d = 0; d < dim; ++d) {
        blk[(static_cast<std::size_t>(b) * dim + d) * kBlock + j] = v[d];
      }
    }
  }

  void quantize_i8() {
    aos_i8.resize(static_cast<std::size_t>(n) * dim);
    i8_scale.resize(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
      const float* v = aos_item(i);
      float maxa = 0.f;
      for (int d = 0; d < dim; ++d) maxa = std::max(maxa, std::fabs(v[d]));
      const float scale = maxa > 0.f ? maxa / 127.f : 1.f;
      i8_scale[static_cast<std::size_t>(i)] = scale;
      const float inv = 1.f / scale;
      for (int d = 0; d < dim; ++d) {
        int q = static_cast<int>(std::lround(v[d] * inv));
        aos_i8[static_cast<std::size_t>(i) * dim + d] =
            static_cast<std::int8_t>(q > 127 ? 127 : (q < -127 ? -127 : q));
      }
    }
  }

  // Scattered single-item score, used on the graph-traversal path where the
  // candidate ids are not contiguous and blocking cannot help.
  float dot_item(const float* query, const QuantizedQuery* qq, int item, Kernel k) const {
    switch (k) {
      case Kernel::Int8: {
        if (aos_i8.empty() || qq == nullptr) return dot_simd(query, aos_item(item), dim);
        const int acc = dot_i8(qq->q.data(), i8_item(item), dim);
        return static_cast<float>(acc) * qq->scale * i8_scale[static_cast<std::size_t>(item)];
      }
      case Kernel::SoaStrided: {
        float s = 0.f;
        for (int d = 0; d < dim; ++d) s += query[d] * soa[static_cast<std::size_t>(d) * n + item];
        return s;
      }
      case Kernel::Scalar:
        return dot_scalar(query, aos_item(item), dim);
      case Kernel::Blocked:
      case Kernel::Simd:
      default:
        return dot_simd(query, aos_item(item), dim);
    }
  }

  // Contiguous scan of `count` items starting at `begin`. `begin` must be a
  // multiple of kBlock for the Blocked kernel. This is the bandwidth-bound path.
  void score_range(const float* query, const QuantizedQuery* qq, Kernel k, int begin, int count,
                   float* out) const {
    switch (k) {
      case Kernel::Blocked: {
        const int nb = (count + kBlock - 1) / kBlock;
        for (int b = 0; b < nb; ++b) {
          const int blk_idx = begin / kBlock + b;
          const float* base = blk.data() + static_cast<std::size_t>(blk_idx) * dim * kBlock;
          float tmp[kBlock];
          score_block_f32(query, base, dim, tmp);
          const int lim = std::min(kBlock, count - b * kBlock);
          for (int j = 0; j < lim; ++j) out[b * kBlock + j] = tmp[j];
        }
        break;
      }
      case Kernel::Int8: {
        if (aos_i8.empty() || qq == nullptr) {
          for (int i = 0; i < count; ++i) out[i] = dot_simd(query, aos_item(begin + i), dim);
          break;
        }
        const float qs = qq->scale;
        for (int i = 0; i < count; ++i) {
          const int item = begin + i;
          const int acc = dot_i8(qq->q.data(), i8_item(item), dim);
          out[i] = static_cast<float>(acc) * qs * i8_scale[static_cast<std::size_t>(item)];
        }
        break;
      }
      default:
        for (int i = 0; i < count; ++i) out[i] = dot_item(query, qq, begin + i, k);
        break;
    }
  }

  // Flat on-disk snapshot so a large catalog is built once and reloaded by every
  // bench run, the way a serving host loads an index rather than building it.
  bool save(const std::string& path) const {
    std::ofstream o(path, std::ios::binary);
    if (!o) return false;
    const std::uint32_t magic = 0x43415431u;  // CAT1
    o.write(reinterpret_cast<const char*>(&magic), 4);
    o.write(reinterpret_cast<const char*>(&n), 4);
    o.write(reinterpret_cast<const char*>(&dim), 4);
    o.write(reinterpret_cast<const char*>(aos.data()),
            static_cast<std::streamsize>(aos.size() * sizeof(float)));
    return static_cast<bool>(o);
  }

  bool load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::uint32_t magic = 0;
    int nn = 0, dd = 0;
    in.read(reinterpret_cast<char*>(&magic), 4);
    in.read(reinterpret_cast<char*>(&nn), 4);
    in.read(reinterpret_cast<char*>(&dd), 4);
    if (!in || magic != 0x43415431u || nn <= 0 || dd <= 0) return false;
    resize(nn, dd);
    in.read(reinterpret_cast<char*>(aos.data()),
            static_cast<std::streamsize>(aos.size() * sizeof(float)));
    return static_cast<bool>(in);
  }
};

// Linear ranker: embedding dot + a few explicit feature weights.
inline float rank_score(float dot, float user_ctr, float item_ctr, float recency,
                        const float w[4]) {
  return w[0] * dot + w[1] * user_ctr + w[2] * item_ctr + w[3] * recency;
}

}  // namespace recserve
