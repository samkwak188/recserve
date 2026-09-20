#include "recserve/gpu.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cub/block/block_radix_sort.cuh>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>

namespace recserve {
namespace {
constexpr int kThreads = 256, kPerThread = 4, kChunk = kThreads * kPerThread;
constexpr int kTile = 65536;
using Key = unsigned long long;

void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void check(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("cuBLAS status " + std::to_string(static_cast<int>(status)));
}

// Lexicographic score descending, then stable item ID ascending, including ties.
__device__ Key pack(float score, unsigned id) {
  if (score == 0.f) score = 0.f;  // normalize negative zero
  unsigned bits = __float_as_uint(score);
  unsigned ordered = bits ^ ((bits & 0x80000000u) ? 0xffffffffu : 0x80000000u);
  return (static_cast<Key>(ordered) << 32) | (0xffffffffu - id);
}

template<bool First>
__global__ void select_chunks(const float* scores, const Key* input, Key* output,
                              int count, int output_count, int k, int base) {
  using Sort = cub::BlockRadixSort<Key, kThreads, kPerThread>;
  __shared__ typename Sort::TempStorage storage;
  Key keys[kPerThread];
  const int q = static_cast<int>(blockIdx.y);
  const int block = static_cast<int>(blockIdx.x);
  #pragma unroll
  for (int j = 0; j < kPerThread; ++j) {
    const int i = block * kChunk + static_cast<int>(threadIdx.x) * kPerThread + j;
    keys[j] = i < count ? (First ? pack(scores[static_cast<std::size_t>(q) * count + i],
                                       static_cast<unsigned>(base + i))
                                 : input[static_cast<std::size_t>(q) * count + i]) : 0;
  }
  Sort(storage).SortDescending(keys);
  #pragma unroll
  for (int j = 0; j < kPerThread; ++j) {
    int rank = static_cast<int>(threadIdx.x) * kPerThread + j;
    if (rank < k) output[static_cast<std::size_t>(q) * output_count + block * k + rank] = keys[j];
  }
}

__global__ void merge_topk(const Key* tile, Key* accumulated, int k, bool first) {
  using Sort = cub::BlockRadixSort<Key, kThreads, kPerThread>;
  __shared__ typename Sort::TempStorage storage;
  Key keys[kPerThread];
  std::size_t offset = static_cast<std::size_t>(blockIdx.x) * k;
  #pragma unroll
  for (int j = 0; j < kPerThread; ++j) {
    int i = static_cast<int>(threadIdx.x) * kPerThread + j;
    keys[j] = i < k ? tile[offset + i] :
              (!first && i < 2 * k ? accumulated[offset + i - k] : 0);
  }
  // All old accumulated values are loaded before any thread writes results.
  Sort(storage).SortDescending(keys);
  #pragma unroll
  for (int j = 0; j < kPerThread; ++j) {
    int i = static_cast<int>(threadIdx.x) * kPerThread + j;
    if (i < k) accumulated[offset + i] = keys[j];
  }
}

template<typename T> struct DeviceBuffer {
  T* p = nullptr;
  ~DeviceBuffer() { if (p) cudaFree(p); }
  void allocate(std::size_t count) { check(cudaMalloc(reinterpret_cast<void**>(&p), count * sizeof(T))); }
};
template<typename T> struct HostBuffer {
  T* p = nullptr;
  ~HostBuffer() { if (p) cudaFreeHost(p); }
  void allocate(std::size_t count) { check(cudaMallocHost(reinterpret_cast<void**>(&p), count * sizeof(T))); }
};
struct Stream {
  cudaStream_t value = nullptr;
  Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { if (value) cudaStreamDestroy(value); }
};
struct Blas {
  cublasHandle_t value = nullptr;
  Blas() { check(cublasCreate(&value)); }
  ~Blas() { if (value) cublasDestroy(value); }
};
struct Event {
  cudaEvent_t value = nullptr;
  Event() { check(cudaEventCreate(&value)); }
  ~Event() { if (value) cudaEventDestroy(value); }
};

class CudaScorer final : public GpuScorer {
 public:
  CudaScorer(const float* catalog, int n, int d, int batch) : n_(n), dim_(d), max_batch_(batch) {
    if (!catalog || n <= 0 || d <= 0 || d > 4096 || batch <= 0 || batch > 2048)
      throw std::invalid_argument("invalid CUDA catalog shape or batch limit");
    check(cudaGetDevice(&device_));
    cudaDeviceProp props{};
    check(cudaGetDeviceProperties(&props, device_));
    name_ = props.name;
    tile_ = std::min(n, kTile);
    const std::size_t nb = static_cast<std::size_t>(batch);
    const std::size_t partial = static_cast<std::size_t>((tile_ + kChunk - 1) / kChunk) * kMaxK;
    bytes_ = static_cast<std::size_t>(n) * d * sizeof(float) + nb * d * sizeof(float) +
             nb * tile_ * sizeof(float) + 2 * nb * partial * sizeof(Key) + nb * kMaxK * sizeof(Key);
    std::size_t free = 0, total = 0;
    check(cudaMemGetInfo(&free, &total));
    if (bytes_ > free * 8 / 10) throw std::runtime_error("CUDA workspace exceeds 80% of free VRAM");
    for (std::size_t i = 0; i < static_cast<std::size_t>(n) * d; ++i)
      if (!std::isfinite(catalog[i])) throw std::invalid_argument("nonfinite catalog value");
    cat_.allocate(static_cast<std::size_t>(n) * d);
    query_.allocate(nb * d);
    scores_.allocate(nb * tile_);
    a_.allocate(nb * partial);
    b_.allocate(nb * partial);
    best_.allocate(nb * kMaxK);
    host_query_.allocate(nb * d);
    host_best_.allocate(nb * kMaxK);
    check(cudaMemcpy(cat_.p, catalog, static_cast<std::size_t>(n) * d * sizeof(float), cudaMemcpyHostToDevice));
    check(cublasSetStream(blas_.value, stream_.value));
    check(cublasSetMathMode(blas_.value, CUBLAS_PEDANTIC_MATH));
  }

  void topk(const float* queries, int batch, int k, ScoredItem* output, GpuTiming* timing) override {
    if (!queries || !output || batch < 1 || batch > max_batch_ || k < 1 || k > n_ || k > static_cast<int>(kMaxK))
      throw std::invalid_argument("invalid CUDA query batch or top-k");
    const auto started = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(mutex_);
    check(cudaSetDevice(device_));
    for (std::size_t i = 0; i < static_cast<std::size_t>(batch) * dim_; ++i) {
      if (!std::isfinite(queries[i])) throw std::invalid_argument("nonfinite query value");
      host_query_.p[i] = queries[i];
    }
    auto record = [&](int i) { check(cudaEventRecord(events_[i].value, stream_.value)); };
    record(0);
    check(cudaMemcpyAsync(query_.p, host_query_.p, static_cast<std::size_t>(batch) * dim_ * sizeof(float),
                          cudaMemcpyHostToDevice, stream_.value));
    record(1);
    float alpha = 1, beta = 0;
    for (int base = 0; base < n_; base += tile_) {
      int count = std::min(tile_, n_ - base);
      // Row-major catalog (count x dim) is column-major (dim x count).
      check(cublasSgemm(blas_.value, CUBLAS_OP_T, CUBLAS_OP_N, count, batch, dim_,
                        &alpha, cat_.p + static_cast<std::size_t>(base) * dim_, dim_,
                        query_.p, dim_, &beta, scores_.p, count));
      int chunks = (count + kChunk - 1) / kChunk;
      int next = chunks * k;
      select_chunks<true><<<dim3(chunks, batch), kThreads, 0, stream_.value>>>(scores_.p, nullptr, a_.p, count, next, k, base);
      Key* current = a_.p;
      Key* other = b_.p;
      while (next > k) {
        count = next;
        chunks = (count + kChunk - 1) / kChunk;
        next = chunks * k;
        select_chunks<false><<<dim3(chunks, batch), kThreads, 0, stream_.value>>>(nullptr, current, other, count, next, k, 0);
        std::swap(current, other);
      }
      merge_topk<<<batch, kThreads, 0, stream_.value>>>(current, best_.p, k, base == 0);
      check(cudaGetLastError());
    }
    record(2);
    check(cudaMemcpyAsync(host_best_.p, best_.p, static_cast<std::size_t>(batch) * k * sizeof(Key),
                          cudaMemcpyDeviceToHost, stream_.value));
    record(3);
    check(cudaStreamSynchronize(stream_.value));
    for (std::size_t i = 0; i < static_cast<std::size_t>(batch) * k; ++i) {
      Key key = host_best_.p[i];
      unsigned ordered = static_cast<unsigned>(key >> 32);
      unsigned bits = ordered ^ ((ordered & 0x80000000u) ? 0x80000000u : 0xffffffffu);
      float score;
      std::memcpy(&score, &bits, sizeof(score));
      if (!std::isfinite(score)) throw std::runtime_error("nonfinite CUDA score");
      output[i] = {0xffffffffu - static_cast<unsigned>(key), score};
    }
    if (timing) {
      float h2d, compute, d2h;
      check(cudaEventElapsedTime(&h2d, events_[0].value, events_[1].value));
      check(cudaEventElapsedTime(&compute, events_[1].value, events_[2].value));
      check(cudaEventElapsedTime(&d2h, events_[2].value, events_[3].value));
      *timing = {h2d, compute, d2h, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-started).count()};
    }
  }
  std::size_t device_bytes() const override { return bytes_; }
  std::string device_name() const override { return name_; }

 private:
  int n_, dim_, max_batch_, tile_ = 0, device_ = 0;
  std::size_t bytes_ = 0;
  std::string name_;
  std::mutex mutex_;
  Stream stream_;
  Blas blas_;
  std::array<Event, 4> events_;
  DeviceBuffer<float> cat_, query_, scores_;
  DeviceBuffer<Key> a_, b_, best_;
  HostBuffer<float> host_query_;
  HostBuffer<Key> host_best_;
};
}  // namespace

std::unique_ptr<GpuScorer> make_gpu_scorer(const float* cat, int n, int dim, int batch, std::string* error) {
  try { return std::make_unique<CudaScorer>(cat, n, dim, batch); }
  catch (const std::exception& exc) { if (error) *error = exc.what(); return nullptr; }
}
}  // namespace recserve
