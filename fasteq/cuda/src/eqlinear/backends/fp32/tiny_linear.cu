#include "tiny_linear.cuh"

#include <climits>
#include <stdexcept>
#include <string>
#include <utility>

#include <cute/tensor.hpp>

#include "device_buffer.h"
#include "numeric_policy.h"

namespace eqlin_fp32 {
namespace {

// All device arithmetic stays FP32. Explicit rounding preserves the error transforms
// even when the surrounding build allows multiply/add contraction.
struct FloatPair {
  float hi = 0.f, lo = 0.f;

  __device__ __forceinline__ static FloatPair two_sum(float a, float b) {
    float const hi = __fadd_rn(a, b);
    float const bv = __fsub_rn(hi, a);
    float const lo = __fadd_rn(__fsub_rn(a, __fsub_rn(hi, bv)), __fsub_rn(b, bv));
    return {hi, lo};
  }

  __device__ __forceinline__ static FloatPair two_prod(float a, float b) {
    float const hi = __fmul_rn(a, b);
    return {hi, __fmaf_rn(a, b, -hi)};
  }

  __device__ __forceinline__ void add(FloatPair b) {
    FloatPair const s = two_sum(hi, b.hi);
    FloatPair const t = two_sum(lo, b.lo);
    FloatPair const u = two_sum(s.lo, t.hi);
    FloatPair const v = two_sum(s.hi, u.hi);
    *this = two_sum(v.hi, __fadd_rn(v.lo, __fadd_rn(u.lo, t.lo)));
  }

  __device__ __forceinline__ FloatPair multiply(FloatPair b) const {
    FloatPair result = two_prod(hi, b.hi);
    result.add(two_prod(hi, b.lo));
    result.add(two_prod(lo, b.hi));
    result.add(two_prod(lo, b.lo));
    return result;
  }

  __device__ __forceinline__ float rounded() const { return __fadd_rn(hi, lo); }
};

// Split the original CG on the CPU, never after a float conversion in the runner.
float2 split_scale(double scale) {
  float const hi = static_cast<float>(scale);
  return make_float2(hi, static_cast<float>(scale - static_cast<double>(hi)));
}

struct Path {
  long long a_off = 0, b_off = 0, lda = 0;
  int K = 0;
  float2 alpha = {1.f, 0.f};
};

struct Group {
  int first = 0, count = 0;
  long long c_off = 0, ldc = 0;
};

// Logical grid is (batch, group, N-tile32). Flatten batch/group so four warps
// in a 128-thread block can process unrelated rows without block synchronization.
__global__ void tiny_linear_kernel(float const *__restrict__ a,
                                    float const *__restrict__ weights,
                                    float *__restrict__ out,
                                    Path const *__restrict__ paths,
                                    Group const *__restrict__ groups,
                                    int B, int N, int num_groups) {
  int const lane = threadIdx.x & 31;
  long long const row = static_cast<long long>(blockIdx.x) * (blockDim.x / 32) +
                        threadIdx.x / 32;
  if (row >= static_cast<long long>(B) * num_groups) return;
  int const batch = static_cast<int>(row / num_groups);
  int const group_index = static_cast<int>(row % num_groups);
  long long const col = static_cast<long long>(blockIdx.y) * 32 + lane;
  unsigned const active = __ballot_sync(0xffffffffu, col < N);
  if (col >= N) return;
  int const n = static_cast<int>(col);
  Group const group = groups[group_index];

  FloatPair total;
  for (int p = 0; p < group.count; ++p) {
    Path const path = paths[group.first + p];
    auto const A = cute::make_tensor(
        cute::make_gmem_ptr(a + path.a_off),
        cute::make_layout(cute::make_shape(B, path.K),
                          cute::make_stride(path.lda, cute::Int<1>{})));
    auto const W = cute::make_tensor(
        cute::make_gmem_ptr(weights + path.b_off),
        cute::make_layout(cute::make_shape(path.K, N),
                          cute::make_stride(N, cute::Int<1>{})));
    FloatPair dot;
    for (int k = 0; k < path.K; ++k) {
      float av = 0.f;
      if (lane == 0) av = A(batch, k);
      av = __shfl_sync(active, av, 0);
      dot.add(FloatPair::two_prod(av, W(k, n)));
    }
    total.add(dot.multiply(FloatPair{path.alpha.x, path.alpha.y}));
  }
  auto C = cute::make_tensor(
      cute::make_gmem_ptr(out + group.c_off),
      cute::make_layout(cute::make_shape(B, N),
                        cute::make_stride(group.ldc, cute::Int<1>{})));
  C(batch, n) = total.rounded();
}

void require_cuda(cudaError_t status, char const *operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("Tiny: ") + operation + ": " +
                             cudaGetErrorString(status));
}

// Waiting on the supplied stream also makes uploads safe with nonblocking streams
// and pageable host metadata. No staging pointers survive a completed upload.
template <class T>
cudaError_t upload(DeviceBuffer const &buffer, std::vector<T> const &values,
                   cudaStream_t stream) {
  cudaError_t status = cudaMemcpyAsync(buffer.as<T>(), values.data(),
                                       values.size() * sizeof(T), cudaMemcpyHostToDevice, stream);
  if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
  return status;
}

class TinyState {
 public:
  TinyState(int B, int N, std::vector<Path> paths, std::vector<Group> groups,
             float const *weights)
      : B_(B), N_(N), weights_(weights), paths_(std::move(paths)), groups_(std::move(groups)) {
    if (B < 1 || !NumericPolicy::tiny_batch(B) || N < 1 || !weights || paths_.empty() || groups_.empty() ||
        paths_.size() > INT_MAX || groups_.size() > INT_MAX)
      throw std::invalid_argument("Tiny: invalid shape, weights, or metadata size");
    for (Path const &p : paths_) {
      if (p.K <= 0 || p.lda < p.K || p.a_off < 0 || p.b_off < 0)
        throw std::invalid_argument("Tiny: invalid path");
    }
    for (Group const &g : groups_) {
      if (g.first < 0 || g.count < 1 ||
          static_cast<size_t>(g.first) + g.count > paths_.size() ||
          g.c_off < 0 || g.ldc < N)
        throw std::invalid_argument("Tiny: invalid group");
    }
    long long const rows = static_cast<long long>(B_) * groups_.size();
    threads_ = rows < 512 ? 32 : 128;
    long long const blocks = (rows + threads_ / 32 - 1) / (threads_ / 32);
    long long const tiles = (static_cast<long long>(N_) + 31) / 32;
    if (blocks > INT_MAX || tiles > 65535)
      throw std::invalid_argument("Tiny: grid exceeds CUDA launch limits");
    grid_ = dim3(static_cast<unsigned>(blocks), static_cast<unsigned>(tiles));
    if (d_paths_.reset(paths_.size() * sizeof(Path)) ||
        d_groups_.reset(groups_.size() * sizeof(Group)))
      throw std::runtime_error("Tiny: device metadata allocation failed");
    require_cuda(upload(d_groups_, groups_, nullptr), "group upload");
    require_cuda(upload(d_paths_, paths_, nullptr), "path upload");
    ready_ = true;
  }

  int set_scales(std::vector<double> const &scales, cudaStream_t stream) {
    if (scales.size() != paths_.size()) return 1;
    for (size_t p = 0; p < paths_.size(); ++p) paths_[p].alpha = split_scale(scales[p]);
    ready_ = false;
    ready_ = upload(d_paths_, paths_, stream) == cudaSuccess;
    return ready_ ? 0 : 1;
  }

  int launch(float const *a, float *out, cudaStream_t stream) const {
    if (!ready_ || !a || !out) return 1;
    tiny_linear_kernel<<<grid_, threads_, 0, stream>>>(
        a, weights_, out, d_paths_.as<Path>(), d_groups_.as<Group>(),
        B_, N_, static_cast<int>(groups_.size()));
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
  }

 private:
  int B_, N_, threads_ = 32;
  dim3 grid_;
  float const *weights_;
  std::vector<Path> paths_;
  std::vector<Group> groups_;
  // Declared last: failed construction fences pending copies before destroying host metadata.
  DeviceBuffer d_paths_, d_groups_;
  bool ready_ = false;
};

class TinyForwardPlan final : public GroupedGemmPlan::ImplBase {
 public:
  TinyForwardPlan(int B, int N, std::vector<Path> paths, std::vector<Group> groups,
                   float const *a, float const *w)
      : state_(B, N, std::move(paths), std::move(groups), w), a_(a) {}

  bool original_weight_layout() const override { return true; }
  bool scales_after_dot() const override { return true; }

  int set_scales(std::vector<double> const &scales, cudaStream_t stream) override {
    return state_.set_scales(scales, stream);
  }

  int bind_a(float const *a, cudaStream_t) override {
    if (!a) return 1;
    a_ = a;
    return 0;
  }

  int launch(float *out, cudaStream_t stream) override { return state_.launch(a_, out, stream); }

 private:
  TinyState state_;
  float const *a_;
};

class TinyDgradPlan final : public BwdPlan {
 public:
  TinyDgradPlan(int B, int N, std::vector<Path> paths, std::vector<Group> groups,
                 float const *weights)
      : state_(B, N, std::move(paths), std::move(groups), weights) {}

  bool transposes_weight() const override { return true; }
  bool scales_after_dot() const override { return true; }

  int launch(float const *dy, float *dx, float const *, float *, cudaStream_t stream) override {
    return state_.launch(dy, dx, stream);
  }

 private:
  TinyState state_;
};

// One block writes one contiguous row of [P,V,U]; all addressing is expressed
// through CuTe's original and transposed layouts of the same logical matrix.
__device__ void transpose_weight_row(float const *w, float *transposed,
                                     int p, int v, int U, int V) {
  long long const base = static_cast<long long>(p) * U * V;
  auto const src = cute::make_tensor(
      cute::make_gmem_ptr(w + base),
      cute::make_layout(cute::make_shape(U, V), cute::make_stride(V, cute::Int<1>{})));
  auto dst = cute::make_tensor(
      cute::make_gmem_ptr(transposed + base),
      cute::make_layout(cute::make_shape(V, U), cute::make_stride(U, cute::Int<1>{})));
  for (long long u = threadIdx.x; u < U; u += blockDim.x) dst(v, u) = src(u, v);
}

__global__ void transpose_weight_kernel(float const *w, float *transposed, int U, int V) {
  int const p = blockIdx.x / V;
  int const v = blockIdx.x % V;
  transpose_weight_row(w, transposed, p, v, U, V);
}

}  // namespace

std::unique_ptr<GroupedGemmPlan::ImplBase> make_tiny_forward_plan(
    int B, int U, int V, int J, long long lda, std::vector<RowPlan> const &rows,
    std::vector<int> const &rbase, std::vector<int> const &pg,
    float const *a, float const *w) {
  if (U <= 0 || V <= 0 || J <= 0 || rows.empty() || rows.size() > INT_MAX ||
      rbase.empty() != pg.empty())
    throw std::invalid_argument("Tiny forward: invalid shape or path ranges");
  bool const split = !rbase.empty();
  if (split ? (rbase.size() != static_cast<size_t>(J) || pg.size() != static_cast<size_t>(J))
            : rows.size() != static_cast<size_t>(J))
    throw std::invalid_argument("Tiny forward: output group count mismatch");
  std::vector<Path> paths;
  paths.reserve(rows.size());
  for (RowPlan const &r : rows) {
    if (r.K != U || r.b_off < 0 || r.b_off % U)
      throw std::invalid_argument("Tiny forward: each problem must be one complete input path");
    long long const weight_path = r.b_off / U;
    paths.push_back(Path{r.a_off, weight_path * U * V, lda, U, {1.f, 0.f}});
  }
  std::vector<Group> groups;
  groups.reserve(J);
  for (int j = 0; j < J; ++j)
    groups.push_back(Group{split ? rbase[j] : j, split ? pg[j] : 1,
                            static_cast<long long>(j) * V, static_cast<long long>(J) * V});
  return std::make_unique<TinyForwardPlan>(B, V, std::move(paths), std::move(groups), a, w);
}

std::unique_ptr<BwdPlan> make_tiny_dgrad_plan(
    int B, int N, std::vector<BwdProblem> const &problems, float const *weights) {
  if (problems.empty() || problems.size() > INT_MAX)
    throw std::invalid_argument("Tiny dgrad: invalid problem count");
  std::vector<Path> paths;
  std::vector<Group> groups;
  paths.reserve(problems.size());
  groups.reserve(problems.size());
  for (BwdProblem const &q : problems) {
    if (q.kind != 0 || q.M != B || q.N != N || q.K <= 0 || q.weight_path < 0)
      throw std::invalid_argument("Tiny dgrad: requires uniform dgrad problems with M = B");
    int const first = static_cast<int>(paths.size());
    paths.push_back(Path{q.a_off, static_cast<long long>(q.weight_path) * N * q.K,
                          q.lda, q.K, split_scale(q.scale)});
    groups.push_back(Group{first, 1, q.c_off, q.ldc});
  }
  return std::make_unique<TinyDgradPlan>(B, N, std::move(paths), std::move(groups), weights);
}

int launch_tiny_weight_transpose(float const *w, float *transposed,
                                 int P, int U, int V, cudaStream_t stream) {
  long long const rows = static_cast<long long>(P) * V;
  if (!w || !transposed || w == transposed || P <= 0 || U <= 0 || V <= 0 || rows > INT_MAX)
    return 1;
  transpose_weight_kernel<<<static_cast<unsigned>(rows), 128, 0, stream>>>(w, transposed, U, V);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

}  // namespace eqlin_fp32
