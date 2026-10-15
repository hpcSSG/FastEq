#include <torch/extension.h>

#include <ATen/Parallel.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#define NATIVE_CPU_HAS_CBLAS 1
#elif defined(NATIVE_CPU_USE_OPENBLAS)
#include <cblas.h>
#define NATIVE_CPU_HAS_CBLAS 1
#else
#define NATIVE_CPU_HAS_CBLAS 0
#endif

namespace {

constexpr int64_t kSh3Dim = 16;
constexpr double kShEps = 1.0e-12;

template <typename scalar_t>
inline scalar_t sqr(scalar_t value) {
  return value * value;
}

template <typename scalar_t>
struct Jet3 {
  scalar_t value{};
  scalar_t dx{};
  scalar_t dy{};
  scalar_t dz{};
};

template <typename scalar_t>
inline Jet3<scalar_t> make_jet(scalar_t value, scalar_t dx, scalar_t dy, scalar_t dz) {
  return {value, dx, dy, dz};
}

template <typename scalar_t>
inline Jet3<scalar_t> operator+(const Jet3<scalar_t>& a, const Jet3<scalar_t>& b) {
  return {a.value + b.value, a.dx + b.dx, a.dy + b.dy, a.dz + b.dz};
}

template <typename scalar_t>
inline Jet3<scalar_t> operator-(const Jet3<scalar_t>& a, const Jet3<scalar_t>& b) {
  return {a.value - b.value, a.dx - b.dx, a.dy - b.dy, a.dz - b.dz};
}

template <typename scalar_t>
inline Jet3<scalar_t> operator-(const Jet3<scalar_t>& a) {
  return {-a.value, -a.dx, -a.dy, -a.dz};
}

template <typename scalar_t>
inline Jet3<scalar_t> operator*(const Jet3<scalar_t>& a, const Jet3<scalar_t>& b) {
  return {
      a.value * b.value,
      a.dx * b.value + a.value * b.dx,
      a.dy * b.value + a.value * b.dy,
      a.dz * b.value + a.value * b.dz,
  };
}

template <typename scalar_t>
inline Jet3<scalar_t> operator*(const Jet3<scalar_t>& a, double scale) {
  const scalar_t s = static_cast<scalar_t>(scale);
  return {a.value * s, a.dx * s, a.dy * s, a.dz * s};
}

template <typename scalar_t>
inline Jet3<scalar_t> operator*(double scale, const Jet3<scalar_t>& a) {
  return a * scale;
}

template <typename scalar_t>
inline Jet3<scalar_t> operator/(const Jet3<scalar_t>& a, double divisor) {
  return a * (1.0 / divisor);
}

template <typename scalar_t>
inline Jet3<scalar_t> operator/(const Jet3<scalar_t>& a, const Jet3<scalar_t>& b) {
  const scalar_t inv = static_cast<scalar_t>(1.0) / b.value;
  const scalar_t inv2 = inv * inv;
  return {
      a.value * inv,
      (a.dx * b.value - a.value * b.dx) * inv2,
      (a.dy * b.value - a.value * b.dy) * inv2,
      (a.dz * b.value - a.value * b.dz) * inv2,
  };
}

template <typename scalar_t>
inline Jet3<scalar_t> jet_square(const Jet3<scalar_t>& a) {
  return a * a;
}

template <typename scalar_t>
inline Jet3<scalar_t> jet_sqrt(const Jet3<scalar_t>& a) {
  const scalar_t root = static_cast<scalar_t>(std::sqrt(static_cast<double>(a.value)));
  const scalar_t inv = static_cast<scalar_t>(1.0) / root;
  return {root, a.dx * inv * static_cast<scalar_t>(0.5),
          a.dy * inv * static_cast<scalar_t>(0.5),
          a.dz * inv * static_cast<scalar_t>(0.5)};
}

template <typename scalar_t>
void sh3_reference(
    scalar_t x,
    scalar_t y,
    scalar_t z,
    scalar_t* out) {
  static const scalar_t sqrt3 = static_cast<scalar_t>(std::sqrt(3.0));
  static const scalar_t sqrt5 = static_cast<scalar_t>(std::sqrt(5.0));
  static const scalar_t sqrt7 = static_cast<scalar_t>(std::sqrt(7.0));
  static const scalar_t sqrt15 = static_cast<scalar_t>(std::sqrt(15.0));
  static const scalar_t sqrt42 = static_cast<scalar_t>(std::sqrt(42.0));
  static const scalar_t sqrt168 = static_cast<scalar_t>(std::sqrt(168.0));
  static const scalar_t inv_4pi = static_cast<scalar_t>(
      1.0 / std::sqrt(4.0 * 3.141592653589793238462643383279502884));

  const scalar_t sh_0_0 = static_cast<scalar_t>(1.0);
  const scalar_t sh_1_0 = sqrt3 * x;
  const scalar_t sh_1_1 = sqrt3 * y;
  const scalar_t sh_1_2 = sqrt3 * z;

  const scalar_t sh_2_0 = sqrt15 * x * z;
  const scalar_t sh_2_1 = sqrt15 * x * y;
  const scalar_t y2 = y * y;
  const scalar_t x2z2 = x * x + z * z;
  const scalar_t sh_2_2 = sqrt5 * (y2 - static_cast<scalar_t>(0.5) * x2z2);
  const scalar_t sh_2_3 = sqrt15 * y * z;
  const scalar_t sh_2_4 =
      static_cast<scalar_t>(0.5) * sqrt15 * (z * z - x * x);

  const scalar_t sh_3_0 =
      static_cast<scalar_t>(1.0 / 6.0) * sqrt42 *
      (sh_2_0 * z + sh_2_4 * x);
  const scalar_t sh_3_1 = sqrt7 * sh_2_0 * y;
  const scalar_t sh_3_2 =
      static_cast<scalar_t>(1.0 / 8.0) * sqrt168 *
      (static_cast<scalar_t>(4.0) * y2 - x2z2) * x;
  const scalar_t sh_3_3 =
      static_cast<scalar_t>(0.5) * sqrt7 * y *
      (static_cast<scalar_t>(2.0) * y2 - static_cast<scalar_t>(3.0) * x2z2);
  const scalar_t sh_3_4 =
      static_cast<scalar_t>(1.0 / 8.0) * sqrt168 * z *
      (static_cast<scalar_t>(4.0) * y2 - x2z2);
  const scalar_t sh_3_5 = sqrt7 * sh_2_4 * y;
  const scalar_t sh_3_6 =
      static_cast<scalar_t>(1.0 / 6.0) * sqrt42 *
      (sh_2_4 * z - sh_2_0 * x);

  out[0] = sh_0_0;
  out[1] = sh_1_0;
  out[2] = sh_1_1;
  out[3] = sh_1_2;
  out[4] = sh_2_0;
  out[5] = sh_2_1;
  out[6] = sh_2_2;
  out[7] = sh_2_3;
  out[8] = sh_2_4;
  out[9] = sh_3_0;
  out[10] = sh_3_1;
  out[11] = sh_3_2;
  out[12] = sh_3_3;
  out[13] = sh_3_4;
  out[14] = sh_3_5;
  out[15] = sh_3_6;

  if (inv_4pi != static_cast<scalar_t>(1.0)) {
    for (int64_t i = 0; i < kSh3Dim; ++i) {
      out[i] *= inv_4pi;
    }
  }
}

template <typename scalar_t>
void sh3_jet(
    const Jet3<scalar_t>& x,
    const Jet3<scalar_t>& y,
    const Jet3<scalar_t>& z,
    Jet3<scalar_t>* out) {
  static const double sqrt3 = std::sqrt(3.0);
  static const double sqrt5 = std::sqrt(5.0);
  static const double sqrt7 = std::sqrt(7.0);
  static const double sqrt15 = std::sqrt(15.0);
  static const double sqrt42 = std::sqrt(42.0);
  static const double sqrt168 = std::sqrt(168.0);
  static const double inv_4pi =
      1.0 / std::sqrt(4.0 * 3.141592653589793238462643383279502884);

  const Jet3<scalar_t> sh_0_0 =
      make_jet<scalar_t>(static_cast<scalar_t>(1.0), 0, 0, 0);
  const Jet3<scalar_t> sh_1_0 = x * sqrt3;
  const Jet3<scalar_t> sh_1_1 = y * sqrt3;
  const Jet3<scalar_t> sh_1_2 = z * sqrt3;

  const Jet3<scalar_t> sh_2_0 = (x * z) * sqrt15;
  const Jet3<scalar_t> sh_2_1 = (x * y) * sqrt15;
  const Jet3<scalar_t> y2 = jet_square(y);
  const Jet3<scalar_t> x2z2 = jet_square(x) + jet_square(z);
  const Jet3<scalar_t> sh_2_2 =
      (y2 - x2z2 * 0.5) * sqrt5;
  const Jet3<scalar_t> sh_2_3 = (y * z) * sqrt15;
  const Jet3<scalar_t> sh_2_4 =
      (jet_square(z) - jet_square(x)) * (0.5 * sqrt15);

  const Jet3<scalar_t> sh_3_0 =
      (sh_2_0 * z + sh_2_4 * x) * (sqrt42 / 6.0);
  const Jet3<scalar_t> sh_3_1 = (sh_2_0 * y) * sqrt7;
  const Jet3<scalar_t> sh_3_2 =
      ((y2 * 4.0 - x2z2) * x) * (sqrt168 / 8.0);
  const Jet3<scalar_t> sh_3_3 =
      (y * (y2 * 2.0 - x2z2 * 3.0)) * (0.5 * sqrt7);
  const Jet3<scalar_t> sh_3_4 =
      (z * (y2 * 4.0 - x2z2)) * (sqrt168 / 8.0);
  const Jet3<scalar_t> sh_3_5 = (sh_2_4 * y) * sqrt7;
  const Jet3<scalar_t> sh_3_6 =
      (sh_2_4 * z - sh_2_0 * x) * (sqrt42 / 6.0);

  out[0] = sh_0_0;
  out[1] = sh_1_0;
  out[2] = sh_1_1;
  out[3] = sh_1_2;
  out[4] = sh_2_0;
  out[5] = sh_2_1;
  out[6] = sh_2_2;
  out[7] = sh_2_3;
  out[8] = sh_2_4;
  out[9] = sh_3_0;
  out[10] = sh_3_1;
  out[11] = sh_3_2;
  out[12] = sh_3_3;
  out[13] = sh_3_4;
  out[14] = sh_3_5;
  out[15] = sh_3_6;

  for (int64_t i = 0; i < kSh3Dim; ++i) {
    out[i] = out[i] * inv_4pi;
  }
}

inline const int64_t* int64_data(const torch::Tensor& tensor) {
  TORCH_CHECK(tensor.device().is_cpu(), "metadata tensors must be CPU");
  TORCH_CHECK(tensor.scalar_type() == torch::kInt64, "metadata must be int64");
  return tensor.data_ptr<int64_t>();
}

inline const double* double_data(const torch::Tensor& tensor) {
  TORCH_CHECK(tensor.device().is_cpu(), "metadata tensors must be CPU");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat64, "metadata must be float64");
  return tensor.data_ptr<double>();
}

#if NATIVE_CPU_HAS_CBLAS

template <typename scalar_t>
struct CblasGemm;

template <>
struct CblasGemm<float> {
  static void run(
      CBLAS_TRANSPOSE trans_a,
      CBLAS_TRANSPOSE trans_b,
      int64_t m,
      int64_t n,
      int64_t k,
      float alpha,
      const float* a,
      int64_t lda,
      const float* b,
      int64_t ldb,
      float beta,
      float* c,
      int64_t ldc) {
    cblas_sgemm(
        CblasRowMajor,
        trans_a,
        trans_b,
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        alpha,
        a,
        static_cast<int>(lda),
        b,
        static_cast<int>(ldb),
        beta,
        c,
        static_cast<int>(ldc));
  }
};

template <>
struct CblasGemm<double> {
  static void run(
      CBLAS_TRANSPOSE trans_a,
      CBLAS_TRANSPOSE trans_b,
      int64_t m,
      int64_t n,
      int64_t k,
      double alpha,
      const double* a,
      int64_t lda,
      const double* b,
      int64_t ldb,
      double beta,
      double* c,
      int64_t ldc) {
    cblas_dgemm(
        CblasRowMajor,
        trans_a,
        trans_b,
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        alpha,
        a,
        static_cast<int>(lda),
        b,
        static_cast<int>(ldb),
        beta,
        c,
        static_cast<int>(ldc));
  }
};

#endif

template <typename scalar_t>
inline void cpu_gemm(
    bool transpose_a,
    bool transpose_b,
    int64_t m,
    int64_t n,
    int64_t k,
    const scalar_t* a,
    int64_t lda,
    const scalar_t* b,
    int64_t ldb,
    scalar_t* c,
    int64_t ldc) {
#if NATIVE_CPU_HAS_CBLAS
  CblasGemm<scalar_t>::run(
      transpose_a ? CblasTrans : CblasNoTrans,
      transpose_b ? CblasTrans : CblasNoTrans,
      m,
      n,
      k,
      static_cast<scalar_t>(1.0),
      a,
      lda,
      b,
      ldb,
      static_cast<scalar_t>(0.0),
      c,
      ldc);
#else
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      scalar_t acc = 0;
      for (int64_t p = 0; p < k; ++p) {
        const scalar_t av =
            transpose_a ? a[p * lda + i] : a[i * lda + p];
        const scalar_t bv =
            transpose_b ? b[j * ldb + p] : b[p * ldb + j];
        acc += av * bv;
      }
      c[i * ldc + j] = acc;
    }
  }
#endif
}

struct TensorShape {
  const int64_t* offset;
  const int64_t* mul;
  const int64_t* dim;
};

template <typename scalar_t>
inline scalar_t load_weight(
    const scalar_t* weight,
    int64_t batch,
    int64_t weight_numel,
    bool shared_weights,
    int64_t index) {
  return weight[(shared_weights ? 0 : batch * weight_numel) + index];
}

template <typename F>
inline void native_parallel_for(
    int64_t begin,
    int64_t end,
    int64_t grain_size,
    const F& function) {
  if (begin >= end) {
    return;
  }
#if defined(_OPENMP)
  const int64_t work = end - begin;
  const int64_t available_threads = static_cast<int64_t>(omp_get_max_threads());
  // The old grain values were tuned for very large batches.  For batch=8
  // they often reduced the requested team to one thread.  Work size is the
  // real parallel upper bound; keep OpenMP overhead bounded by work itself.
  const int64_t requested = std::min<int64_t>(available_threads, work);
  if (requested <= 1) {
    function(begin, end);
    return;
  }
#pragma omp parallel num_threads(requested)
  {
    const int64_t threads = omp_get_num_threads();
    const int64_t thread_id = omp_get_thread_num();
    const int64_t chunk = (work + threads - 1) / threads;
    const int64_t thread_begin = begin + thread_id * chunk;
    if (thread_begin < end) {
      function(thread_begin, std::min(end, thread_begin + chunk));
    }
  }
#else
  at::parallel_for(begin, end, grain_size, function);
#endif
}

template <typename scalar_t>
inline void cpu_gemm_packed_nn(
    int64_t m,
    int64_t n,
    int64_t k,
    const scalar_t* a,
    int64_t lda,
    const scalar_t* b,
    int64_t ldb,
    scalar_t* c,
    int64_t ldc) {
  native_parallel_for(0, m, 1, [&](int64_t begin, int64_t end) {
    std::vector<scalar_t> acc(n);
    for (int64_t i = begin; i < end; ++i) {
      std::fill(acc.begin(), acc.end(), static_cast<scalar_t>(0));
      for (int64_t p = 0; p < k; ++p) {
        const scalar_t av = a[i * lda + p];
        const scalar_t* b_row = b + p * ldb;
        for (int64_t j = 0; j < n; ++j) {
          acc[j] += av * b_row[j];
        }
      }
      scalar_t* c_row = c + i * ldc;
      for (int64_t j = 0; j < n; ++j) {
        c_row[j] = acc[j];
      }
    }
  });
}

template <typename scalar_t>
inline void cpu_gemm_packed_tn(
    int64_t m,
    int64_t n,
    int64_t k,
    const scalar_t* a,
    int64_t lda,
    const scalar_t* b,
    int64_t ldb,
    scalar_t* c,
    int64_t ldc) {
  native_parallel_for(0, m, 1, [&](int64_t begin, int64_t end) {
    std::vector<scalar_t> acc(n);
    for (int64_t i = begin; i < end; ++i) {
      std::fill(acc.begin(), acc.end(), static_cast<scalar_t>(0));
      for (int64_t p = 0; p < k; ++p) {
        const scalar_t av = a[p * lda + i];
        const scalar_t* b_row = b + p * ldb;
        for (int64_t j = 0; j < n; ++j) {
          acc[j] += av * b_row[j];
        }
      }
      scalar_t* c_row = c + i * ldc;
      for (int64_t j = 0; j < n; ++j) {
        c_row[j] = acc[j];
      }
    }
  });
}

inline void parallel_zero_(torch::Tensor tensor) {
  if (!tensor.is_contiguous()) {
    tensor = tensor.contiguous();
  }
  const int64_t count = tensor.numel();
  if (count == 0) {
    return;
  }
  AT_DISPATCH_FLOATING_TYPES(
      tensor.scalar_type(), "parallel_zero", [&] {
        auto* ptr = tensor.data_ptr<scalar_t>();
        native_parallel_for(0, count, 4096, [&](int64_t begin, int64_t end) {
          std::fill(
              ptr + begin,
              ptr + end,
              static_cast<scalar_t>(0));
        });
      });
}

inline torch::Tensor parallel_zeros_like(const torch::Tensor& reference) {
  auto out = torch::empty_like(reference);
  parallel_zero_(out);
  return out;
}

inline torch::Tensor parallel_zeros(
    at::IntArrayRef sizes,
    const torch::TensorOptions& options) {
  auto out = torch::empty(sizes, options);
  parallel_zero_(out);
  return out;
}

template <typename scalar_t>
inline void radial_mlp_linear_forward(
    const scalar_t* input,
    const scalar_t* weight,
    scalar_t* output,
    int64_t rows,
    int64_t in_dim,
    int64_t out_dim) {
  native_parallel_for(0, rows, 4, [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      const scalar_t* input_row = input + row * in_dim;
      scalar_t* output_row = output + row * out_dim;
      std::fill(output_row, output_row + out_dim, static_cast<scalar_t>(0));
      for (int64_t input_index = 0; input_index < in_dim; ++input_index) {
        const scalar_t value = input_row[input_index];
        const scalar_t* weight_row = weight + input_index * out_dim;
        for (int64_t output_index = 0; output_index < out_dim; ++output_index) {
          output_row[output_index] += value * weight_row[output_index];
        }
      }
    }
  });
}

template <typename scalar_t>
inline void radial_mlp_linear_backward(
    const scalar_t* grad,
    const scalar_t* weight,
    scalar_t* grad_input,
    int64_t rows,
    int64_t in_dim,
    int64_t out_dim) {
  native_parallel_for(0, rows, 4, [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      const scalar_t* grad_row = grad + row * out_dim;
      scalar_t* grad_input_row = grad_input + row * in_dim;
      for (int64_t input_index = 0; input_index < in_dim; ++input_index) {
        const scalar_t* weight_row = weight + input_index * out_dim;
        scalar_t acc = 0;
        for (int64_t output_index = 0; output_index < out_dim; ++output_index) {
          acc += grad_row[output_index] * weight_row[output_index];
        }
        grad_input_row[input_index] = acc;
      }
    }
  });
}

}  // namespace


torch::Tensor sh3_forward(torch::Tensor x, bool normalize) {
  TORCH_CHECK(x.device().is_cpu(), "sh3_forward expects a CPU tensor");
  TORCH_CHECK(x.dim() == 2 && x.size(1) == 3, "x must have shape [B, 3]");
  auto x_contig = x.contiguous();
  auto out = torch::empty({x_contig.size(0), kSh3Dim}, x_contig.options());

  AT_DISPATCH_FLOATING_TYPES(x_contig.scalar_type(), "sh3_forward", [&] {
    const auto* x_ptr = x_contig.data_ptr<scalar_t>();
    auto* out_ptr = out.data_ptr<scalar_t>();
    const int64_t batch = x_contig.size(0);

    native_parallel_for(0, batch, 64, [&](int64_t begin, int64_t end) {
      for (int64_t b = begin; b < end; ++b) {
        scalar_t px = x_ptr[b * 3];
        scalar_t py = x_ptr[b * 3 + 1];
        scalar_t pz = x_ptr[b * 3 + 2];
        if (normalize) {
          const scalar_t norm = std::sqrt(sqr(px) + sqr(py) + sqr(pz));
          const scalar_t denom = std::max(norm, static_cast<scalar_t>(kShEps));
          px /= denom;
          py /= denom;
          pz /= denom;
        }
        sh3_reference<scalar_t>(px, py, pz, out_ptr + b * kSh3Dim);
      }
    });
  });
  return out;
}


torch::Tensor sh3_backward(torch::Tensor x, torch::Tensor grad_out, bool normalize) {
  TORCH_CHECK(x.device().is_cpu(), "sh3_backward expects a CPU tensor");
  TORCH_CHECK(grad_out.device().is_cpu(), "grad_out must be CPU");
  TORCH_CHECK(x.dim() == 2 && x.size(1) == 3, "x must have shape [B, 3]");
  TORCH_CHECK(
      grad_out.dim() == 2 && grad_out.size(0) == x.size(0) &&
          grad_out.size(1) == kSh3Dim,
      "grad_out must have shape [B, 16]");
  auto x_contig = x.contiguous();
  auto grad_contig = grad_out.contiguous();
  auto grad_x = torch::empty_like(x_contig);

  AT_DISPATCH_FLOATING_TYPES(x_contig.scalar_type(), "sh3_backward", [&] {
    const auto* x_ptr = x_contig.data_ptr<scalar_t>();
    const auto* grad_ptr = grad_contig.data_ptr<scalar_t>();
    auto* grad_x_ptr = grad_x.data_ptr<scalar_t>();
    const int64_t batch = x_contig.size(0);

    native_parallel_for(0, batch, 64, [&](int64_t begin, int64_t end) {
      for (int64_t b = begin; b < end; ++b) {
        const scalar_t raw_x = x_ptr[b * 3];
        const scalar_t raw_y = x_ptr[b * 3 + 1];
        const scalar_t raw_z = x_ptr[b * 3 + 2];

        Jet3<scalar_t> px = make_jet<scalar_t>(
            raw_x, static_cast<scalar_t>(1.0), 0, 0);
        Jet3<scalar_t> py = make_jet<scalar_t>(
            raw_y, 0, static_cast<scalar_t>(1.0), 0);
        Jet3<scalar_t> pz = make_jet<scalar_t>(
            raw_z, 0, 0, static_cast<scalar_t>(1.0));

        if (normalize) {
          const scalar_t norm_value =
              std::sqrt(sqr(raw_x) + sqr(raw_y) + sqr(raw_z));
          if (norm_value >= static_cast<scalar_t>(kShEps)) {
            const Jet3<scalar_t> radius = jet_sqrt(
                jet_square(px) + jet_square(py) + jet_square(pz));
            px = px / radius;
            py = py / radius;
            pz = pz / radius;
          } else {
            const scalar_t inv = static_cast<scalar_t>(1.0) /
                static_cast<scalar_t>(kShEps);
            px = px * static_cast<double>(inv);
            py = py * static_cast<double>(inv);
            pz = pz * static_cast<double>(inv);
          }
        }

        Jet3<scalar_t> values[kSh3Dim];
        sh3_jet<scalar_t>(px, py, pz, values);

        scalar_t gx = 0;
        scalar_t gy = 0;
        scalar_t gz = 0;
        for (int64_t c = 0; c < kSh3Dim; ++c) {
          const scalar_t upstream = grad_ptr[b * kSh3Dim + c];
          gx += upstream * values[c].dx;
          gy += upstream * values[c].dy;
          gz += upstream * values[c].dz;
        }
        grad_x_ptr[b * 3] = gx;
        grad_x_ptr[b * 3 + 1] = gy;
        grad_x_ptr[b * 3 + 2] = gz;
      }
    });
  });
  return grad_x;
}


torch::Tensor fctp_forward(
    torch::Tensor x1,
    torch::Tensor x2,
    torch::Tensor weight,
    torch::Tensor vstar,
    torch::Tensor in1_offset,
    torch::Tensor in1_mul,
    torch::Tensor in1_dim,
    torch::Tensor in2_offset,
    torch::Tensor in2_mul,
    torch::Tensor in2_dim,
    torch::Tensor out_offset,
    torch::Tensor out_mul,
    torch::Tensor out_dim,
    torch::Tensor ins_i1,
    torch::Tensor ins_i2,
    torch::Tensor ins_out,
    torch::Tensor ins_weight_offset,
    torch::Tensor ins_path_weight,
    std::vector<torch::Tensor> w3js,
    bool shared_weights,
    bool one_hot) {
  TORCH_CHECK(x1.device().is_cpu() && x2.device().is_cpu(), "inputs must be CPU");
  TORCH_CHECK(weight.device().is_cpu(), "weight must be CPU");
  TORCH_CHECK(x1.dim() == 2 && x2.dim() == 2, "x1/x2 must be 2D");
  TORCH_CHECK(weight.dim() == 1 || weight.dim() == 2, "weight must be 1D or 2D");
  TORCH_CHECK(x1.size(0) == x2.size(0), "batch mismatch");
  const int64_t batch = x1.size(0);
  const int64_t weight_numel = weight.numel() / (shared_weights ? 1 : batch);
  const int64_t ins_count = ins_i1.numel();
  TORCH_CHECK(
      ins_i2.numel() == ins_count && ins_out.numel() == ins_count &&
          ins_weight_offset.numel() == ins_count &&
          ins_path_weight.numel() == ins_count,
      "instruction metadata length mismatch");
  TORCH_CHECK(static_cast<int64_t>(w3js.size()) == ins_count, "w3j count mismatch");
  if (one_hot) {
    TORCH_CHECK(
        vstar.device().is_cpu() && vstar.scalar_type() == torch::kInt64 &&
            vstar.numel() == batch,
        "vstar must be int64 [B]");
  }

  auto x1_contig = x1.contiguous();
  auto x2_contig = x2.contiguous();
  auto weight_contig = weight.contiguous();
  auto vstar_contig = one_hot ? vstar.contiguous() : vstar;
  auto out_sizes = out_dim.clone();
  int64_t out_total = 0;
  for (int64_t i = 0; i < out_sizes.numel(); ++i) {
    out_total += out_sizes.data_ptr<int64_t>()[i] *
        out_mul.data_ptr<int64_t>()[i];
  }
  auto out = torch::empty({batch, out_total}, x1_contig.options());

  const auto* in1_offset_ptr = int64_data(in1_offset);
  const auto* in1_mul_ptr = int64_data(in1_mul);
  const auto* in1_dim_ptr = int64_data(in1_dim);
  const auto* in2_offset_ptr = int64_data(in2_offset);
  const auto* in2_mul_ptr = int64_data(in2_mul);
  const auto* in2_dim_ptr = int64_data(in2_dim);
  const auto* out_offset_ptr = int64_data(out_offset);
  const auto* out_mul_ptr = int64_data(out_mul);
  const auto* out_dim_ptr = int64_data(out_dim);
  const auto* ins_i1_ptr = int64_data(ins_i1);
  const auto* ins_i2_ptr = int64_data(ins_i2);
  const auto* ins_out_ptr = int64_data(ins_out);
  const auto* ins_weight_offset_ptr = int64_data(ins_weight_offset);
  const auto* ins_path_weight_ptr = double_data(ins_path_weight);

  AT_DISPATCH_FLOATING_TYPES(x1_contig.scalar_type(), "fctp_forward", [&] {
    const auto* x1_ptr = x1_contig.data_ptr<scalar_t>();
    const auto* x2_ptr = x2_contig.data_ptr<scalar_t>();
    const auto* weight_ptr = weight_contig.data_ptr<scalar_t>();
    auto* out_ptr = out.data_ptr<scalar_t>();
    const auto* vstar_ptr =
        one_hot ? vstar_contig.data_ptr<int64_t>() : static_cast<const int64_t*>(nullptr);

    native_parallel_for(0, batch, 8, [&](int64_t begin, int64_t end) {
      std::vector<scalar_t> accum(out_total);
      for (int64_t b = begin; b < end; ++b) {
        std::fill(accum.begin(), accum.end(), static_cast<scalar_t>(0));
        for (int64_t ins = 0; ins < ins_count; ++ins) {
          const int64_t i1 = ins_i1_ptr[ins];
          const int64_t i2 = ins_i2_ptr[ins];
          const int64_t io = ins_out_ptr[ins];
          const int64_t mul1 = in1_mul_ptr[i1];
          const int64_t mul2 = in2_mul_ptr[i2];
          const int64_t mul_out = out_mul_ptr[io];
          const int64_t dim1 = in1_dim_ptr[i1];
          const int64_t dim2 = in2_dim_ptr[i2];
          const int64_t dim_out = out_dim_ptr[io];
          const auto* w3j = w3js[ins].data_ptr<scalar_t>();
          const scalar_t path_weight =
              static_cast<scalar_t>(ins_path_weight_ptr[ins]);
          const int64_t woff = ins_weight_offset_ptr[ins];

          for (int64_t u = 0; u < mul1; ++u) {
            for (int64_t v = 0; v < mul2; ++v) {
              if (one_hot && v != vstar_ptr[b]) {
                continue;
              }
              for (int64_t wo = 0; wo < mul_out; ++wo) {
                const int64_t w_index =
                    woff + ((u * mul2 + v) * mul_out + wo);
                const scalar_t w_value = load_weight(
                    weight_ptr, b, weight_numel, shared_weights, w_index);
                auto* dst = accum.data() + out_offset_ptr[io] + wo * dim_out;
                for (int64_t i = 0; i < dim1; ++i) {
                  const scalar_t x1_value =
                      x1_ptr[b * x1_contig.size(1) +
                             in1_offset_ptr[i1] + u * dim1 + i];
                  for (int64_t j = 0; j < dim2; ++j) {
                    const scalar_t x2_value =
                        x2_ptr[b * x2_contig.size(1) +
                               in2_offset_ptr[i2] + v * dim2 + j];
                    const scalar_t base =
                        w_value * x1_value * x2_value * path_weight;
                    if (base == static_cast<scalar_t>(0)) {
                      continue;
                    }
                    for (int64_t k = 0; k < dim_out; ++k) {
                      dst[k] += base * w3j[(i * dim2 + j) * dim_out + k];
                    }
                  }
                }
              }
            }
          }
        }
        for (int64_t col = 0; col < out_total; ++col) {
          out_ptr[b * out_total + col] = accum[col];
        }
      }
    });
  });
  return out;
}


torch::Tensor fctp_backward_x(
    torch::Tensor grad_out,
    torch::Tensor x1,
    torch::Tensor x2,
    torch::Tensor weight,
    torch::Tensor vstar,
    torch::Tensor in1_offset,
    torch::Tensor in1_mul,
    torch::Tensor in1_dim,
    torch::Tensor in2_offset,
    torch::Tensor in2_mul,
    torch::Tensor in2_dim,
    torch::Tensor out_offset,
    torch::Tensor out_mul,
    torch::Tensor out_dim,
    torch::Tensor ins_i1,
    torch::Tensor ins_i2,
    torch::Tensor ins_out,
    torch::Tensor ins_weight_offset,
    torch::Tensor ins_path_weight,
    std::vector<torch::Tensor> w3js,
    bool shared_weights,
    bool one_hot) {
  TORCH_CHECK(grad_out.device().is_cpu(), "grad_out must be CPU");
  TORCH_CHECK(x1.device().is_cpu() && x2.device().is_cpu(), "inputs must be CPU");
  TORCH_CHECK(weight.device().is_cpu(), "weight must be CPU");
  auto grad_contig = grad_out.contiguous();
  auto x1_contig = x1.contiguous();
  auto x2_contig = x2.contiguous();
  auto weight_contig = weight.contiguous();
  auto vstar_contig = one_hot ? vstar.contiguous() : vstar;
  auto grad_x1 = parallel_zeros_like(x1_contig);

  const int64_t batch = x1_contig.size(0);
  const int64_t weight_numel = weight.numel() / (shared_weights ? 1 : batch);
  const int64_t ins_count = ins_i1.numel();
  const auto* in1_offset_ptr = int64_data(in1_offset);
  const auto* in1_mul_ptr = int64_data(in1_mul);
  const auto* in1_dim_ptr = int64_data(in1_dim);
  const auto* in2_offset_ptr = int64_data(in2_offset);
  const auto* in2_mul_ptr = int64_data(in2_mul);
  const auto* in2_dim_ptr = int64_data(in2_dim);
  const auto* out_offset_ptr = int64_data(out_offset);
  const auto* out_mul_ptr = int64_data(out_mul);
  const auto* out_dim_ptr = int64_data(out_dim);
  const auto* ins_i1_ptr = int64_data(ins_i1);
  const auto* ins_i2_ptr = int64_data(ins_i2);
  const auto* ins_out_ptr = int64_data(ins_out);
  const auto* ins_weight_offset_ptr = int64_data(ins_weight_offset);
  const auto* ins_path_weight_ptr = double_data(ins_path_weight);

  AT_DISPATCH_FLOATING_TYPES(x1_contig.scalar_type(), "fctp_backward_x", [&] {
    const auto* grad_ptr = grad_contig.data_ptr<scalar_t>();
    const auto* x1_ptr = x1_contig.data_ptr<scalar_t>();
    const auto* x2_ptr = x2_contig.data_ptr<scalar_t>();
    const auto* weight_ptr = weight_contig.data_ptr<scalar_t>();
    auto* grad_x1_ptr = grad_x1.data_ptr<scalar_t>();
    const auto* vstar_ptr =
        one_hot ? vstar_contig.data_ptr<int64_t>() : static_cast<const int64_t*>(nullptr);

    native_parallel_for(0, batch, 8, [&](int64_t begin, int64_t end) {
      for (int64_t b = begin; b < end; ++b) {
        for (int64_t ins = 0; ins < ins_count; ++ins) {
          const int64_t i1 = ins_i1_ptr[ins];
          const int64_t i2 = ins_i2_ptr[ins];
          const int64_t io = ins_out_ptr[ins];
          const int64_t mul1 = in1_mul_ptr[i1];
          const int64_t mul2 = in2_mul_ptr[i2];
          const int64_t mul_out = out_mul_ptr[io];
          const int64_t dim1 = in1_dim_ptr[i1];
          const int64_t dim2 = in2_dim_ptr[i2];
          const int64_t dim_out = out_dim_ptr[io];
          const auto* w3j = w3js[ins].data_ptr<scalar_t>();
          const scalar_t path_weight =
              static_cast<scalar_t>(ins_path_weight_ptr[ins]);
          const int64_t woff = ins_weight_offset_ptr[ins];

          for (int64_t u = 0; u < mul1; ++u) {
            for (int64_t i = 0; i < dim1; ++i) {
              scalar_t accum = 0;
              for (int64_t v = 0; v < mul2; ++v) {
                if (one_hot && v != vstar_ptr[b]) {
                  continue;
                }
                for (int64_t j = 0; j < dim2; ++j) {
                  const scalar_t x2_value =
                      x2_ptr[b * x2_contig.size(1) +
                             in2_offset_ptr[i2] + v * dim2 + j];
                  if (x2_value == static_cast<scalar_t>(0)) {
                    continue;
                  }
                  for (int64_t wo = 0; wo < mul_out; ++wo) {
                    const scalar_t w_value = load_weight(
                        weight_ptr, b, weight_numel, shared_weights,
                        woff + ((u * mul2 + v) * mul_out + wo));
                    const auto* grad_segment =
                        grad_ptr + b * grad_contig.size(1) +
                        out_offset_ptr[io] + wo * dim_out;
                    const auto* w3j_base = w3j + (i * dim2 + j) * dim_out;
                    for (int64_t k = 0; k < dim_out; ++k) {
                      accum += grad_segment[k] * w_value * x2_value *
                          path_weight * w3j_base[k];
                    }
                  }
                }
              }
              grad_x1_ptr[b * x1_contig.size(1) +
                         in1_offset_ptr[i1] + u * dim1 + i] += accum;
            }
          }
        }
      }
    });
  });
  return grad_x1;
}


torch::Tensor tpu_forward(
    torch::Tensor x1,
    torch::Tensor x2,
    torch::Tensor weight,
    torch::Tensor in1_offset,
    torch::Tensor in1_mul,
    torch::Tensor in1_dim,
    torch::Tensor in2_offset,
    torch::Tensor in2_mul,
    torch::Tensor in2_dim,
    torch::Tensor out_offset,
    torch::Tensor out_mul,
    torch::Tensor out_dim,
    torch::Tensor ins_i1,
    torch::Tensor ins_i2,
    torch::Tensor ins_out,
    torch::Tensor ins_weight_offset,
    torch::Tensor ins_path_weight,
    std::vector<torch::Tensor> w3js,
    bool shared_weights,
    bool disjoint_output) {
  auto x1_contig = x1.contiguous();
  auto x2_contig = x2.contiguous();
  auto weight_contig = weight.contiguous();
  const int64_t batch = x1_contig.size(0);
  const int64_t weight_numel = weight.numel() / (shared_weights ? 1 : batch);
  int64_t out_total = 0;
  for (int64_t i = 0; i < out_dim.numel(); ++i) {
    out_total += out_dim.data_ptr<int64_t>()[i] *
        out_mul.data_ptr<int64_t>()[i];
  }
  auto out = disjoint_output
      ? torch::empty({batch, out_total}, x1_contig.options())
      : parallel_zeros({batch, out_total}, x1_contig.options());
  const int64_t ins_count = ins_i1.numel();
  const auto* in1_offset_ptr = int64_data(in1_offset);
  const auto* in1_mul_ptr = int64_data(in1_mul);
  const auto* in1_dim_ptr = int64_data(in1_dim);
  const auto* in2_offset_ptr = int64_data(in2_offset);
  const auto* in2_mul_ptr = int64_data(in2_mul);
  const auto* in2_dim_ptr = int64_data(in2_dim);
  const auto* out_offset_ptr = int64_data(out_offset);
  const auto* out_mul_ptr = int64_data(out_mul);
  const auto* out_dim_ptr = int64_data(out_dim);
  const auto* ins_i1_ptr = int64_data(ins_i1);
  const auto* ins_i2_ptr = int64_data(ins_i2);
  const auto* ins_out_ptr = int64_data(ins_out);
  const auto* ins_weight_offset_ptr = int64_data(ins_weight_offset);
  const auto* ins_path_weight_ptr = double_data(ins_path_weight);
  bool separable = true;
  std::vector<int64_t> transformed_offsets(ins_count + 1, 0);
  for (int64_t ins = 0; ins < ins_count; ++ins) {
    const int64_t i1 = ins_i1_ptr[ins];
    const int64_t i2 = ins_i2_ptr[ins];
    const int64_t io = ins_out_ptr[ins];
    if (in1_dim_ptr[i1] != 1 || in2_mul_ptr[i2] != 1) {
      separable = false;
      break;
    }
    transformed_offsets[ins + 1] =
        transformed_offsets[ins] + out_dim_ptr[io];
  }

  AT_DISPATCH_FLOATING_TYPES(x1_contig.scalar_type(), "tpu_forward", [&] {
    const auto* x1_ptr = x1_contig.data_ptr<scalar_t>();
    const auto* x2_ptr = x2_contig.data_ptr<scalar_t>();
    const auto* weight_ptr = weight_contig.data_ptr<scalar_t>();
    auto* out_ptr = out.data_ptr<scalar_t>();
    if (separable) {
      const int64_t transformed_total = transformed_offsets.back();
      native_parallel_for(0, batch, 16, [&](int64_t begin, int64_t end) {
        std::vector<scalar_t> transformed(transformed_total);
        for (int64_t b = begin; b < end; ++b) {
          for (int64_t ins = 0; ins < ins_count; ++ins) {
            const int64_t i1 = ins_i1_ptr[ins];
            const int64_t i2 = ins_i2_ptr[ins];
            const int64_t io = ins_out_ptr[ins];
            const int64_t m1 = in1_mul_ptr[i1];
            const int64_t d2 = in2_dim_ptr[i2];
            const int64_t do_ = out_dim_ptr[io];
            const auto* w3j = w3js[ins].data_ptr<scalar_t>();
            const scalar_t coeff =
                static_cast<scalar_t>(ins_path_weight_ptr[ins]);
            const int64_t woff = ins_weight_offset_ptr[ins];
            auto* transformed_ins =
                transformed.data() + transformed_offsets[ins];
            const auto* x2_values =
                x2_ptr + b * x2_contig.size(1) + in2_offset_ptr[i2];
            for (int64_t k = 0; k < do_; ++k) {
              scalar_t acc = 0;
              for (int64_t j = 0; j < d2; ++j) {
                acc += x2_values[j] * w3j[j * do_ + k];
              }
              transformed_ins[k] = acc * coeff;
            }
            const auto* x1_values =
                x1_ptr + b * x1_contig.size(1) + in1_offset_ptr[i1];
            auto* out_values =
                out_ptr + b * out_total + out_offset_ptr[io];
            for (int64_t u = 0; u < m1; ++u) {
              const scalar_t base = load_weight(
                  weight_ptr, b, weight_numel, shared_weights, woff + u) *
                  x1_values[u];
              auto* dst = out_values + u * do_;
              for (int64_t k = 0; k < do_; ++k) {
                if (disjoint_output) {
                  dst[k] = base * transformed_ins[k];
                } else {
                  dst[k] += base * transformed_ins[k];
                }
              }
            }
          }
        }
      });
    } else {
      native_parallel_for(0, batch, 8, [&](int64_t begin, int64_t end) {
        std::vector<scalar_t> accum(out_total);
        for (int64_t b = begin; b < end; ++b) {
          std::fill(accum.begin(), accum.end(), static_cast<scalar_t>(0));
          for (int64_t ins = 0; ins < ins_count; ++ins) {
            const int64_t i1 = ins_i1_ptr[ins];
            const int64_t i2 = ins_i2_ptr[ins];
            const int64_t io = ins_out_ptr[ins];
            const int64_t m1 = in1_mul_ptr[i1];
            const int64_t m2 = in2_mul_ptr[i2];
            const int64_t d1 = in1_dim_ptr[i1];
            const int64_t d2 = in2_dim_ptr[i2];
            const int64_t do_ = out_dim_ptr[io];
            const int64_t mo = out_mul_ptr[io];
            const auto* w3j = w3js[ins].data_ptr<scalar_t>();
            const scalar_t coeff =
                static_cast<scalar_t>(ins_path_weight_ptr[ins]);
            const int64_t woff = ins_weight_offset_ptr[ins];
            for (int64_t u = 0; u < m1; ++u) {
              for (int64_t v = 0; v < m2; ++v) {
                const scalar_t w_value = load_weight(
                    weight_ptr, b, weight_numel, shared_weights,
                    woff + u * m2 + v);
                for (int64_t i = 0; i < d1; ++i) {
                  const scalar_t x1_value =
                      x1_ptr[b * x1_contig.size(1) +
                             in1_offset_ptr[i1] + u * d1 + i];
                  for (int64_t j = 0; j < d2; ++j) {
                    const scalar_t x2_value =
                        x2_ptr[b * x2_contig.size(1) +
                               in2_offset_ptr[i2] + v * d2 + j];
                    const scalar_t base =
                        w_value * x1_value * x2_value * coeff;
                    auto* dst =
                        accum.data() + out_offset_ptr[io] + u * do_;
                    const auto* w3j_row = w3j + (i * d2 + j) * do_;
                    for (int64_t k = 0; k < do_; ++k) {
                      dst[k] += base * w3j_row[k];
                    }
                  }
                }
              }
            }
            TORCH_CHECK(mo == m1, "uvu output multiplicity mismatch");
          }
          for (int64_t col = 0; col < out_total; ++col) {
            out_ptr[b * out_total + col] = accum[col];
          }
        }
      });
    }
  });
  return out;
}


std::vector<torch::Tensor> tpu_backward(
    torch::Tensor grad_out,
    torch::Tensor x1,
    torch::Tensor x2,
    torch::Tensor weight,
    torch::Tensor in1_offset,
    torch::Tensor in1_mul,
    torch::Tensor in1_dim,
    torch::Tensor in2_offset,
    torch::Tensor in2_mul,
    torch::Tensor in2_dim,
    torch::Tensor out_offset,
    torch::Tensor out_mul,
    torch::Tensor out_dim,
    torch::Tensor ins_i1,
    torch::Tensor ins_i2,
    torch::Tensor ins_out,
    torch::Tensor ins_weight_offset,
    torch::Tensor ins_path_weight,
    std::vector<torch::Tensor> w3js,
    bool shared_weights) {
  auto grad_contig = grad_out.contiguous();
  auto x1_contig = x1.contiguous();
  auto x2_contig = x2.contiguous();
  auto weight_contig = weight.contiguous();
  const int64_t batch = grad_contig.size(0);
  const int64_t weight_numel = weight.numel() / (shared_weights ? 1 : batch);
  auto grad_x1 = parallel_zeros_like(x1_contig);
  const int64_t ins_count = ins_i1.numel();
  const auto* in1_offset_ptr = int64_data(in1_offset);
  const auto* in1_mul_ptr = int64_data(in1_mul);
  const auto* in1_dim_ptr = int64_data(in1_dim);
  const auto* in2_offset_ptr = int64_data(in2_offset);
  const auto* in2_mul_ptr = int64_data(in2_mul);
  const auto* in2_dim_ptr = int64_data(in2_dim);
  const auto* out_offset_ptr = int64_data(out_offset);
  const auto* out_dim_ptr = int64_data(out_dim);
  const auto* ins_i1_ptr = int64_data(ins_i1);
  const auto* ins_i2_ptr = int64_data(ins_i2);
  const auto* ins_out_ptr = int64_data(ins_out);
  const auto* ins_weight_offset_ptr = int64_data(ins_weight_offset);
  const auto* ins_path_weight_ptr = double_data(ins_path_weight);
  auto disjoint_ranges = [&](const int64_t* offsets,
                             const int64_t* starts,
                             const int64_t* sizes) {
    for (int64_t i = 0; i < ins_count; ++i) {
      const int64_t a0 = offsets[i] + starts[i];
      const int64_t a1 = a0 + sizes[i];
      for (int64_t j = i + 1; j < ins_count; ++j) {
        const int64_t b0 = offsets[j] + starts[j];
        const int64_t b1 = b0 + sizes[j];
        if (a0 < b1 && b0 < a1) {
          return false;
        }
      }
    }
    return true;
  };
  std::vector<int64_t> in2_sizes(ins_count);
  std::vector<int64_t> weight_sizes(ins_count);
  for (int64_t ins = 0; ins < ins_count; ++ins) {
    in2_sizes[ins] = in2_mul_ptr[ins_i2_ptr[ins]] *
        in2_dim_ptr[ins_i2_ptr[ins]];
    weight_sizes[ins] = in1_mul_ptr[ins_i1_ptr[ins]] *
        in2_mul_ptr[ins_i2_ptr[ins]];
  }
  const bool disjoint_x2 = disjoint_ranges(
      in2_offset_ptr, std::vector<int64_t>(ins_count, 0).data(),
      in2_sizes.data());
  const bool disjoint_weight = disjoint_ranges(
      std::vector<int64_t>(ins_count, 0).data(), ins_weight_offset_ptr,
      weight_sizes.data());
  auto grad_x2 = disjoint_x2
      ? torch::empty_like(x2_contig)
      : parallel_zeros_like(x2_contig);
  auto grad_weight = disjoint_weight
      ? torch::empty_like(weight_contig)
      : parallel_zeros_like(weight_contig);
  bool separable = true;
  std::vector<int64_t> transformed_offsets(ins_count + 1, 0);
  for (int64_t ins = 0; ins < ins_count; ++ins) {
    const int64_t i1 = ins_i1_ptr[ins];
    const int64_t i2 = ins_i2_ptr[ins];
    const int64_t io = ins_out_ptr[ins];
    if (in1_dim_ptr[i1] != 1 || in2_mul_ptr[i2] != 1) {
      separable = false;
      break;
    }
    transformed_offsets[ins + 1] =
        transformed_offsets[ins] + out_dim_ptr[io];
  }

  AT_DISPATCH_FLOATING_TYPES(grad_contig.scalar_type(), "tpu_backward", [&] {
    const auto* grad_ptr = grad_contig.data_ptr<scalar_t>();
    const auto* x1_ptr = x1_contig.data_ptr<scalar_t>();
    const auto* x2_ptr = x2_contig.data_ptr<scalar_t>();
    const auto* weight_ptr = weight_contig.data_ptr<scalar_t>();
    auto* grad_x1_ptr = grad_x1.data_ptr<scalar_t>();
    auto* grad_x2_ptr = grad_x2.data_ptr<scalar_t>();
    auto* grad_weight_ptr = grad_weight.data_ptr<scalar_t>();
    if (separable) {
      const int64_t transformed_total = transformed_offsets.back();
      int64_t max_m1 = 1;
      int64_t max_do = 1;
      for (int64_t ins = 0; ins < ins_count; ++ins) {
        const int64_t i1 = ins_i1_ptr[ins];
        const int64_t io = ins_out_ptr[ins];
        max_m1 = std::max(max_m1, in1_mul_ptr[i1]);
        max_do = std::max(max_do, out_dim_ptr[io]);
      }
      native_parallel_for(0, batch, 16, [&](int64_t begin, int64_t end) {
        std::vector<scalar_t> transformed(transformed_total);
        std::vector<scalar_t> q(max_m1);
        std::vector<scalar_t> s(max_do);
        for (int64_t b = begin; b < end; ++b) {
          for (int64_t ins = 0; ins < ins_count; ++ins) {
            const int64_t i1 = ins_i1_ptr[ins];
            const int64_t i2 = ins_i2_ptr[ins];
            const int64_t io = ins_out_ptr[ins];
            const int64_t m1 = in1_mul_ptr[i1];
            const int64_t d2 = in2_dim_ptr[i2];
            const int64_t do_ = out_dim_ptr[io];
            const auto* w3j = w3js[ins].data_ptr<scalar_t>();
            const scalar_t coeff =
                static_cast<scalar_t>(ins_path_weight_ptr[ins]);
            const int64_t woff = ins_weight_offset_ptr[ins];
            auto* transformed_ins =
                transformed.data() + transformed_offsets[ins];
            const auto* x2_values =
                x2_ptr + b * x2_contig.size(1) + in2_offset_ptr[i2];
            for (int64_t k = 0; k < do_; ++k) {
              scalar_t acc = 0;
              for (int64_t j = 0; j < d2; ++j) {
                acc += x2_values[j] * w3j[j * do_ + k];
              }
              transformed_ins[k] = acc;
            }
            const auto* x1_values =
                x1_ptr + b * x1_contig.size(1) + in1_offset_ptr[i1];
            const auto* grad_values =
                grad_ptr + b * grad_contig.size(1) + out_offset_ptr[io];
            auto* grad_x2_values =
                grad_x2_ptr + b * x2_contig.size(1) + in2_offset_ptr[i2];
            auto* grad_weight_values =
                grad_weight_ptr +
                (shared_weights ? 0 : b * weight_numel) + woff;
            for (int64_t u = 0; u < m1; ++u) {
              const scalar_t w_value = load_weight(
                  weight_ptr, b, weight_numel, shared_weights, woff + u);
              const auto* grad_u = grad_values + u * do_;
              scalar_t q_value = 0;
              for (int64_t k = 0; k < do_; ++k) {
                q_value += grad_u[k] * transformed_ins[k];
              }
              q[u] = q_value;
              const scalar_t x1_value = x1_values[u];
              grad_x1_ptr[b * x1_contig.size(1) +
                          in1_offset_ptr[i1] + u] +=
                  coeff * w_value * q_value;
              if (disjoint_weight) {
                grad_weight_values[u] = coeff * x1_value * q_value;
              } else {
                grad_weight_values[u] += coeff * x1_value * q_value;
              }
            }
            for (int64_t k = 0; k < do_; ++k) {
              scalar_t acc = 0;
              for (int64_t u = 0; u < m1; ++u) {
                const scalar_t w_value = load_weight(
                    weight_ptr, b, weight_numel, shared_weights, woff + u);
                acc += grad_values[u * do_ + k] * w_value *
                    x1_values[u];
              }
              s[k] = acc;
            }
            for (int64_t j = 0; j < d2; ++j) {
              scalar_t acc = 0;
              for (int64_t k = 0; k < do_; ++k) {
                acc += s[k] * w3j[j * do_ + k];
              }
              if (disjoint_x2) {
                grad_x2_values[j] = coeff * acc;
              } else {
                grad_x2_values[j] += coeff * acc;
              }
            }
          }
        }
      });
    } else {
      native_parallel_for(0, batch, 8, [&](int64_t begin, int64_t end) {
        for (int64_t b = begin; b < end; ++b) {
          for (int64_t ins = 0; ins < ins_count; ++ins) {
            const int64_t i1 = ins_i1_ptr[ins];
            const int64_t i2 = ins_i2_ptr[ins];
            const int64_t io = ins_out_ptr[ins];
            const int64_t m1 = in1_mul_ptr[i1];
            const int64_t m2 = in2_mul_ptr[i2];
            const int64_t d1 = in1_dim_ptr[i1];
            const int64_t d2 = in2_dim_ptr[i2];
            const int64_t do_ = out_dim_ptr[io];
            const auto* w3j = w3js[ins].data_ptr<scalar_t>();
            const scalar_t coeff =
                static_cast<scalar_t>(ins_path_weight_ptr[ins]);
            const int64_t woff = ins_weight_offset_ptr[ins];
            for (int64_t u = 0; u < m1; ++u) {
              const auto* grad_u =
                  grad_ptr + b * grad_contig.size(1) +
                  out_offset_ptr[io] + u * do_;
              for (int64_t v = 0; v < m2; ++v) {
                scalar_t gw = 0;
                const scalar_t w_value = load_weight(
                    weight_ptr, b, weight_numel, shared_weights,
                    woff + u * m2 + v);
                for (int64_t i = 0; i < d1; ++i) {
                  const scalar_t x1_value =
                      x1_ptr[b * x1_contig.size(1) +
                             in1_offset_ptr[i1] + u * d1 + i];
                  for (int64_t j = 0; j < d2; ++j) {
                    const scalar_t x2_value =
                        x2_ptr[b * x2_contig.size(1) +
                               in2_offset_ptr[i2] + v * d2 + j];
                    scalar_t gx1 = 0;
                    scalar_t gx2 = 0;
                    const auto* w3j_row = w3j + (i * d2 + j) * do_;
                    for (int64_t k = 0; k < do_; ++k) {
                      const scalar_t term =
                          grad_u[k] * w_value * coeff * w3j_row[k];
                      gx1 += term * x2_value;
                      gx2 += term * x1_value;
                      gw += grad_u[k] * x1_value * x2_value *
                          coeff * w3j_row[k];
                    }
                    grad_x1_ptr[b * x1_contig.size(1) +
                                in1_offset_ptr[i1] + u * d1 + i] += gx1;
                    grad_x2_ptr[b * x2_contig.size(1) +
                                in2_offset_ptr[i2] + v * d2 + j] += gx2;
                  }
                }
                grad_weight_ptr[
                    (shared_weights ? 0 : b * weight_numel) +
                    woff + u * m2 + v] += gw;
              }
            }
          }
        }
      });
    }
  });
  return {grad_x1, grad_x2, grad_weight};
}


torch::Tensor linear_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor in_offset,
    torch::Tensor in_mul,
    torch::Tensor in_dim,
    torch::Tensor out_offset,
    torch::Tensor out_mul,
    torch::Tensor out_dim,
    torch::Tensor ins_in,
    torch::Tensor ins_out,
    torch::Tensor ins_weight_offset,
    torch::Tensor ins_path_weight,
    bool shared_weights) {
  TORCH_CHECK(x.device().is_cpu() && weight.device().is_cpu(), "tensors must be CPU");
  TORCH_CHECK(x.dim() == 2, "x must be [B, Din]");
  TORCH_CHECK(weight.dim() == 1 || weight.dim() == 2, "weight must be 1D or 2D");
  auto x_contig = x.contiguous();
  auto weight_contig = weight.contiguous();
  const int64_t batch = x_contig.size(0);
  const int64_t weight_numel = weight.numel() / (shared_weights ? 1 : batch);
  const int64_t ins_count = ins_in.numel();
  int64_t out_total = 0;
  for (int64_t i = 0; i < out_dim.numel(); ++i) {
    out_total += out_dim.data_ptr<int64_t>()[i] *
        out_mul.data_ptr<int64_t>()[i];
  }
  auto out = torch::empty({batch, out_total}, x_contig.options());

  const auto* in_offset_ptr = int64_data(in_offset);
  const auto* in_mul_ptr = int64_data(in_mul);
  const auto* in_dim_ptr = int64_data(in_dim);
  const auto* out_offset_ptr = int64_data(out_offset);
  const auto* out_mul_ptr = int64_data(out_mul);
  const auto* out_dim_ptr = int64_data(out_dim);
  const auto* ins_in_ptr = int64_data(ins_in);
  const auto* ins_out_ptr = int64_data(ins_out);
  const auto* ins_weight_offset_ptr = int64_data(ins_weight_offset);
  const auto* ins_path_weight_ptr = double_data(ins_path_weight);

  AT_DISPATCH_FLOATING_TYPES(x_contig.scalar_type(), "linear_forward", [&] {
    const auto* x_ptr = x_contig.data_ptr<scalar_t>();
    const auto* weight_ptr = weight_contig.data_ptr<scalar_t>();
    auto* out_ptr = out.data_ptr<scalar_t>();
    native_parallel_for(0, batch, 32, [&](int64_t begin, int64_t end) {
      std::vector<scalar_t> accum(out_total);
      for (int64_t b = begin; b < end; ++b) {
        std::fill(accum.begin(), accum.end(), static_cast<scalar_t>(0));
        for (int64_t ins = 0; ins < ins_count; ++ins) {
          const int64_t ii = ins_in_ptr[ins];
          const int64_t oo = ins_out_ptr[ins];
          const int64_t mul_in = in_mul_ptr[ii];
          const int64_t mul_out = out_mul_ptr[oo];
          const int64_t dim = in_dim_ptr[ii];
          TORCH_CHECK(dim == out_dim_ptr[oo], "linear path irrep dim mismatch");
          const int64_t woff = ins_weight_offset_ptr[ins];
          const scalar_t path_weight =
              static_cast<scalar_t>(ins_path_weight_ptr[ins]);
          for (int64_t u = 0; u < mul_in; ++u) {
            const auto* src =
                x_ptr + b * x_contig.size(1) + in_offset_ptr[ii] + u * dim;
            for (int64_t wo = 0; wo < mul_out; ++wo) {
              const scalar_t w_value = path_weight * load_weight(
                  weight_ptr, b, weight_numel, shared_weights,
                  woff + u * mul_out + wo);
              auto* dst =
                  accum.data() + out_offset_ptr[oo] + wo * dim;
              for (int64_t k = 0; k < dim; ++k) {
                dst[k] += src[k] * w_value;
              }
            }
          }
        }
        for (int64_t col = 0; col < out_total; ++col) {
          out_ptr[b * out_total + col] = accum[col];
        }
      }
    });
  });
  return out;
}


torch::Tensor linear_backward_x(
    torch::Tensor grad_out,
    torch::Tensor weight,
    torch::Tensor in_offset,
    torch::Tensor in_mul,
    torch::Tensor in_dim,
    torch::Tensor out_offset,
    torch::Tensor out_mul,
    torch::Tensor out_dim,
    torch::Tensor ins_in,
    torch::Tensor ins_out,
    torch::Tensor ins_weight_offset,
    torch::Tensor ins_path_weight,
    bool shared_weights) {
  TORCH_CHECK(grad_out.device().is_cpu() && weight.device().is_cpu(), "tensors must be CPU");
  auto grad_contig = grad_out.contiguous();
  auto weight_contig = weight.contiguous();
  const int64_t batch = grad_contig.size(0);
  const int64_t weight_numel = weight.numel() / (shared_weights ? 1 : batch);
  int64_t in_total = 0;
  for (int64_t i = 0; i < in_dim.numel(); ++i) {
    in_total += in_dim.data_ptr<int64_t>()[i] *
        in_mul.data_ptr<int64_t>()[i];
  }
  auto grad_x = parallel_zeros(
      {batch, in_total}, grad_contig.options());

  const auto* in_offset_ptr = int64_data(in_offset);
  const auto* in_mul_ptr = int64_data(in_mul);
  const auto* in_dim_ptr = int64_data(in_dim);
  const auto* out_offset_ptr = int64_data(out_offset);
  const auto* out_mul_ptr = int64_data(out_mul);
  const auto* out_dim_ptr = int64_data(out_dim);
  const auto* ins_in_ptr = int64_data(ins_in);
  const auto* ins_out_ptr = int64_data(ins_out);
  const auto* ins_weight_offset_ptr = int64_data(ins_weight_offset);
  const auto* ins_path_weight_ptr = double_data(ins_path_weight);
  const int64_t ins_count = ins_in.numel();

  AT_DISPATCH_FLOATING_TYPES(grad_contig.scalar_type(), "linear_backward_x", [&] {
    const auto* grad_ptr = grad_contig.data_ptr<scalar_t>();
    const auto* weight_ptr = weight_contig.data_ptr<scalar_t>();
    auto* grad_x_ptr = grad_x.data_ptr<scalar_t>();
    native_parallel_for(0, batch, 32, [&](int64_t begin, int64_t end) {
      for (int64_t b = begin; b < end; ++b) {
        for (int64_t ins = 0; ins < ins_count; ++ins) {
          const int64_t ii = ins_in_ptr[ins];
          const int64_t oo = ins_out_ptr[ins];
          const int64_t mul_in = in_mul_ptr[ii];
          const int64_t mul_out = out_mul_ptr[oo];
          const int64_t dim = in_dim_ptr[ii];
          const int64_t woff = ins_weight_offset_ptr[ins];
          const scalar_t path_weight =
              static_cast<scalar_t>(ins_path_weight_ptr[ins]);
          for (int64_t u = 0; u < mul_in; ++u) {
            auto* dst =
                grad_x_ptr + b * in_total + in_offset_ptr[ii] + u * dim;
            for (int64_t wo = 0; wo < mul_out; ++wo) {
              const scalar_t w_value = path_weight * load_weight(
                  weight_ptr, b, weight_numel, shared_weights,
                  woff + u * mul_out + wo);
              const auto* src =
                  grad_ptr + b * grad_contig.size(1) +
                  out_offset_ptr[oo] + wo * dim;
              for (int64_t k = 0; k < dim; ++k) {
                dst[k] += src[k] * w_value;
              }
            }
          }
        }
      }
    });
  });
  return grad_x;
}


torch::Tensor fctp_onehot_packed_forward(
    torch::Tensor x,
    torch::Tensor packed_weight,
    torch::Tensor vstar,
    torch::Tensor ins_in_offset,
    torch::Tensor ins_out_offset,
    torch::Tensor ins_mul_in,
    torch::Tensor ins_mul_out,
    torch::Tensor ins_dim,
    torch::Tensor ins_weight_offset,
    torch::Tensor ins_coeff_offset,
    torch::Tensor coeff,
    int64_t out_total) {
  TORCH_CHECK(x.device().is_cpu() && packed_weight.device().is_cpu(), "tensors must be CPU");
  TORCH_CHECK(x.dim() == 2, "x must be [B, Din]");
  TORCH_CHECK(packed_weight.dim() == 2, "packed_weight must be [E, packed_numel]");
  TORCH_CHECK(vstar.scalar_type() == torch::kInt64 && vstar.numel() == x.size(0), "bad vstar");
  auto x_contig = x.contiguous();
  auto weight_contig = packed_weight.contiguous();
  auto vstar_contig = vstar.contiguous();
  auto coeff_contig = coeff.contiguous();
  auto out = parallel_zeros(
      {x_contig.size(0), out_total}, x_contig.options());

  const int64_t ins_count = ins_in_offset.numel();
  const auto* in_offset = int64_data(ins_in_offset);
  const auto* out_offset = int64_data(ins_out_offset);
  const auto* mul_in = int64_data(ins_mul_in);
  const auto* mul_out = int64_data(ins_mul_out);
  const auto* dims = int64_data(ins_dim);
  const auto* weight_offset = int64_data(ins_weight_offset);
  const auto* coeff_offset = int64_data(ins_coeff_offset);
  const auto* vstar_ptr = vstar_contig.data_ptr<int64_t>();
  const int64_t packed_stride = weight_contig.size(1);

  AT_DISPATCH_FLOATING_TYPES(x_contig.scalar_type(), "fctp_onehot_packed_forward", [&] {
    const auto* x_ptr = x_contig.data_ptr<scalar_t>();
    const auto* weight_ptr = weight_contig.data_ptr<scalar_t>();
  const auto* coeff_ptr = coeff_contig.data_ptr<scalar_t>();
  auto* out_ptr = out.data_ptr<scalar_t>();
  const int64_t x_stride = x_contig.size(1);
  int64_t max_temp = 0;
  for (int64_t ins = 0; ins < ins_count; ++ins) {
    max_temp = std::max(max_temp, mul_out[ins] * dims[ins]);
  }

  native_parallel_for(0, x_contig.size(0), 16, [&](int64_t begin, int64_t end) {
    std::vector<scalar_t> temp(max_temp);
    for (int64_t b = begin; b < end; ++b) {
      const int64_t element = vstar_ptr[b];
        for (int64_t ins = 0; ins < ins_count; ++ins) {
          const int64_t mi = mul_in[ins];
          const int64_t mo = mul_out[ins];
          const int64_t dim = dims[ins];
          const auto* xs = x_ptr + b * x_stride + in_offset[ins];
          const auto* ws =
              weight_ptr + element * packed_stride + weight_offset[ins];
          auto* ys = out_ptr + b * out_total + out_offset[ins];
          const auto* cs = coeff_ptr + coeff_offset[ins];
          cpu_gemm<scalar_t>(
              false, false, mo, dim, mi, ws, mi, xs, dim,
              temp.data(), dim);
          for (int64_t wo = 0; wo < mo; ++wo) {
            auto* y_row = ys + wo * dim;
            const auto* t_row = temp.data() + wo * dim;
            for (int64_t k = 0; k < dim; ++k) {
              y_row[k] += t_row[k] * cs[k];
            }
          }
        }
      }
    });
  });
  return out;
}


torch::Tensor fctp_onehot_packed_backward_x(
    torch::Tensor grad_out,
    torch::Tensor packed_weight,
    torch::Tensor vstar,
    torch::Tensor ins_in_offset,
    torch::Tensor ins_out_offset,
    torch::Tensor ins_mul_in,
    torch::Tensor ins_mul_out,
    torch::Tensor ins_dim,
    torch::Tensor ins_weight_offset,
    torch::Tensor ins_coeff_offset,
    torch::Tensor coeff,
    int64_t in_total) {
  TORCH_CHECK(grad_out.device().is_cpu() && packed_weight.device().is_cpu(), "tensors must be CPU");
  auto grad_contig = grad_out.contiguous();
  auto weight_contig = packed_weight.contiguous();
  auto vstar_contig = vstar.contiguous();
  auto coeff_contig = coeff.contiguous();
  auto grad_x = parallel_zeros(
      {grad_contig.size(0), in_total}, grad_contig.options());

  const int64_t ins_count = ins_in_offset.numel();
  const auto* in_offset = int64_data(ins_in_offset);
  const auto* out_offset = int64_data(ins_out_offset);
  const auto* mul_in = int64_data(ins_mul_in);
  const auto* mul_out = int64_data(ins_mul_out);
  const auto* dims = int64_data(ins_dim);
  const auto* weight_offset = int64_data(ins_weight_offset);
  const auto* coeff_offset = int64_data(ins_coeff_offset);
  const auto* vstar_ptr = vstar_contig.data_ptr<int64_t>();
  const int64_t packed_stride = weight_contig.size(1);

  AT_DISPATCH_FLOATING_TYPES(grad_contig.scalar_type(), "fctp_onehot_packed_backward_x", [&] {
    const auto* grad_ptr = grad_contig.data_ptr<scalar_t>();
    const auto* weight_ptr = weight_contig.data_ptr<scalar_t>();
  const auto* coeff_ptr = coeff_contig.data_ptr<scalar_t>();
  auto* grad_x_ptr = grad_x.data_ptr<scalar_t>();
  const int64_t grad_stride = grad_contig.size(1);
  int64_t max_temp = 0;
  for (int64_t ins = 0; ins < ins_count; ++ins) {
    max_temp = std::max(max_temp, mul_in[ins] * dims[ins]);
  }

  native_parallel_for(0, grad_contig.size(0), 16, [&](int64_t begin, int64_t end) {
    std::vector<scalar_t> temp(max_temp);
    for (int64_t b = begin; b < end; ++b) {
        const int64_t element = vstar_ptr[b];
        for (int64_t ins = 0; ins < ins_count; ++ins) {
          const int64_t mi = mul_in[ins];
          const int64_t mo = mul_out[ins];
          const int64_t dim = dims[ins];
          const auto* gs = grad_ptr + b * grad_stride + out_offset[ins];
          const auto* ws =
              weight_ptr + element * packed_stride + weight_offset[ins];
          auto* gxs = grad_x_ptr + b * in_total + in_offset[ins];
          const auto* cs = coeff_ptr + coeff_offset[ins];
          cpu_gemm<scalar_t>(
              true, false, mi, dim, mo, ws, mi, gs, dim,
              temp.data(), dim);
          for (int64_t u = 0; u < mi; ++u) {
            auto* gx_row = gxs + u * dim;
            const auto* t_row = temp.data() + u * dim;
            for (int64_t k = 0; k < dim; ++k) {
              gx_row[k] += t_row[k] * cs[k];
            }
          }
        }
      }
    });
  });
  return grad_x;
}


torch::Tensor linear_packed_forward(
    torch::Tensor x,
    torch::Tensor packed_weight,
    torch::Tensor ins_in_offset,
    torch::Tensor ins_out_offset,
    torch::Tensor ins_mul_in,
    torch::Tensor ins_mul_out,
    torch::Tensor ins_dim,
    torch::Tensor ins_weight_offset,
    int64_t out_total) {
  TORCH_CHECK(x.device().is_cpu() && packed_weight.device().is_cpu(), "tensors must be CPU");
  auto x_contig = x.contiguous();
  auto weight_contig = packed_weight.contiguous();
  const int64_t batch = x_contig.size(0);
  const int64_t ins_count = ins_in_offset.numel();
  const auto* in_offset = int64_data(ins_in_offset);
  const auto* out_offset = int64_data(ins_out_offset);
  const auto* mul_in = int64_data(ins_mul_in);
  const auto* mul_out = int64_data(ins_mul_out);
  const auto* dims = int64_data(ins_dim);
  const auto* weight_offset = int64_data(ins_weight_offset);
  bool disjoint_output = true;
  for (int64_t i = 0; i < ins_count && disjoint_output; ++i) {
    const int64_t a0 = out_offset[i];
    const int64_t a1 = a0 + mul_out[i] * dims[i];
    for (int64_t j = i + 1; j < ins_count; ++j) {
      const int64_t b0 = out_offset[j];
      const int64_t b1 = b0 + mul_out[j] * dims[j];
      if (a0 < b1 && b0 < a1) {
        disjoint_output = false;
        break;
      }
    }
  }
  auto out = disjoint_output
      ? torch::empty({batch, out_total}, x_contig.options())
      : parallel_zeros({batch, out_total}, x_contig.options());

  AT_DISPATCH_FLOATING_TYPES(x_contig.scalar_type(), "linear_packed_forward", [&] {
    const auto* x_ptr = x_contig.data_ptr<scalar_t>();
    const auto* weight_ptr = weight_contig.data_ptr<scalar_t>();
    auto* out_ptr = out.data_ptr<scalar_t>();
    const int64_t x_stride = x_contig.size(1);
    if (ins_count == 1 && dims[0] == 1 &&
        in_offset[0] == 0 && out_offset[0] == 0 &&
        mul_in[0] == x_stride && mul_out[0] == out_total) {
      cpu_gemm<scalar_t>(
          false, true, batch, mul_out[0], mul_in[0],
          x_ptr, x_stride,
          weight_ptr, mul_in[0],
          out_ptr, out_total);
      return;
    }
    parallel_zero_(out);
    std::vector<int64_t> pack_offsets(ins_count + 1, 0);
    std::vector<int64_t> temp_offsets(ins_count + 1, 0);
    for (int64_t ins = 0; ins < ins_count; ++ins) {
      pack_offsets[ins + 1] =
          pack_offsets[ins] + mul_in[ins] * batch * dims[ins];
      temp_offsets[ins + 1] =
          temp_offsets[ins] + mul_out[ins] * batch * dims[ins];
    }
    std::vector<scalar_t> x_pack(pack_offsets.back());
    std::vector<scalar_t> temp(temp_offsets.back());
    native_parallel_for(0, ins_count * batch, 1, [&](int64_t begin, int64_t end) {
      for (int64_t task = begin; task < end; ++task) {
        const int64_t ins = task / batch;
        const int64_t b = task - ins * batch;
        const int64_t mi = mul_in[ins];
        const int64_t dim = dims[ins];
        const int64_t packed_cols = batch * dim;
        for (int64_t u = 0; u < mi; ++u) {
          const auto* src =
              x_ptr + b * x_stride + in_offset[ins] + u * dim;
          auto* dst =
              x_pack.data() + pack_offsets[ins] + u * packed_cols + b * dim;
          std::memcpy(dst, src, dim * sizeof(scalar_t));
        }
      }
    });
    for (int64_t ins = 0; ins < ins_count; ++ins) {
      const int64_t mi = mul_in[ins];
      const int64_t mo = mul_out[ins];
      const int64_t dim = dims[ins];
      const int64_t packed_cols = batch * dim;
      const auto* ws = weight_ptr + weight_offset[ins];
      const auto* packed = x_pack.data() + pack_offsets[ins];
      auto* packed_out = temp.data() + temp_offsets[ins];
      const char* small_gemm_env =
          std::getenv("NATIVE_CPU_MACE_SMALL_GEMM");
      const bool use_small_gemm =
          small_gemm_env != nullptr && std::string(small_gemm_env) == "1";
      if (use_small_gemm && mo <= 256 && packed_cols <= 256 && mi <= 256) {
        cpu_gemm_packed_nn<scalar_t>(
            mo,
            packed_cols,
            mi,
            ws,
            mi,
            packed,
            packed_cols,
            packed_out,
            packed_cols);
      } else {
        cpu_gemm<scalar_t>(
            false, false, mo, packed_cols, mi, ws, mi, packed,
            packed_cols, packed_out, packed_cols);
      }
    }
    native_parallel_for(0, ins_count * batch, 1, [&](int64_t begin, int64_t end) {
      for (int64_t task = begin; task < end; ++task) {
        const int64_t ins = task / batch;
        const int64_t b = task - ins * batch;
        const int64_t mo = mul_out[ins];
        const int64_t dim = dims[ins];
        const int64_t packed_cols = batch * dim;
        const auto* packed_out = temp.data() + temp_offsets[ins];
        for (int64_t wo = 0; wo < mo; ++wo) {
          auto* dst =
              out_ptr + b * out_total + out_offset[ins] + wo * dim;
          const auto* src = packed_out + wo * packed_cols + b * dim;
          if (disjoint_output) {
            std::memcpy(dst, src, dim * sizeof(scalar_t));
            continue;
          }
          for (int64_t k = 0; k < dim; ++k) {
            dst[k] += src[k];
          }
        }
      }
    });
  });
  return out;
}


torch::Tensor linear_packed_backward_x(
    torch::Tensor grad_out,
    torch::Tensor packed_weight,
    torch::Tensor ins_in_offset,
    torch::Tensor ins_out_offset,
    torch::Tensor ins_mul_in,
    torch::Tensor ins_mul_out,
    torch::Tensor ins_dim,
    torch::Tensor ins_weight_offset,
    int64_t in_total) {
  TORCH_CHECK(grad_out.device().is_cpu() && packed_weight.device().is_cpu(), "tensors must be CPU");
  auto grad_contig = grad_out.contiguous();
  auto weight_contig = packed_weight.contiguous();
  const int64_t batch = grad_contig.size(0);
  const int64_t ins_count = ins_in_offset.numel();
  const auto* in_offset = int64_data(ins_in_offset);
  const auto* out_offset = int64_data(ins_out_offset);
  const auto* mul_in = int64_data(ins_mul_in);
  const auto* mul_out = int64_data(ins_mul_out);
  const auto* dims = int64_data(ins_dim);
  const auto* weight_offset = int64_data(ins_weight_offset);
  bool disjoint_input = true;
  for (int64_t i = 0; i < ins_count && disjoint_input; ++i) {
    const int64_t a0 = in_offset[i];
    const int64_t a1 = a0 + mul_in[i] * dims[i];
    for (int64_t j = i + 1; j < ins_count; ++j) {
      const int64_t b0 = in_offset[j];
      const int64_t b1 = b0 + mul_in[j] * dims[j];
      if (a0 < b1 && b0 < a1) {
        disjoint_input = false;
        break;
      }
    }
  }
  auto grad_x = disjoint_input
      ? torch::empty({batch, in_total}, grad_contig.options())
      : parallel_zeros({batch, in_total}, grad_contig.options());

  AT_DISPATCH_FLOATING_TYPES(grad_contig.scalar_type(), "linear_packed_backward_x", [&] {
    const auto* grad_ptr = grad_contig.data_ptr<scalar_t>();
    const auto* weight_ptr = weight_contig.data_ptr<scalar_t>();
    auto* grad_x_ptr = grad_x.data_ptr<scalar_t>();
    const int64_t grad_stride = grad_contig.size(1);
    if (ins_count == 1 && dims[0] == 1 &&
        in_offset[0] == 0 && out_offset[0] == 0 &&
        mul_in[0] == grad_x.size(1) && mul_out[0] == grad_stride) {
      cpu_gemm<scalar_t>(
          false, false, batch, mul_in[0], mul_out[0],
          grad_ptr, grad_stride,
          weight_ptr, mul_in[0],
          grad_x_ptr, grad_x.size(1));
      return;
    }
    parallel_zero_(grad_x);
    std::vector<int64_t> pack_offsets(ins_count + 1, 0);
    std::vector<int64_t> temp_offsets(ins_count + 1, 0);
    for (int64_t ins = 0; ins < ins_count; ++ins) {
      pack_offsets[ins + 1] =
          pack_offsets[ins] + mul_out[ins] * batch * dims[ins];
      temp_offsets[ins + 1] =
          temp_offsets[ins] + mul_in[ins] * batch * dims[ins];
    }
    std::vector<scalar_t> grad_pack(pack_offsets.back());
    std::vector<scalar_t> temp(temp_offsets.back());
    native_parallel_for(0, ins_count * batch, 1, [&](int64_t begin, int64_t end) {
      for (int64_t task = begin; task < end; ++task) {
        const int64_t ins = task / batch;
        const int64_t b = task - ins * batch;
        const int64_t mo = mul_out[ins];
        const int64_t dim = dims[ins];
        const int64_t packed_cols = batch * dim;
        for (int64_t wo = 0; wo < mo; ++wo) {
          const auto* src =
              grad_ptr + b * grad_stride + out_offset[ins] + wo * dim;
          auto* dst =
              grad_pack.data() + pack_offsets[ins] + wo * packed_cols + b * dim;
          std::memcpy(dst, src, dim * sizeof(scalar_t));
        }
      }
    });
    for (int64_t ins = 0; ins < ins_count; ++ins) {
      const int64_t mi = mul_in[ins];
      const int64_t mo = mul_out[ins];
      const int64_t dim = dims[ins];
      const int64_t packed_cols = batch * dim;
      const auto* ws = weight_ptr + weight_offset[ins];
      const auto* packed = grad_pack.data() + pack_offsets[ins];
      auto* packed_out = temp.data() + temp_offsets[ins];
      const char* small_gemm_env =
          std::getenv("NATIVE_CPU_MACE_SMALL_GEMM");
      const bool use_small_gemm =
          small_gemm_env != nullptr && std::string(small_gemm_env) == "1";
      if (use_small_gemm && mi <= 256 && packed_cols <= 256 && mo <= 256) {
        cpu_gemm_packed_tn<scalar_t>(
            mi,
            packed_cols,
            mo,
            ws,
            mi,
            packed,
            packed_cols,
            packed_out,
            packed_cols);
      } else {
        cpu_gemm<scalar_t>(
            true, false, mi, packed_cols, mo, ws, mi, packed,
            packed_cols, packed_out, packed_cols);
      }
    }
    native_parallel_for(0, ins_count * batch, 1, [&](int64_t begin, int64_t end) {
      for (int64_t task = begin; task < end; ++task) {
        const int64_t ins = task / batch;
        const int64_t b = task - ins * batch;
        const int64_t mi = mul_in[ins];
        const int64_t dim = dims[ins];
        const int64_t packed_cols = batch * dim;
        const auto* packed_out = temp.data() + temp_offsets[ins];
        for (int64_t u = 0; u < mi; ++u) {
          auto* dst =
              grad_x_ptr + b * in_total + in_offset[ins] + u * dim;
          const auto* src = packed_out + u * packed_cols + b * dim;
          if (disjoint_input) {
            std::memcpy(dst, src, dim * sizeof(scalar_t));
            continue;
          }
          for (int64_t k = 0; k < dim; ++k) {
            dst[k] += src[k];
          }
        }
      }
    });
  });
  return grad_x;
}


std::vector<torch::Tensor> radial_mlp_forward(
    torch::Tensor x,
    std::vector<torch::Tensor> weights,
    torch::Tensor scales,
    torch::Tensor active) {
  TORCH_CHECK(x.device().is_cpu(), "radial MLP x must be CPU");
  TORCH_CHECK(x.dim() == 2, "radial MLP x must be [N, in_features]");
  TORCH_CHECK(scales.device().is_cpu() && scales.scalar_type() == torch::kFloat64,
              "radial MLP scales must be a CPU float64 tensor");
  TORCH_CHECK(active.device().is_cpu() && active.scalar_type() == torch::kInt64,
              "radial MLP active flags must be a CPU int64 tensor");
  TORCH_CHECK(
      scales.numel() == static_cast<int64_t>(weights.size()),
      "radial MLP scale count mismatch");
  TORCH_CHECK(
      active.numel() == static_cast<int64_t>(weights.size()),
      "radial MLP active count mismatch");

  auto current = x.contiguous();
  const int64_t rows = current.size(0);
  std::vector<torch::Tensor> pre_activations;
  pre_activations.reserve(weights.size());

  AT_DISPATCH_FLOATING_TYPES(current.scalar_type(), "radial_mlp_forward", [&] {
    const auto* scales_ptr = scales.data_ptr<double>();
    const auto* active_ptr = active.data_ptr<int64_t>();
    const char* manual_env = std::getenv("NATIVE_CPU_MACE_RADIAL_MANUAL");
    const bool use_manual =
        manual_env == nullptr || std::string(manual_env) != "0";
    for (size_t layer = 0; layer < weights.size(); ++layer) {
      auto weight = weights[layer].contiguous();
      TORCH_CHECK(weight.device().is_cpu(), "radial MLP weight must be CPU");
      TORCH_CHECK(weight.dim() == 2, "radial MLP weight must be 2D");
      TORCH_CHECK(weight.scalar_type() == current.scalar_type(),
                  "radial MLP weight dtype mismatch");
      TORCH_CHECK(weight.size(0) == current.size(1),
                  "radial MLP input dimension mismatch");

      const int64_t in_dim = current.size(1);
      const int64_t out_dim = weight.size(1);
      auto value = torch::empty({rows, out_dim}, current.options());
      if (use_manual) {
        radial_mlp_linear_forward<scalar_t>(
            current.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            value.data_ptr<scalar_t>(),
            rows,
            in_dim,
            out_dim);
      } else {
        cpu_gemm<scalar_t>(
            false, false, rows, out_dim, in_dim,
            current.data_ptr<scalar_t>(), in_dim,
            weight.data_ptr<scalar_t>(), out_dim,
            value.data_ptr<scalar_t>(), out_dim);
      }

      if (active_ptr[layer] == 0) {
        current = value;
        continue;
      }
      const double activation_scale = scales_ptr[layer];
      pre_activations.push_back(value);
      auto activated = torch::empty_like(value);
      auto* activated_ptr = activated.data_ptr<scalar_t>();
      const auto* value_ptr = value.data_ptr<scalar_t>();
      const int64_t count = value.numel();
      native_parallel_for(0, count, 256, [&](int64_t begin, int64_t end) {
        for (int64_t index = begin; index < end; ++index) {
          const scalar_t v = value_ptr[index];
          const scalar_t sigmoid =
              static_cast<scalar_t>(1.0) /
              (static_cast<scalar_t>(1.0) +
               std::exp(-static_cast<double>(v)));
          activated_ptr[index] = v * sigmoid *
              static_cast<scalar_t>(activation_scale);
        }
      });
      current = activated;
    }
  });

  std::vector<torch::Tensor> outputs;
  outputs.reserve(pre_activations.size() + 1);
  outputs.push_back(current);
  outputs.insert(outputs.end(), pre_activations.begin(), pre_activations.end());
  return outputs;
}


torch::Tensor radial_mlp_backward(
    torch::Tensor grad_out,
    std::vector<torch::Tensor> weights,
    torch::Tensor scales,
    torch::Tensor active,
    std::vector<torch::Tensor> pre_activations) {
  TORCH_CHECK(grad_out.device().is_cpu(), "radial MLP grad must be CPU");
  TORCH_CHECK(scales.device().is_cpu() && scales.scalar_type() == torch::kFloat64,
              "radial MLP scales must be a CPU float64 tensor");
  TORCH_CHECK(active.device().is_cpu() && active.scalar_type() == torch::kInt64,
              "radial MLP active flags must be a CPU int64 tensor");
  TORCH_CHECK(
      scales.numel() == static_cast<int64_t>(weights.size()),
      "radial MLP scale count mismatch");
  TORCH_CHECK(
      active.numel() == static_cast<int64_t>(weights.size()),
      "radial MLP active count mismatch");

  auto grad = grad_out.contiguous();
  const int64_t rows = grad.size(0);
  int64_t pre_index = static_cast<int64_t>(pre_activations.size());

  AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "radial_mlp_backward", [&] {
    const auto* scales_ptr = scales.data_ptr<double>();
    const auto* active_ptr = active.data_ptr<int64_t>();
    const char* manual_env = std::getenv("NATIVE_CPU_MACE_RADIAL_MANUAL");
    const bool use_manual =
        manual_env == nullptr || std::string(manual_env) != "0";
    for (int64_t layer = static_cast<int64_t>(weights.size()) - 1;
         layer >= 0;
         --layer) {
      auto weight = weights[layer].contiguous();
      const int64_t out_dim = weight.size(1);
      const int64_t in_dim = weight.size(0);
      TORCH_CHECK(grad.size(1) == out_dim, "radial MLP grad dimension mismatch");

      if (active_ptr[layer] != 0) {
        TORCH_CHECK(pre_index > 0, "radial MLP saved activation mismatch");
        const double activation_scale = scales_ptr[layer];
        auto value = pre_activations[--pre_index].contiguous();
        auto scaled_grad = torch::empty_like(grad);
        const auto* value_ptr = value.data_ptr<scalar_t>();
        const auto* grad_ptr = grad.data_ptr<scalar_t>();
        auto* scaled_ptr = scaled_grad.data_ptr<scalar_t>();
        const int64_t count = grad.numel();
        native_parallel_for(0, count, 256, [&](int64_t begin, int64_t end) {
          for (int64_t index = begin; index < end; ++index) {
            const scalar_t v = value_ptr[index];
            const scalar_t sigmoid =
                static_cast<scalar_t>(1.0) /
                (static_cast<scalar_t>(1.0) +
                 std::exp(-static_cast<double>(v)));
            const scalar_t derivative = sigmoid *
                (static_cast<scalar_t>(1.0) +
                 v * (static_cast<scalar_t>(1.0) - sigmoid));
            scaled_ptr[index] = grad_ptr[index] * derivative *
                static_cast<scalar_t>(activation_scale);
          }
        });
        grad = scaled_grad;
      }

      auto grad_x = torch::empty({rows, in_dim}, grad.options());
      if (use_manual) {
        radial_mlp_linear_backward<scalar_t>(
            grad.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            grad_x.data_ptr<scalar_t>(),
            rows,
            in_dim,
            out_dim);
      } else {
        cpu_gemm<scalar_t>(
            false, true, rows, in_dim, out_dim,
            grad.data_ptr<scalar_t>(), out_dim,
            weight.data_ptr<scalar_t>(), out_dim,
            grad_x.data_ptr<scalar_t>(), in_dim);
      }
      grad = grad_x;
    }
  });

  return grad;
}


torch::Tensor symmetric_contraction3_forward(
    torch::Tensor U,
    torch::Tensor weight,
    torch::Tensor x,
    torch::Tensor element) {
  TORCH_CHECK(U.device().is_cpu() && weight.device().is_cpu(), "tensors must be CPU");
  TORCH_CHECK(x.device().is_cpu() && element.device().is_cpu(), "tensors must be CPU");
  TORCH_CHECK(U.dim() >= 3, "U must have shape [*S,I,K]");
  TORCH_CHECK(weight.dim() == 3, "weight must have shape [E,K,C]");
  TORCH_CHECK(x.dim() == 3, "x must have shape [B,C,I]");
  TORCH_CHECK(element.scalar_type() == torch::kInt64, "element must be int64");

  auto U_contig = U.contiguous();
  auto weight_contig = weight.contiguous();
  auto x_contig = x.contiguous();
  auto element_contig = element.contiguous();
  const int64_t irreps = U_contig.size(-2);
  const int64_t params = U_contig.size(-1);
  const int64_t elements = weight_contig.size(0);
  const int64_t channels = weight_contig.size(2);
  const int64_t batch = x_contig.size(0);
  int64_t spatial = 1;
  std::vector<int64_t> spatial_dims;
  for (int64_t i = 0; i < U_contig.dim() - 2; ++i) {
    spatial_dims.push_back(U_contig.size(i));
    spatial *= U_contig.size(i);
  }
  const int64_t m_dim = spatial * irreps;
  TORCH_CHECK(weight_contig.size(1) == params, "weight K mismatch");
  TORCH_CHECK(x_contig.size(1) == channels && x_contig.size(2) == irreps, "x shape mismatch");

  std::vector<int64_t> out_sizes = {batch, channels};
  out_sizes.insert(out_sizes.end(), spatial_dims.begin(), spatial_dims.end());
  auto out = torch::empty(out_sizes, x_contig.options());
  // [E, C, M] makes the irrep contraction in stage B contiguous.
  auto transformed = torch::empty({elements, channels, m_dim}, x_contig.options());

  AT_DISPATCH_FLOATING_TYPES(x_contig.scalar_type(), "symmetric_contraction3_forward", [&] {
    const auto* u_ptr = U_contig.data_ptr<scalar_t>();
    const auto* weight_ptr = weight_contig.data_ptr<scalar_t>();
    const auto* x_ptr = x_contig.data_ptr<scalar_t>();
    const auto* element_ptr = element_contig.data_ptr<int64_t>();
    auto* transformed_ptr = transformed.data_ptr<scalar_t>();
    auto* out_ptr = out.data_ptr<scalar_t>();

    native_parallel_for(0, elements, 1, [&](int64_t begin, int64_t end) {
      for (int64_t e = begin; e < end; ++e) {
        cpu_gemm<scalar_t>(
            true, true, channels, m_dim, params,
            weight_ptr + e * params * channels, channels,
            u_ptr, params,
            transformed_ptr + e * channels * m_dim, m_dim);
      }
    });

    native_parallel_for(0, batch, 8, [&](int64_t begin, int64_t end) {
      for (int64_t b = begin; b < end; ++b) {
        const int64_t e = element_ptr[b];
        const auto* t_ptr = transformed_ptr + e * channels * m_dim;
        const auto* xb = x_ptr + b * channels * irreps;
        auto* out_b = out_ptr + b * channels * spatial;
        for (int64_t c = 0; c < channels; ++c) {
          const auto* tc = t_ptr + c * m_dim;
          const auto* xc = xb + c * irreps;
          auto* out_c = out_b + c * spatial;
          for (int64_t s = 0; s < spatial; ++s) {
            scalar_t acc = 0;
            for (int64_t i = 0; i < irreps; ++i) {
              acc += tc[s * irreps + i] * xc[i];
            }
            out_c[s] = acc;
          }
        }
      }
    });
  });
  return out;
}


torch::Tensor symmetric_contraction3_backward_x(
    torch::Tensor grad_out,
    torch::Tensor U,
    torch::Tensor weight,
    torch::Tensor element) {
  TORCH_CHECK(grad_out.device().is_cpu(), "grad_out must be CPU");
  TORCH_CHECK(U.device().is_cpu() && weight.device().is_cpu(), "tensors must be CPU");
  TORCH_CHECK(element.device().is_cpu(), "element must be CPU");
  auto grad_contig = grad_out.contiguous();
  auto U_contig = U.contiguous();
  auto weight_contig = weight.contiguous();
  auto element_contig = element.contiguous();
  const int64_t batch = grad_contig.size(0);
  const int64_t channels = grad_contig.size(1);
  const int64_t irreps = U_contig.size(-2);
  const int64_t params = U_contig.size(-1);
  const int64_t elements = weight_contig.size(0);
  int64_t spatial = 1;
  for (int64_t i = 0; i < U_contig.dim() - 2; ++i) {
    spatial *= U_contig.size(i);
  }
  const int64_t m_dim = spatial * irreps;
  auto grad_x = torch::empty(
      {batch, channels, irreps}, grad_contig.options());
  auto transformed = torch::empty({elements, channels, m_dim}, grad_contig.options());

  AT_DISPATCH_FLOATING_TYPES(grad_contig.scalar_type(), "symmetric_contraction3_backward_x", [&] {
    const auto* u_ptr = U_contig.data_ptr<scalar_t>();
    const auto* weight_ptr = weight_contig.data_ptr<scalar_t>();
    const auto* element_ptr = element_contig.data_ptr<int64_t>();
    const auto* grad_ptr = grad_contig.data_ptr<scalar_t>();
    auto* transformed_ptr = transformed.data_ptr<scalar_t>();
    auto* grad_x_ptr = grad_x.data_ptr<scalar_t>();

    native_parallel_for(0, elements, 1, [&](int64_t begin, int64_t end) {
      for (int64_t e = begin; e < end; ++e) {
        cpu_gemm<scalar_t>(
            true, true, channels, m_dim, params,
            weight_ptr + e * params * channels, channels,
            u_ptr, params,
            transformed_ptr + e * channels * m_dim, m_dim);
      }
    });

    native_parallel_for(0, batch, 8, [&](int64_t begin, int64_t end) {
      for (int64_t b = begin; b < end; ++b) {
        const auto* t_ptr = transformed_ptr +
            element_ptr[b] * channels * m_dim;
        const auto* grad_b = grad_ptr + b * channels * spatial;
        auto* grad_x_b = grad_x_ptr + b * channels * irreps;
        for (int64_t c = 0; c < channels; ++c) {
          const auto* tc = t_ptr + c * m_dim;
          const auto* grad_c = grad_b + c * spatial;
          auto* grad_x_c = grad_x_b + c * irreps;
          for (int64_t i = 0; i < irreps; ++i) {
            scalar_t acc = 0;
            for (int64_t s = 0; s < spatial; ++s) {
              acc += tc[s * irreps + i] * grad_c[s];
            }
            grad_x_c[i] = acc;
          }
        }
      }
    });
  });
  return grad_x;
}


// Fused corr=3 scalar SymmetricContraction used by the MACE-OFF models.
//
// This follows the optimized FX schedule used by MACE:
//   main[b,c,w,x] = sum_ik U3[w,x,i,k] * Wmax[e_b,k,c] * x[b,c,i]
//   ct             = main + W0[e_b]^T @ U2^T
//   out1[b,c,x]    = sum_w ct[b,c,w,x] * x[b,c,w]
//   ct1            = out1 + U1 @ W1[e_b]^T
//   out[b,c]       = sum_x ct1[b,c,x] * x[b,c,x]
//
// The two spatial dimensions are equal in the current MACE-OFF models, but
// they are kept as explicit dimensions so the kernel also follows the exact
// contraction order of the generated graph.
//
// Metadata folds Wmax and U3 once into
//   effective[e,c,i,w,x] = sum_k Wmax[e,c,k] * U3[w,x,i,k],
// so the hot path performs a small row-wise contraction instead of materializing
// the multi-megabyte temporary used by the original blocked formulation.
std::vector<torch::Tensor> symmetric_contraction3_scalar_fused_forward(
    torch::Tensor x,
    torch::Tensor element,
    torch::Tensor effective,
    torch::Tensor C0,
    torch::Tensor C1) {
  TORCH_CHECK(x.device().is_cpu() && element.device().is_cpu(),
              "x and element must be CPU tensors");
  TORCH_CHECK(effective.device().is_cpu() &&
              C0.device().is_cpu() && C1.device().is_cpu(),
              "SC tensors must be CPU tensors");
  TORCH_CHECK(x.dim() == 3, "x must have shape [B,C,I]");
  TORCH_CHECK(element.scalar_type() == torch::kInt64,
              "element must be int64");
  TORCH_CHECK(effective.dim() == 4,
              "effective must have shape [E,C,I,W*X]");
  TORCH_CHECK(C0.dim() == 3, "C0 must have shape [E,C,W*X]");
  TORCH_CHECK(C1.dim() == 3, "C1 must have shape [E,C,W]");

  auto x_contig = x.contiguous();
  auto element_contig = element.contiguous();
  auto eff = effective.contiguous();
  auto c0 = C0.contiguous();
  auto c1 = C1.contiguous();

  const int64_t batch = x_contig.size(0);
  const int64_t channels = x_contig.size(1);
  const int64_t irrep = x_contig.size(2);
  const int64_t elements = eff.size(0);
  const int64_t m_dim = eff.size(3);
  const int64_t wdim = c1.size(2);
  TORCH_CHECK(eff.size(1) == channels && eff.size(2) == irrep,
              "effective shape mismatch");
  TORCH_CHECK(wdim * wdim == m_dim,
              "effective last dimension must be W*X with W=X");
  TORCH_CHECK(c0.size(0) == elements && c0.size(1) == channels &&
              c0.size(2) == m_dim, "C0 shape mismatch");
  TORCH_CHECK(c1.size(0) == elements && c1.size(1) == channels &&
              c1.size(2) == wdim, "C1 shape mismatch");

  auto out = torch::empty({batch, channels}, x_contig.options());
  auto saved_ct = torch::empty({batch, channels, m_dim}, x_contig.options());
  auto saved_ct1 = torch::empty({batch, channels, wdim}, x_contig.options());

  AT_DISPATCH_FLOATING_TYPES(
      x_contig.scalar_type(), "symmetric_contraction3_scalar_fused_forward", [&] {
        const auto* x_ptr = x_contig.data_ptr<scalar_t>();
        const auto* element_ptr = element_contig.data_ptr<int64_t>();
        const auto* eff_ptr = eff.data_ptr<scalar_t>();
        const auto* c0_ptr = c0.data_ptr<scalar_t>();
        const auto* c1_ptr = c1.data_ptr<scalar_t>();
        auto* out_ptr = out.data_ptr<scalar_t>();
        auto* ct_ptr = saved_ct.data_ptr<scalar_t>();
        auto* ct1_ptr = saved_ct1.data_ptr<scalar_t>();

        const int64_t rows = batch * channels;
        for (int64_t b = 0; b < batch; ++b) {
          const int64_t e = element_ptr[b];
          TORCH_CHECK(e >= 0 && e < elements, "element index out of range");
        }
        auto process_row = [&](int64_t node, int64_t c) {
          const int64_t row = node * channels + c;
          const int64_t e = element_ptr[node];
          const auto* xr = x_ptr + row * irrep;
          const auto* er = eff_ptr + (e * channels + c) * irrep * m_dim;
          const auto* c0r = c0_ptr + (e * channels + c) * m_dim;
          const auto* c1r = c1_ptr + (e * channels + c) * wdim;
          auto* ctr = ct_ptr + row * m_dim;
          auto* ct1r = ct1_ptr + row * wdim;

          for (int64_t m = 0; m < m_dim; ++m) {
            ctr[m] = c0r[m];
          }
          for (int64_t i = 0; i < irrep; ++i) {
            const scalar_t xi = xr[i];
            const auto* eri = er + i * m_dim;
            for (int64_t m = 0; m < m_dim; ++m) {
              ctr[m] += xi * eri[m];
            }
          }

          scalar_t final = 0;
          for (int64_t xx = 0; xx < wdim; ++xx) {
            scalar_t acc = c1r[xx];
            for (int64_t ww = 0; ww < wdim; ++ww) {
              acc += ctr[ww * wdim + xx] * xr[ww];
            }
            ct1r[xx] = acc;
            final += acc * xr[xx];
          }
          out_ptr[row] = final;
        };

        const char* grouped_env = std::getenv("NATIVE_CPU_MACE_SC_GROUPED");
        if (grouped_env != nullptr && std::string(grouped_env) != "0") {
          std::vector<int64_t> offsets(elements + 1, 0);
          for (int64_t node = 0; node < batch; ++node) {
            ++offsets[element_ptr[node] + 1];
          }
          for (int64_t e = 0; e < elements; ++e) {
            offsets[e + 1] += offsets[e];
          }
          std::vector<int64_t> cursor = offsets;
          std::vector<int64_t> ordered(batch);
          for (int64_t node = 0; node < batch; ++node) {
            const int64_t e = element_ptr[node];
            ordered[cursor[e]++] = node;
          }

          const int64_t tasks = elements * channels;
          const int64_t threads = std::max<int64_t>(1, at::get_num_threads());
          const int64_t grain = std::max<int64_t>(1, (tasks + threads - 1) / threads);
          native_parallel_for(0, tasks, grain, [&](int64_t begin, int64_t end) {
            for (int64_t task = begin; task < end; ++task) {
              const int64_t e = task / channels;
              const int64_t c = task - e * channels;
              for (int64_t idx = offsets[e]; idx < offsets[e + 1]; ++idx) {
                process_row(ordered[idx], c);
              }
            }
          });
        } else {
          native_parallel_for(0, rows, 256, [&](int64_t begin, int64_t end) {
            for (int64_t row = begin; row < end; ++row) {
              process_row(row / channels, row % channels);
            }
          });
        }
      });

  return std::vector<torch::Tensor>{out, saved_ct, saved_ct1};
}

torch::Tensor symmetric_contraction3_scalar_fused_backward_x(
    torch::Tensor grad_out,
    torch::Tensor x,
    torch::Tensor element,
    torch::Tensor effective,
    torch::Tensor saved_ct,
    torch::Tensor saved_ct1) {
  TORCH_CHECK(grad_out.device().is_cpu() && x.device().is_cpu(),
              "grad_out and x must be CPU tensors");
  TORCH_CHECK(element.device().is_cpu(), "element must be a CPU tensor");
  TORCH_CHECK(effective.device().is_cpu(), "effective must be a CPU tensor");
  TORCH_CHECK(saved_ct.device().is_cpu() && saved_ct1.device().is_cpu(),
              "saved intermediates must be CPU tensors");
  TORCH_CHECK(grad_out.dim() == 2, "grad_out must have shape [B,C]");
  TORCH_CHECK(x.dim() == 3, "x must have shape [B,C,I]");
  TORCH_CHECK(element.scalar_type() == torch::kInt64,
              "element must be int64");

  auto grad_contig = grad_out.contiguous();
  auto x_contig = x.contiguous();
  auto element_contig = element.contiguous();
  auto eff = effective.contiguous();
  auto ct = saved_ct.contiguous();
  auto ct1 = saved_ct1.contiguous();

  const int64_t batch = x_contig.size(0);
  const int64_t channels = x_contig.size(1);
  const int64_t irrep = x_contig.size(2);
  const int64_t elements = eff.size(0);
  const int64_t m_dim = eff.size(3);
  const int64_t wdim = ct1.size(2);
  TORCH_CHECK(eff.size(1) == channels && eff.size(2) == irrep,
              "effective shape mismatch");
  TORCH_CHECK(wdim * wdim == m_dim,
              "effective last dimension must be W*X with W=X");
  TORCH_CHECK(ct.sizes() == torch::IntArrayRef({batch, channels, m_dim}),
              "saved_ct shape mismatch");
  TORCH_CHECK(ct1.sizes() == torch::IntArrayRef({batch, channels, wdim}),
              "saved_ct1 shape mismatch");

  auto grad_x = torch::empty(
      {batch, channels, irrep}, x_contig.options());

  AT_DISPATCH_FLOATING_TYPES(
      x_contig.scalar_type(), "symmetric_contraction3_scalar_fused_backward_x", [&] {
        const auto* grad_ptr = grad_contig.data_ptr<scalar_t>();
        const auto* x_ptr = x_contig.data_ptr<scalar_t>();
        const auto* element_ptr = element_contig.data_ptr<int64_t>();
        const auto* eff_ptr = eff.data_ptr<scalar_t>();
        const auto* ct_ptr = ct.data_ptr<scalar_t>();
        const auto* ct1_ptr = ct1.data_ptr<scalar_t>();
        auto* grad_x_ptr = grad_x.data_ptr<scalar_t>();

        const int64_t rows = batch * channels;
        for (int64_t b = 0; b < batch; ++b) {
          const int64_t e = element_ptr[b];
          TORCH_CHECK(e >= 0 && e < elements, "element index out of range");
        }
        auto process_row = [&](int64_t node, int64_t c) {
          const int64_t row = node * channels + c;
          const int64_t e = element_ptr[node];
          const scalar_t grad = grad_ptr[row];
          const auto* xr = x_ptr + row * irrep;
          const auto* er = eff_ptr + (e * channels + c) * irrep * m_dim;
          const auto* ctr = ct_ptr + row * m_dim;
          const auto* ct1r = ct1_ptr + row * wdim;
          auto* grad_xr = grad_x_ptr + row * irrep;

          if (wdim == 3) {
            const scalar_t x0 = xr[0];
            const scalar_t x1 = xr[1];
            const scalar_t x2 = xr[2];
            const scalar_t outer[9] = {
                x0 * x0, x0 * x1, x0 * x2,
                x1 * x0, x1 * x1, x1 * x2,
                x2 * x0, x2 * x1, x2 * x2,
            };
            for (int64_t i = 0; i < irrep; ++i) {
              scalar_t acc =
                  i < 3 ? grad * ct1r[i] : static_cast<scalar_t>(0);
              if (i < 3) {
                const auto* ctri = ctr + i * 3;
                acc += grad * (
                    x0 * ctri[0] + x1 * ctri[1] + x2 * ctri[2]
                );
              }
              const auto* eri = er + i * 9;
              acc += grad * (
                  outer[0] * eri[0] + outer[1] * eri[1] +
                  outer[2] * eri[2] + outer[3] * eri[3] +
                  outer[4] * eri[4] + outer[5] * eri[5] +
                  outer[6] * eri[6] + outer[7] * eri[7] +
                  outer[8] * eri[8]
              );
              grad_xr[i] = acc;
            }
          } else {
            for (int64_t i = 0; i < irrep; ++i) {
              scalar_t acc =
                  i < wdim ? grad * ct1r[i] : static_cast<scalar_t>(0);
              if (i < wdim) {
                const auto* ctri = ctr + i * wdim;
                for (int64_t xx = 0; xx < wdim; ++xx) {
                  acc += grad * xr[xx] * ctri[xx];
                }
              }
              const auto* eri = er + i * m_dim;
              for (int64_t ww = 0; ww < wdim; ++ww) {
                const scalar_t xw = xr[ww];
                const auto* eriw = eri + ww * wdim;
                for (int64_t xx = 0; xx < wdim; ++xx) {
                  acc += grad * xr[xx] * xw * eriw[xx];
                }
              }
              grad_xr[i] = acc;
            }
          }
        };

        const char* grouped_env = std::getenv("NATIVE_CPU_MACE_SC_GROUPED");
        if (grouped_env != nullptr && std::string(grouped_env) != "0") {
          std::vector<int64_t> offsets(elements + 1, 0);
          for (int64_t node = 0; node < batch; ++node) {
            ++offsets[element_ptr[node] + 1];
          }
          for (int64_t e = 0; e < elements; ++e) {
            offsets[e + 1] += offsets[e];
          }
          std::vector<int64_t> cursor = offsets;
          std::vector<int64_t> ordered(batch);
          for (int64_t node = 0; node < batch; ++node) {
            const int64_t e = element_ptr[node];
            ordered[cursor[e]++] = node;
          }

          const int64_t tasks = elements * channels;
          const int64_t threads = std::max<int64_t>(1, at::get_num_threads());
          const int64_t grain = std::max<int64_t>(1, (tasks + threads - 1) / threads);
          native_parallel_for(0, tasks, grain, [&](int64_t begin, int64_t end) {
            for (int64_t task = begin; task < end; ++task) {
              const int64_t e = task / channels;
              const int64_t c = task - e * channels;
              for (int64_t idx = offsets[e]; idx < offsets[e + 1]; ++idx) {
                process_row(ordered[idx], c);
              }
            }
          });
        } else {
          native_parallel_for(0, rows, 256, [&](int64_t begin, int64_t end) {
            for (int64_t row = begin; row < end; ++row) {
              process_row(row / channels, row % channels);
            }
          });
        }
      });

  return grad_x;
}

torch::Tensor scatter_sum_forward(
    torch::Tensor src,
    torch::Tensor index,
    int64_t dim_size) {
  TORCH_CHECK(src.device().is_cpu(), "scatter src must be CPU");
  TORCH_CHECK(index.device().is_cpu(), "scatter index must be CPU");
  TORCH_CHECK(src.dim() == 2, "scatter src must be [E, D]");
  TORCH_CHECK(index.scalar_type() == torch::kInt64, "scatter index must be int64");
  TORCH_CHECK(src.size(0) == index.numel(), "scatter size mismatch");
  TORCH_CHECK(dim_size >= 0, "scatter dim_size must be non-negative");

  auto src_contig = src.contiguous();
  auto index_contig = index.contiguous();
  const int64_t edges = src_contig.size(0);
  const int64_t dim = src_contig.size(1);
  auto out = parallel_zeros({dim_size, dim}, src_contig.options());
  if (edges == 0 || dim == 0) {
    return out;
  }

  AT_DISPATCH_FLOATING_TYPES(
      src_contig.scalar_type(), "scatter_sum_forward", [&] {
        const auto* src_ptr = src_contig.data_ptr<scalar_t>();
        const auto* index_ptr = index_contig.data_ptr<int64_t>();
        auto* out_ptr = out.data_ptr<scalar_t>();

        std::vector<int64_t> offsets(dim_size + 1, 0);
        for (int64_t edge = 0; edge < edges; ++edge) {
          const int64_t node = index_ptr[edge];
          TORCH_CHECK(node >= 0 && node < dim_size,
                      "scatter index out of range");
          ++offsets[node + 1];
        }
        for (int64_t node = 0; node < dim_size; ++node) {
          offsets[node + 1] += offsets[node];
        }

        std::vector<int64_t> cursor = offsets;
        std::vector<int64_t> order(edges);
        for (int64_t edge = 0; edge < edges; ++edge) {
          const int64_t node = index_ptr[edge];
          order[cursor[node]++] = edge;
        }

        native_parallel_for(0, dim_size, 1, [&](int64_t begin, int64_t end) {
          for (int64_t node = begin; node < end; ++node) {
            auto* out_row = out_ptr + node * dim;
            for (int64_t item = offsets[node]; item < offsets[node + 1]; ++item) {
              const auto* src_row = src_ptr + order[item] * dim;
              for (int64_t col = 0; col < dim; ++col) {
                out_row[col] += src_row[col];
              }
            }
          }
        });
      });
  return out;
}

torch::Tensor scatter_sum_backward(
    torch::Tensor grad_out,
    torch::Tensor index,
    int64_t num_edges) {
  TORCH_CHECK(grad_out.device().is_cpu(), "scatter grad must be CPU");
  TORCH_CHECK(index.device().is_cpu(), "scatter index must be CPU");
  TORCH_CHECK(grad_out.dim() == 2, "scatter grad must be [N, D]");
  TORCH_CHECK(index.scalar_type() == torch::kInt64, "scatter index must be int64");

  auto grad_contig = grad_out.contiguous();
  auto index_contig = index.contiguous();
  const int64_t dim = grad_contig.size(1);
  auto grad_src = torch::empty({num_edges, dim}, grad_contig.options());

  AT_DISPATCH_FLOATING_TYPES(
      grad_contig.scalar_type(), "scatter_sum_backward", [&] {
        const auto* grad_ptr = grad_contig.data_ptr<scalar_t>();
        const auto* index_ptr = index_contig.data_ptr<int64_t>();
        auto* grad_src_ptr = grad_src.data_ptr<scalar_t>();
        native_parallel_for(0, num_edges, 64, [&](int64_t begin, int64_t end) {
          for (int64_t edge = begin; edge < end; ++edge) {
            const int64_t node = index_ptr[edge];
            const auto* grad_row = grad_ptr + node * dim;
            auto* out_row = grad_src_ptr + edge * dim;
            std::memcpy(out_row, grad_row, dim * sizeof(scalar_t));
          }
        });
      });
  return grad_src;
}

torch::Tensor radial_embedding_forward(
    torch::Tensor lengths,
    torch::Tensor bessel_weights,
    torch::Tensor r_max,
    int64_t cutoff_p) {
  TORCH_CHECK(lengths.device().is_cpu(), "radial embedding lengths must be CPU");
  TORCH_CHECK(bessel_weights.device().is_cpu(), "bessel weights must be CPU");
  TORCH_CHECK(lengths.dim() == 2 && lengths.size(1) == 1,
              "lengths must have shape [E,1]");
  TORCH_CHECK(bessel_weights.dim() == 1, "bessel weights must be [B]");
  TORCH_CHECK(lengths.scalar_type() == bessel_weights.scalar_type(),
              "radial embedding dtype mismatch");
  auto lengths_contig = lengths.contiguous();
  auto weights_contig = bessel_weights.contiguous();
  const int64_t edges = lengths_contig.size(0);
  const int64_t basis = weights_contig.size(0);
  const double rmax = r_max.to(torch::kCPU).data_ptr<double>()[0];
  const double p = static_cast<double>(cutoff_p);
  const double prefactor = std::sqrt(2.0 / rmax);
  auto out = torch::empty({edges, basis}, lengths_contig.options());

  AT_DISPATCH_FLOATING_TYPES(
      lengths_contig.scalar_type(), "radial_embedding_forward", [&] {
        const auto* lengths_ptr = lengths_contig.data_ptr<scalar_t>();
        const auto* weights_ptr = weights_contig.data_ptr<scalar_t>();
        auto* out_ptr = out.data_ptr<scalar_t>();
        native_parallel_for(0, edges, 1, [&](int64_t begin, int64_t end) {
          for (int64_t edge = begin; edge < end; ++edge) {
            const double r = std::max(
                static_cast<double>(lengths_ptr[edge]),
                1.0e-12
            );
            const double x = r / rmax;
            const double xp = std::pow(x, p);
            const double envelope =
                1.0 -
                ((p + 1.0) * (p + 2.0) / 2.0) * xp +
                p * (p + 2.0) * xp * x -
                (p * (p + 1.0) / 2.0) * xp * x * x;
            auto* out_row = out_ptr + edge * basis;
            for (int64_t k = 0; k < basis; ++k) {
              const double w = static_cast<double>(weights_ptr[k]);
              out_row[k] = static_cast<scalar_t>(
                  prefactor * std::sin(w * r) / r * envelope
              );
            }
          }
        });
      });
  return out;
}

#if 1
std::vector<torch::Tensor> edge_geometry_forward(
    torch::Tensor positions,
    torch::Tensor edge_index,
    torch::Tensor shifts,
    bool normalize,
    double eps) {
  TORCH_CHECK(positions.device().is_cpu(), "positions must be CPU");
  TORCH_CHECK(edge_index.device().is_cpu(), "edge_index must be CPU");
  TORCH_CHECK(shifts.device().is_cpu(), "shifts must be CPU");
  TORCH_CHECK(
      positions.dim() == 2 && positions.size(1) == 3,
      "positions must have shape [N,3]");
  TORCH_CHECK(
      edge_index.dim() == 2 && edge_index.size(0) == 2,
      "edge_index must have shape [2,E]");
  TORCH_CHECK(
      edge_index.scalar_type() == torch::kInt64,
      "edge_index must be int64");
  TORCH_CHECK(
      shifts.dim() == 2 && shifts.size(0) == edge_index.size(1) &&
          shifts.size(1) == 3,
      "shifts must have shape [E,3]");
  auto positions_contig = positions.contiguous();
  auto edge_index_contig = edge_index.contiguous();
  auto shifts_contig = shifts.contiguous();
  const int64_t edges = edge_index_contig.size(1);
  auto vectors = torch::empty({edges, 3}, positions_contig.options());
  auto lengths = torch::empty({edges, 1}, positions_contig.options());

  AT_DISPATCH_FLOATING_TYPES(
      positions_contig.scalar_type(), "edge_geometry_forward", [&] {
        const auto* positions_ptr = positions_contig.data_ptr<scalar_t>();
        const auto* edge_ptr = edge_index_contig.data_ptr<int64_t>();
        const auto* shifts_ptr = shifts_contig.data_ptr<scalar_t>();
        auto* vectors_ptr = vectors.data_ptr<scalar_t>();
        auto* lengths_ptr = lengths.data_ptr<scalar_t>();
        const scalar_t safe_eps = static_cast<scalar_t>(eps);
        native_parallel_for(0, edges, 64, [&](int64_t begin, int64_t end) {
          for (int64_t edge = begin; edge < end; ++edge) {
            const int64_t sender = edge_ptr[edge];
            const int64_t receiver = edge_ptr[edges + edge];
            scalar_t vx = positions_ptr[receiver * 3] -
                positions_ptr[sender * 3] + shifts_ptr[edge * 3];
            scalar_t vy = positions_ptr[receiver * 3 + 1] -
                positions_ptr[sender * 3 + 1] + shifts_ptr[edge * 3 + 1];
            scalar_t vz = positions_ptr[receiver * 3 + 2] -
                positions_ptr[sender * 3 + 2] + shifts_ptr[edge * 3 + 2];
            const scalar_t length = std::sqrt(
                vx * vx + vy * vy + vz * vz);
            lengths_ptr[edge] = length;
            if (normalize) {
              const scalar_t denom = std::max(length, safe_eps);
              vx /= denom;
              vy /= denom;
              vz /= denom;
            }
            vectors_ptr[edge * 3] = vx;
            vectors_ptr[edge * 3 + 1] = vy;
            vectors_ptr[edge * 3 + 2] = vz;
          }
        });
      });
  return {vectors, lengths};
}

std::vector<torch::Tensor> edge_geometry_backward(
    torch::Tensor grad_vectors,
    torch::Tensor grad_lengths,
    torch::Tensor vectors,
    torch::Tensor lengths,
    torch::Tensor edge_index,
    int64_t num_nodes,
    bool normalize,
    double eps) {
  auto grad_vectors_contig = grad_vectors.contiguous();
  auto grad_lengths_contig = grad_lengths.contiguous();
  auto vectors_contig = vectors.contiguous();
  auto lengths_contig = lengths.contiguous();
  auto edge_index_contig = edge_index.contiguous();
  const int64_t edges = edge_index_contig.size(1);
  auto grad_positions = torch::zeros(
      {num_nodes, 3}, vectors_contig.options());
  auto grad_shifts = torch::empty_like(vectors_contig);

  AT_DISPATCH_FLOATING_TYPES(
      vectors_contig.scalar_type(), "edge_geometry_backward", [&] {
        const auto* grad_vectors_ptr =
            grad_vectors_contig.data_ptr<scalar_t>();
        const auto* grad_lengths_ptr =
            grad_lengths_contig.data_ptr<scalar_t>();
        const auto* vectors_ptr = vectors_contig.data_ptr<scalar_t>();
        const auto* lengths_ptr = lengths_contig.data_ptr<scalar_t>();
        const auto* edge_ptr = edge_index_contig.data_ptr<int64_t>();
        auto* grad_positions_ptr = grad_positions.data_ptr<scalar_t>();
        auto* grad_shifts_ptr = grad_shifts.data_ptr<scalar_t>();
        const scalar_t safe_eps = static_cast<scalar_t>(eps);
        native_parallel_for(0, edges, 64, [&](int64_t begin, int64_t end) {
          for (int64_t edge = begin; edge < end; ++edge) {
            const scalar_t length = std::max(
                lengths_ptr[edge], safe_eps);
            const scalar_t g0 = grad_vectors_ptr[edge * 3];
            const scalar_t g1 = grad_vectors_ptr[edge * 3 + 1];
            const scalar_t g2 = grad_vectors_ptr[edge * 3 + 2];
            const scalar_t gl = grad_lengths_ptr[edge];
            const scalar_t v0 = vectors_ptr[edge * 3];
            const scalar_t v1 = vectors_ptr[edge * 3 + 1];
            const scalar_t v2 = vectors_ptr[edge * 3 + 2];
            scalar_t gv0;
            scalar_t gv1;
            scalar_t gv2;
            if (normalize) {
              const scalar_t n0 = v0;
              const scalar_t n1 = v1;
              const scalar_t n2 = v2;
              const scalar_t dot = g0 * n0 + g1 * n1 + g2 * n2;
              const scalar_t inv_length =
                  static_cast<scalar_t>(1.0) / length;
              gv0 = (g0 - n0 * dot) * inv_length + gl * n0;
              gv1 = (g1 - n1 * dot) * inv_length + gl * n1;
              gv2 = (g2 - n2 * dot) * inv_length + gl * n2;
            } else {
              const scalar_t inv_length =
                  gl / length;
              gv0 = g0 + inv_length * v0;
              gv1 = g1 + inv_length * v1;
              gv2 = g2 + inv_length * v2;
            }
            grad_shifts_ptr[edge * 3] = gv0;
            grad_shifts_ptr[edge * 3 + 1] = gv1;
            grad_shifts_ptr[edge * 3 + 2] = gv2;
          }
        });
        for (int64_t edge = 0; edge < edges; ++edge) {
          const int64_t sender = edge_ptr[edge];
          const int64_t receiver = edge_ptr[edges + edge];
          const scalar_t gv0 = grad_shifts_ptr[edge * 3];
          const scalar_t gv1 = grad_shifts_ptr[edge * 3 + 1];
          const scalar_t gv2 = grad_shifts_ptr[edge * 3 + 2];
          grad_positions_ptr[receiver * 3] += gv0;
          grad_positions_ptr[receiver * 3 + 1] += gv1;
          grad_positions_ptr[receiver * 3 + 2] += gv2;
          grad_positions_ptr[sender * 3] -= gv0;
          grad_positions_ptr[sender * 3 + 1] -= gv1;
          grad_positions_ptr[sender * 3 + 2] -= gv2;
        }
      });
  return {grad_positions, grad_shifts};
}
#endif

torch::Tensor radial_embedding_backward(
    torch::Tensor grad_out,
    torch::Tensor lengths,
    torch::Tensor bessel_weights,
    torch::Tensor r_max,
    int64_t cutoff_p) {
  TORCH_CHECK(grad_out.device().is_cpu(), "radial embedding grad must be CPU");
  auto grad_contig = grad_out.contiguous();
  auto lengths_contig = lengths.contiguous();
  auto weights_contig = bessel_weights.contiguous();
  const int64_t edges = lengths_contig.size(0);
  const int64_t basis = weights_contig.size(0);
  const double rmax = r_max.to(torch::kCPU).data_ptr<double>()[0];
  const double p = static_cast<double>(cutoff_p);
  const double prefactor = std::sqrt(2.0 / rmax);
  auto grad_lengths = torch::empty_like(lengths_contig);

  AT_DISPATCH_FLOATING_TYPES(
      lengths_contig.scalar_type(), "radial_embedding_backward", [&] {
        const auto* grad_ptr = grad_contig.data_ptr<scalar_t>();
        const auto* lengths_ptr = lengths_contig.data_ptr<scalar_t>();
        const auto* weights_ptr = weights_contig.data_ptr<scalar_t>();
        auto* grad_lengths_ptr = grad_lengths.data_ptr<scalar_t>();
        native_parallel_for(0, edges, 1, [&](int64_t begin, int64_t end) {
          for (int64_t edge = begin; edge < end; ++edge) {
            const double r = std::max(
                static_cast<double>(lengths_ptr[edge]),
                1.0e-12
            );
            const double x = r / rmax;
            const double xp = std::pow(x, p);
            const double envelope =
                1.0 -
                ((p + 1.0) * (p + 2.0) / 2.0) * xp +
                p * (p + 2.0) * xp * x -
                (p * (p + 1.0) / 2.0) * xp * x * x;
            const double denvelope =
                (
                    -((p + 1.0) * (p + 2.0) / 2.0) * p * std::pow(x, p - 1.0) +
                    p * (p + 2.0) * (p + 1.0) * xp -
                    (p * (p + 1.0) / 2.0) * (p + 2.0) * xp * x
                ) / rmax;
            double acc = 0;
            for (int64_t k = 0; k < basis; ++k) {
              const double w = static_cast<double>(weights_ptr[k]);
              const double bessel = prefactor * std::sin(w * r) / r;
              const double dbessel = prefactor * (
                  w * std::cos(w * r) / r -
                  std::sin(w * r) / (r * r)
              );
              acc += static_cast<double>(grad_ptr[edge * basis + k]) *
                  (dbessel * envelope + bessel * denvelope);
            }
            grad_lengths_ptr[edge] = static_cast<scalar_t>(acc);
          }
        });
      });
  return grad_lengths;
}

std::vector<int64_t> native_thread_info() {
#if defined(_OPENMP)
  return {
      at::get_num_threads(),
      static_cast<int64_t>(omp_get_max_threads()),
      static_cast<int64_t>(omp_get_num_procs()),
  };
#else
  return {at::get_num_threads(), 1, 1};
#endif
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("sh3_forward", &sh3_forward, "Native CPU Cartesian SH l=0..3");
  module.def("sh3_backward", &sh3_backward, "Native CPU Cartesian SH l=0..3 backward");
  module.def("fctp_forward", &fctp_forward, "Native CPU FCTP forward");
  module.def("fctp_backward_x", &fctp_backward_x, "Native CPU FCTP grad_x");
  module.def("tpu_forward", &tpu_forward, "Native CPU uvu tensor product forward");
  module.def("tpu_backward", &tpu_backward, "Native CPU uvu tensor product backward");
  module.def("linear_forward", &linear_forward, "Native CPU equivariant linear forward");
  module.def("linear_backward_x", &linear_backward_x, "Native CPU equivariant linear grad_x");
  module.def(
      "fctp_onehot_packed_forward",
      &fctp_onehot_packed_forward,
      "Native CPU packed one-hot FCTP forward");
  module.def(
      "fctp_onehot_packed_backward_x",
      &fctp_onehot_packed_backward_x,
      "Native CPU packed one-hot FCTP grad_x");
  module.def(
      "linear_packed_forward",
      &linear_packed_forward,
      "Native CPU packed linear forward");
  module.def(
      "linear_packed_backward_x",
      &linear_packed_backward_x,
      "Native CPU packed linear grad_x");
  module.def(
      "radial_mlp_forward",
      &radial_mlp_forward,
      "Native CPU radial MLP forward");
  module.def(
      "radial_mlp_backward",
      &radial_mlp_backward,
      "Native CPU radial MLP grad_x");
  module.def(
      "symmetric_contraction3_forward",
      &symmetric_contraction3_forward,
      "Native CPU symmetric contraction correlation=3");
  module.def(
      "symmetric_contraction3_backward_x",
      &symmetric_contraction3_backward_x,
      "Native CPU symmetric contraction correlation=3 grad_x");
  module.def(
      "symmetric_contraction3_scalar_fused_forward",
      &symmetric_contraction3_scalar_fused_forward,
      "Fused CPU scalar symmetric contraction corr=3 forward");
  module.def(
      "symmetric_contraction3_scalar_fused_backward_x",
      &symmetric_contraction3_scalar_fused_backward_x,
      "Fused CPU scalar symmetric contraction corr=3 grad_x");
  module.def(
      "scatter_sum_forward",
      &scatter_sum_forward,
      "Native CPU CSR scatter sum forward");
  module.def(
      "scatter_sum_backward",
      &scatter_sum_backward,
      "Native CPU scatter sum input gradient");
  module.def(
      "radial_embedding_forward",
      &radial_embedding_forward,
      "Native CPU Bessel radial embedding forward");
#if 1
  module.def(
      "edge_geometry_forward",
      &edge_geometry_forward,
      "Native CPU edge vectors and lengths");
  module.def(
      "edge_geometry_backward",
      &edge_geometry_backward,
      "Native CPU edge geometry backward");
#endif
  module.def(
      "radial_embedding_backward",
      &radial_embedding_backward,
      "Native CPU Bessel radial embedding grad_lengths");
  module.def(
      "native_thread_info",
      &native_thread_info,
      "Native CPU thread configuration");
}
