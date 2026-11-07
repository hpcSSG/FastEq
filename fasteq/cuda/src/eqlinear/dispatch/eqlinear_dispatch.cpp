// Torch-facing eqlinear dispatcher.  This is the only file in the eqlinear tree that includes
// torch; the per-precision backends are torch-free DSOs reached through a small C ABI.

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "eqlinear_backend_api.h"
#include "eqlinear_group_table.h"

extern "C" const fasteq::eqlinear::EqLinearBackendApi* fasteq_eqlinear_tf32_api();
extern "C" const fasteq::eqlinear::EqLinearBackendApi* fasteq_eqlinear_fp32_api();
extern "C" const fasteq::eqlinear::EqLinearBackendApi* fasteq_eqlinear_fp64_api();

namespace {

using fasteq::eqlinear::EqLinearBackendApi;
using fasteq::eqlinear::EqLinearPlanInfo;
using fasteq::eqlinear::EqLinearProblemC;

const EqLinearBackendApi* tf32_api() {
  static const EqLinearBackendApi* api = fasteq_eqlinear_tf32_api();
  return api;
}
const EqLinearBackendApi* fp32_api() {
  static const EqLinearBackendApi* api = fasteq_eqlinear_fp32_api();
  return api;
}
const EqLinearBackendApi* fp64_api() {
  static const EqLinearBackendApi* api = fasteq_eqlinear_fp64_api();
  return api;
}

const EqLinearBackendApi* api_for(int64_t precision) {
  TORCH_CHECK(precision >= 0 && precision <= 2,
              "eqlinear: precision must be 0 (tf32), 1 (fp32), or 2 (fp64); got ", precision);
  const EqLinearBackendApi* api = precision == 0 ? tf32_api() : (precision == 1 ? fp32_api() : fp64_api());
  TORCH_CHECK(api != nullptr && api->abi_version == fasteq::eqlinear::kEqLinearAbiVersion,
              "eqlinear: backend ABI mismatch for precision ", precision);
  return api;
}

struct Handle {
  const EqLinearBackendApi* api = nullptr;
  int64_t precision = 0;
  int64_t batch = 0, input_rows = 0, u = 0, output_rows = 0, v = 0;
  int device_index = -1;
  void* forward_plan = nullptr;
  void* dgrad_plan = nullptr;
  EqLinearPlanInfo forward_info;
  EqLinearPlanInfo dgrad_info;

  ~Handle() {
    if (api != nullptr) {
      if (forward_plan != nullptr) {
        api->forward_destroy(forward_plan);
      }
      if (dgrad_plan != nullptr) {
        api->dgrad_destroy(dgrad_plan);
      }
    }
  }
};

std::mutex g_mutex;
std::unordered_map<int64_t, std::shared_ptr<Handle>> g_handles;
int64_t g_next_handle = 1;

std::shared_ptr<Handle> find_handle(int64_t handle) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_handles.find(handle);
  TORCH_CHECK(it != g_handles.end(), "eqlinear: unknown handle ", handle);
  return it->second;
}

int64_t register_handle(std::shared_ptr<Handle> handle) {
  std::lock_guard<std::mutex> lock(g_mutex);
  const int64_t id = g_next_handle++;
  g_handles.emplace(id, std::move(handle));
  return id;
}

void erase_handle(int64_t id) {
  std::shared_ptr<Handle> retired;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_handles.find(id);
    if (it == g_handles.end()) return;
    retired = std::move(it->second);
    g_handles.erase(it);
  }
  // The last reference may synchronize the device; never hold the registry lock here.
}

int32_t sum_i_dims(const std::vector<int64_t>& i_dims) {
  int64_t total = 0;
  for (int64_t v : i_dims) {
    TORCH_CHECK(v > 0 && v <= INT32_MAX, "eqlinear: i_dims must be positive; got ", v);
    total += v;
  }
  TORCH_CHECK(total <= INT32_MAX, "eqlinear: sum(i_dims) overflow");
  return static_cast<int32_t>(total);
}

std::vector<int> to_int_dims(const std::vector<int64_t>& i_dims) {
  std::vector<int> out;
  out.reserve(i_dims.size());
  for (int64_t v : i_dims) {
    out.push_back(static_cast<int>(v));
  }
  return out;
}

std::vector<int32_t> to_int32_dims(const std::vector<int64_t>& i_dims) {
  std::vector<int32_t> out;
  out.reserve(i_dims.size());
  for (int64_t v : i_dims) {
    out.push_back(static_cast<int32_t>(v));
  }
  return out;
}

EqLinearProblemC make_problem(const std::vector<int64_t>& i_dims,
                              const std::vector<int32_t>& i_dims32, int64_t B, int64_t U,
                              int64_t V, int64_t P) {
  EqLinearProblemC problem;
  problem.B = static_cast<int32_t>(B);
  problem.I = sum_i_dims(i_dims);
  problem.U = static_cast<int32_t>(U);
  problem.P = static_cast<int32_t>(P);
  problem.V = static_cast<int32_t>(V);
  problem.J = fasteq::eqlinear::out_total_i(to_int_dims(i_dims));
  problem.i_dims = i_dims32.data();
  return problem;
}

void check_alignment(const torch::Tensor& t, const char* name) {
  TORCH_CHECK((reinterpret_cast<uintptr_t>(t.data_ptr()) & 15) == 0,
              "eqlinear: ", name, " must be 16-byte aligned");
}

void check_contiguous_cuda(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), "eqlinear: ", name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), "eqlinear: ", name, " must be contiguous");
}

void check_dtype(const torch::Tensor& t, int64_t precision, const char* name) {
  if (precision == 2) {
    TORCH_CHECK(t.scalar_type() == torch::kFloat64,
                "eqlinear fp64: ", name, " must be float64; got ", t.scalar_type());
  } else {
    TORCH_CHECK(t.scalar_type() == torch::kFloat32,
                "eqlinear: ", name, " must be float32; got ", t.scalar_type());
  }
}

void check_multiple(int64_t value, int64_t alignment, const char* name) {
  TORCH_CHECK(value % alignment == 0,
              "eqlinear: ", name, "=", value, " must be a multiple of ", alignment);
}

void validate_problem(const EqLinearProblemC& p, int64_t precision) {
  TORCH_CHECK(p.P >= 1 && p.P <= 32, "eqlinear: P must be in [1,32]; got ", p.P);
  TORCH_CHECK(p.B > 0 && p.I > 0 && p.U > 0 && p.V > 0, "eqlinear: empty problem");
  const int64_t alignment = precision == 2 ? 2 : 4;
  check_multiple(p.B, alignment, "B");
  check_multiple(p.V, alignment, "V");
  check_multiple(static_cast<int64_t>(p.I) * p.U, alignment, "I*U");
  check_multiple(static_cast<int64_t>(p.P) * p.U, alignment, "P*U");
  check_multiple(static_cast<int64_t>(p.J) * p.V, alignment, "J*V");
  if (precision == 1) {
    TORCH_CHECK(p.U == p.V,
                "eqlinear fp32: backward/forward kFused requires U == V; got U=", p.U,
                " V=", p.V);
    check_multiple(p.U, 16, "U");
    TORCH_CHECK(p.V == 96 || p.V == 128 || p.V == 224,
                "eqlinear fp32: V=", p.V, " is outside the measured kFused tile set {96,128,224}");
  }
  if (precision == 2 && p.V <= 128) {
    check_multiple(p.U, 16, "U");
  }
}

void check_capability() {
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 9 && props->minor == 0,
              "eqlinear requires Hopper sm_90a; got sm_", props->major, props->minor);
}

torch::Tensor forward_impl(const torch::Tensor& x, const torch::Tensor& w,
                           const std::vector<int64_t>& i_dims,
                           const std::vector<double>& cg, int64_t precision) {
  check_contiguous_cuda(x, "x");
  check_contiguous_cuda(w, "w");
  check_dtype(x, precision, "x");
  check_dtype(w, precision, "w");
  TORCH_CHECK(x.dim() == 3, "eqlinear: x must be [B,I,U]");
  TORCH_CHECK(w.dim() == 3, "eqlinear: w must be [P,U,V]");
  TORCH_CHECK(x.device() == w.device(), "eqlinear: x and w must be on the same device");
  TORCH_CHECK(cg.size() == i_dims.size(), "eqlinear: cg length must equal len(i_dims)");
  const c10::cuda::CUDAGuard device_guard(w.device());
  check_capability();

  const int64_t B = x.size(0);
  const int64_t I = x.size(1);
  const int64_t U = x.size(2);
  const int64_t P = w.size(0);
  const int64_t V = w.size(2);
  TORCH_CHECK(w.size(1) == U, "eqlinear: w.size(1) must equal U");
  TORCH_CHECK(P == static_cast<int64_t>(i_dims.size()), "eqlinear: w.size(0) must equal len(i_dims)");
  TORCH_CHECK(I == sum_i_dims(i_dims), "eqlinear: x.size(1) must equal sum(i_dims)");
  check_alignment(x, "x");
  check_alignment(w, "w");

  const std::vector<int32_t> i_dims32 = to_int32_dims(i_dims);
  const EqLinearProblemC problem = make_problem(i_dims, i_dims32, B, U, V, P);
  validate_problem(problem, precision);
  check_alignment(x, "x");

  auto out = torch::empty({B, problem.J, V}, x.options());
  check_alignment(out, "out");

  const auto* api = api_for(precision);
  void* plan = nullptr;
  EqLinearPlanInfo info;
  const int create_rc = api->forward_create(&problem, w.data_ptr(), cg.data(),
                                            c10::cuda::getCurrentCUDAStream().stream(), &plan,
                                            &info);
  TORCH_CHECK(create_rc == 0, api->status_string(create_rc));
  const int run_rc = api->forward_run(plan, x.data_ptr(), out.data_ptr(),
                                      c10::cuda::getCurrentCUDAStream().stream());
  api->forward_destroy(plan);
  TORCH_CHECK(run_rc == 0, api->status_string(run_rc));
  return out;
}

torch::Tensor dgrad_impl(const torch::Tensor& dy, const torch::Tensor& w,
                         const std::vector<int64_t>& i_dims,
                         const std::vector<double>& cg, int64_t precision) {
  check_contiguous_cuda(dy, "dy");
  check_contiguous_cuda(w, "w");
  check_dtype(dy, precision, "dy");
  check_dtype(w, precision, "w");
  TORCH_CHECK(dy.dim() == 3, "eqlinear: dy must be [B,J,V]");
  TORCH_CHECK(w.dim() == 3, "eqlinear: w must be [P,U,V]");
  TORCH_CHECK(dy.device() == w.device(), "eqlinear: dy and w must be on the same device");
  TORCH_CHECK(cg.size() == i_dims.size(), "eqlinear: cg length must equal len(i_dims)");
  const c10::cuda::CUDAGuard device_guard(w.device());
  check_capability();

  const int64_t B = dy.size(0);
  const int64_t J = dy.size(1);
  const int64_t V = dy.size(2);
  const int64_t P = w.size(0);
  const int64_t U = w.size(1);
  TORCH_CHECK(w.size(2) == V, "eqlinear: w.size(2) must equal dy.size(2)");
  TORCH_CHECK(P == static_cast<int64_t>(i_dims.size()), "eqlinear: w.size(0) must equal len(i_dims)");

  const std::vector<int32_t> i_dims32 = to_int32_dims(i_dims);
  const EqLinearProblemC problem = make_problem(i_dims, i_dims32, B, U, V, P);
  TORCH_CHECK(problem.J == J, "eqlinear: dy.size(1)=", J, " but out_total_i=", problem.J);
  validate_problem(problem, precision);
  check_alignment(dy, "dy");
  check_alignment(w, "w");

  auto dx = torch::empty({B, problem.I, U}, dy.options());
  check_alignment(dx, "dx");

  const auto* api = api_for(precision);
  void* plan = nullptr;
  EqLinearPlanInfo info;
  const int create_rc = api->dgrad_create(&problem, w.data_ptr(), cg.data(),
                                          c10::cuda::getCurrentCUDAStream().stream(), &plan,
                                          &info);
  TORCH_CHECK(create_rc == 0, api->status_string(create_rc));
  const int run_rc = api->dgrad_run(plan, dy.data_ptr(), dx.data_ptr(),
                                    c10::cuda::getCurrentCUDAStream().stream());
  api->dgrad_destroy(plan);
  TORCH_CHECK(run_rc == 0, api->status_string(run_rc));
  return dx;
}

int64_t out_rows(const std::vector<int64_t>& i_dims) {
  return static_cast<int64_t>(fasteq::eqlinear::out_total_i(to_int_dims(i_dims)));
}

int64_t forward_create(const torch::Tensor& w, const std::vector<int64_t>& i_dims,
                       const std::vector<double>& cg, int64_t B, int64_t U, int64_t P,
                       int64_t V, int64_t precision) {
  check_contiguous_cuda(w, "w");
  check_dtype(w, precision, "w");
  TORCH_CHECK(w.dim() == 3, "eqlinear: w must be [P,U,V]");
  TORCH_CHECK(P == static_cast<int64_t>(i_dims.size()), "eqlinear: P must equal len(i_dims)");
  TORCH_CHECK(w.size(0) == P && w.size(1) == U && w.size(2) == V,
              "eqlinear: w shape does not match P,U,V");
  TORCH_CHECK(cg.size() == i_dims.size(), "eqlinear: cg length must equal len(i_dims)");
  const c10::cuda::CUDAGuard device_guard(w.device());
  check_capability();
  check_alignment(w, "w");

  const std::vector<int32_t> i_dims32 = to_int32_dims(i_dims);
  const EqLinearProblemC problem = make_problem(i_dims, i_dims32, B, U, V, P);
  validate_problem(problem, precision);
  const auto* api = api_for(precision);
  auto handle = std::make_shared<Handle>();
  handle->api = api;
  handle->precision = precision;
  handle->batch = problem.B;
  handle->input_rows = problem.I;
  handle->u = problem.U;
  handle->output_rows = problem.J;
  handle->v = problem.V;
  handle->device_index = w.get_device();
  const int rc = api->forward_create(&problem, w.data_ptr(), cg.data(),
                                     c10::cuda::getCurrentCUDAStream().stream(),
                                     &handle->forward_plan, &handle->forward_info);
  TORCH_CHECK(rc == 0, api->status_string(rc));
  return register_handle(std::move(handle));
}

void forward_run(int64_t handle, const torch::Tensor& x, const torch::Tensor& out) {
  auto h = find_handle(handle);
  TORCH_CHECK(h->forward_plan != nullptr, "eqlinear: handle has no forward plan");
  check_contiguous_cuda(x, "x");
  check_contiguous_cuda(out, "out");
  check_dtype(x, h->precision, "x");
  check_dtype(out, h->precision, "out");
  TORCH_CHECK(x.dim() == 3 && out.dim() == 3, "eqlinear: x/out must be 3-D");
  TORCH_CHECK(x.get_device() == h->device_index &&
                  out.get_device() == h->device_index,
              "eqlinear: x/out must be on the plan device");
  check_alignment(x, "x");
  check_alignment(out, "out");
  const c10::cuda::CUDAGuard device_guard(x.device());
  const int rc = h->api->forward_run(h->forward_plan, x.data_ptr(), out.data_ptr(),
                                     c10::cuda::getCurrentCUDAStream().stream());
  TORCH_CHECK(rc == 0, h->api->status_string(rc));
}

void forward_run_flat(int64_t handle, const torch::Tensor& x, const torch::Tensor& out) {
  auto h = find_handle(handle);
  TORCH_CHECK(h->forward_plan != nullptr, "eqlinear: handle has no forward plan");
  check_contiguous_cuda(x, "x");
  check_contiguous_cuda(out, "out");
  check_dtype(x, h->precision, "x");
  check_dtype(out, h->precision, "out");
  TORCH_CHECK(x.dim() == 2 && out.dim() == 2, "eqlinear: x/out must be 2-D");
  TORCH_CHECK(x.get_device() == h->device_index && out.get_device() == h->device_index,
              "eqlinear: x/out must be on the plan device");
  TORCH_CHECK(x.size(0) == h->batch && x.size(1) == h->input_rows * h->u,
              "eqlinear: flat x shape does not match the plan");
  TORCH_CHECK(out.size(0) == h->batch && out.size(1) == h->output_rows * h->v,
              "eqlinear: flat out shape does not match the plan");
  check_alignment(x, "x");
  check_alignment(out, "out");
  const c10::cuda::CUDAGuard device_guard(x.device());
  const int rc = h->api->forward_run(h->forward_plan, x.data_ptr(), out.data_ptr(),
                                     c10::cuda::getCurrentCUDAStream().stream());
  TORCH_CHECK(rc == 0, h->api->status_string(rc));
}

void forward_destroy(int64_t handle) { erase_handle(handle); }

std::tuple<int64_t, int64_t, int64_t, int64_t> forward_plan_info(int64_t handle) {
  auto h = find_handle(handle);
  TORCH_CHECK(h->forward_plan != nullptr, "eqlinear: handle has no forward plan");
  return {h->forward_info.precision, h->forward_info.forward_impl, h->forward_info.a_view,
          h->forward_info.j_rows};
}

int64_t dgrad_create(const torch::Tensor& w, const std::vector<int64_t>& i_dims,
                     const std::vector<double>& cg, int64_t B, int64_t U, int64_t P,
                     int64_t V, int64_t precision) {
  check_contiguous_cuda(w, "w");
  check_dtype(w, precision, "w");
  TORCH_CHECK(w.dim() == 3, "eqlinear: w must be [P,U,V]");
  TORCH_CHECK(P == static_cast<int64_t>(i_dims.size()), "eqlinear: P must equal len(i_dims)");
  TORCH_CHECK(w.size(0) == P && w.size(1) == U && w.size(2) == V,
              "eqlinear: w shape does not match P,U,V");
  TORCH_CHECK(cg.size() == i_dims.size(), "eqlinear: cg length must equal len(i_dims)");
  const c10::cuda::CUDAGuard device_guard(w.device());
  check_capability();
  check_alignment(w, "w");

  const std::vector<int32_t> i_dims32 = to_int32_dims(i_dims);
  const EqLinearProblemC problem = make_problem(i_dims, i_dims32, B, U, V, P);
  validate_problem(problem, precision);
  const auto* api = api_for(precision);
  auto handle = std::make_shared<Handle>();
  handle->api = api;
  handle->precision = precision;
  handle->batch = problem.B;
  handle->input_rows = problem.I;
  handle->u = problem.U;
  handle->output_rows = problem.J;
  handle->v = problem.V;
  handle->device_index = w.get_device();
  const int rc = api->dgrad_create(&problem, w.data_ptr(), cg.data(),
                                   c10::cuda::getCurrentCUDAStream().stream(),
                                   &handle->dgrad_plan, &handle->dgrad_info);
  TORCH_CHECK(rc == 0, api->status_string(rc));
  return register_handle(std::move(handle));
}

void dgrad_run(int64_t handle, const torch::Tensor& dy, const torch::Tensor& dx) {
  auto h = find_handle(handle);
  TORCH_CHECK(h->dgrad_plan != nullptr, "eqlinear: handle has no dgrad plan");
  check_contiguous_cuda(dy, "dy");
  check_contiguous_cuda(dx, "dx");
  check_dtype(dy, h->precision, "dy");
  check_dtype(dx, h->precision, "dx");
  TORCH_CHECK(dy.dim() == 3 && dx.dim() == 3, "eqlinear: dy/dx must be 3-D");
  TORCH_CHECK(dy.get_device() == h->device_index &&
                  dx.get_device() == h->device_index,
              "eqlinear: dy/dx must be on the plan device");
  check_alignment(dy, "dy");
  check_alignment(dx, "dx");
  const c10::cuda::CUDAGuard device_guard(dy.device());
  const int rc = h->api->dgrad_run(h->dgrad_plan, dy.data_ptr(), dx.data_ptr(),
                                   c10::cuda::getCurrentCUDAStream().stream());
  TORCH_CHECK(rc == 0, h->api->status_string(rc));
}

void dgrad_destroy(int64_t handle) { erase_handle(handle); }

std::tuple<int64_t, int64_t, int64_t, int64_t> dgrad_plan_info(int64_t handle) {
  auto h = find_handle(handle);
  TORCH_CHECK(h->dgrad_plan != nullptr, "eqlinear: handle has no dgrad plan");
  return {h->dgrad_info.precision, h->dgrad_info.dgrad_impl, h->dgrad_info.a_view,
          h->dgrad_info.j_rows};
}

torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& w,
                      const std::vector<int64_t>& i_dims, const std::vector<double>& cg,
                      int64_t precision) {
  return forward_impl(x, w, i_dims, cg, precision);
}

torch::Tensor dgrad(const torch::Tensor& dy, const torch::Tensor& w,
                    const std::vector<int64_t>& i_dims, const std::vector<double>& cg,
                    int64_t precision) {
  return dgrad_impl(dy, w, i_dims, cg, precision);
}

}  // namespace

TORCH_LIBRARY(eqlinear, m) {
  m.def("out_rows", &out_rows);
  m.def("forward", &forward);
  m.def("dgrad", &dgrad);
  m.def("forward_create", &forward_create);
  m.def("forward_run", &forward_run);
  m.def("forward_run_flat", &forward_run_flat);
  m.def("forward_destroy", &forward_destroy);
  m.def("forward_plan_info", &forward_plan_info);
  m.def("dgrad_create", &dgrad_create);
  m.def("dgrad_run", &dgrad_run);
  m.def("dgrad_destroy", &dgrad_destroy);
  m.def("dgrad_plan_info", &dgrad_plan_info);
}
