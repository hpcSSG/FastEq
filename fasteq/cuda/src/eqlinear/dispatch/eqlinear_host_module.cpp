// Optional Python initialization in the SAME _cuda DSO. The parent adds this
// C++ source only to _cuda; Python can then import fasteq.cuda._cuda lazily.
#include <pybind11/pybind11.h>

void fasteq_bind_eqlinear_host(pybind11::module_&);

PYBIND11_MODULE(_cuda, module) {
  fasteq_bind_eqlinear_host(module);
}
