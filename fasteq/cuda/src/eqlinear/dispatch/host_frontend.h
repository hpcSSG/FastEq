#pragma once

// Included inside the dispatcher's private namespace, after its host helpers.
// The callable is only an LRU value: it keeps a weak plan and weak caller refs.
class SinglePathFrontend {
 public:
  SinglePathFrontend(const std::shared_ptr<Handle>& h, pybind11::handle descriptor,
                     pybind11::dict snapshot, pybind11::handle weight,
                     pybind11::object operand_type, pybind11::object path_type,
                     pybind11::object array_type)
      : plan_(h), descriptor_(descriptor), weight_(weight),
        tensor_type_(pybind11::type::of(weight)), descriptor_type_(pybind11::type::of(descriptor)),
        operand_type_(std::move(operand_type)), path_type_(std::move(path_type)),
        array_type_(std::move(array_type)),
        operands_(snapshot["operands"]), paths_(snapshot["paths"]),
        dimensions_(snapshot["dimensions"]), subscripts_(snapshot["coefficient_subscripts"]),
        indices_(snapshot["path_indices"].cast<pybind11::tuple>()[0]),
        forward_ad_(pybind11::module_::import("torch.autograd.forward_ad")),
        transforms_active_(pybind11::module_::import("torch").attr("_C").attr("_are_functorch_transforms_active")) {
    namespace py = pybind11;
    const auto& w = THPVariable_Unpack(weight.ptr());
    sizes_ = w.sizes().vec();
    strides_ = w.strides().vec();
    if (h->problem.P != 1 || !PyTuple_CheckExact(paths_.ptr()) || PyTuple_GET_SIZE(paths_.ptr()) != 1)
      return;
    auto states = snapshot["dimension_state"].cast<py::tuple>();
    auto segments = snapshot["segments"].cast<py::tuple>();
    for (size_t q = 0; q < 3; ++q) {
      dimension_sets_[q] = py::set(states[q]);
      segments_[q] = segments[q];
    }
    auto path = py::reinterpret_borrow<py::object>(PyTuple_GET_ITEM(paths_.ptr(), 0));
    auto coefficient = path.attr("coefficients");
    if (Py_TYPE(coefficient.ptr()) != reinterpret_cast<PyTypeObject*>(array_type_.ptr())) return;
    auto array = py::reinterpret_borrow<py::array>(coefficient);
    if (array.ndim() != 0 || array.nbytes() > 16 || !PyTuple_CheckExact(indices_.ptr())) return;
    dtype_ = array.dtype();
    const auto* bytes = static_cast<const uint8_t*>(array.data());
    coefficient_.assign(bytes, bytes + array.nbytes());
    eligible_ = true;
  }

  bool eligible() const { return eligible_; }

  pybind11::object call(pybind11::handle descriptor, pybind11::handle weight,
                        pybind11::handle input, pybind11::handle precision) const {
    namespace py = pybind11;
    if (Py_TYPE(input.ptr()) != reinterpret_cast<PyTypeObject*>(tensor_type_.ptr()) ||
        Py_TYPE(weight.ptr()) != reinterpret_cast<PyTypeObject*>(tensor_type_.ptr()) ||
        Py_TYPE(descriptor.ptr()) != reinterpret_cast<PyTypeObject*>(descriptor_type_.ptr()) ||
        descriptor_().ptr() != descriptor.ptr() || weight_().ptr() != weight.ptr()) return py::none();
    if (transforms_active_().cast<bool>() || at::impl::torch_function_mode_enabled() ||
        c10::impl::TorchDispatchModeTLS::any_modes_set() ||
        forward_ad_.attr("_current_level").cast<int64_t>() >= 0) return py::none();
    auto h = plan_.lock();
    if (!h || !h->registered.load(std::memory_order_acquire)) return py::none();
    const auto& x = THPVariable_Unpack(input.ptr());
    const auto& w = THPVariable_Unpack(weight.ptr());
    if (!x.is_cuda() || !w.is_cuda() || x.get_device() != h->device_index ||
        w.get_device() != h->device_index || !x.is_contiguous() || !w.is_contiguous() ||
        x.scalar_type() != w.scalar_type() || w.is_inference() ||
        w.scalar_type() != (h->precision == 2 ? at::kDouble : at::kFloat) ||
        w.sizes() != at::IntArrayRef(sizes_) || w.strides() != at::IntArrayRef(strides_) ||
        w.data_ptr() != h->weight_data || w._version() != h->weight_version ||
        x.dim() != 2 || x.size(0) != h->batch || x.size(1) != h->input_rows * h->u ||
        c10::cuda::getCurrentCUDAStream(h->device_index).stream() != h->creation_stream ||
        !precision_matches(precision, h->precision)) return py::none();
    auto desc = py::reinterpret_borrow<py::object>(descriptor);
    auto operands = desc.attr("operands_and_subscripts");
    auto paths = desc.attr("paths");
    if (operands.ptr() != operands_.ptr() || paths.ptr() != paths_.ptr() ||
        !equal(desc.attr("coefficient_subscripts"), subscripts_)) return py::none();
    for (size_t q = 0; q < 3; ++q) {
      auto pair = PyTuple_GET_ITEM(operands.ptr(), q);
      auto operand = py::reinterpret_borrow<py::object>(PyTuple_GET_ITEM(pair, 0));
      if (Py_TYPE(operand.ptr()) != reinterpret_cast<PyTypeObject*>(operand_type_.ptr()) ||
          operand.attr("segments").ptr() != segments_[q].ptr()) return py::none();
    }
    auto path = py::reinterpret_borrow<py::object>(PyTuple_GET_ITEM(paths.ptr(), 0));
    if (Py_TYPE(path.ptr()) != reinterpret_cast<PyTypeObject*>(path_type_.ptr()) ||
        !equal(path.attr("indices"), indices_)) return py::none();
    auto coefficient = path.attr("coefficients");
    if (Py_TYPE(coefficient.ptr()) != reinterpret_cast<PyTypeObject*>(array_type_.ptr())) return py::none();
    auto array = py::reinterpret_borrow<py::array>(coefficient);
    if (array.ndim() != 0 || array.nbytes() != static_cast<py::ssize_t>(coefficient_.size()) ||
        !equal(array.dtype(), dtype_) || std::memcmp(array.data(), coefficient_.data(), coefficient_.size()))
      return py::none();
    // Must call the live method: functools.cache.cache_clear() or replacement
    // dictionaries cannot be hidden by a cached bound method/dict reference.
    py::object dimensions = py::reinterpret_steal<py::object>(
        PyObject_CallMethodNoArgs(descriptor.ptr(), dimension_method_.ptr()));
    if (!dimensions) throw py::error_already_set();
    if (dimensions.ptr() != dimensions_.ptr() || !PyDict_CheckExact(dimensions.ptr())) return py::none();
    for (size_t q = 0; q < 3; ++q) {
      auto values = PyDict_GetItemWithError(dimensions.ptr(), names_[q].ptr());
      if (!values || !PySet_CheckExact(values)) return py::none();
      Py_ssize_t pos = 0;
      PyObject* value;
      Py_hash_t hash;
      while (_PySet_NextEntry(values, &pos, &value, &hash))
        if (!PyLong_CheckExact(value)) return py::none();
      if (!equal(py::handle(values), dimension_sets_[q])) return py::none();
    }
    torch::Tensor result;
    {
      py::gil_scoped_release release;
      result = run_forward_autograd(h, x, w);
    }
    return py::cast(std::move(result));
  }

 private:
  static bool equal(pybind11::handle a, pybind11::handle b) {
    const int result = PyObject_RichCompareBool(a.ptr(), b.ptr(), Py_EQ);
    if (result < 0) throw pybind11::error_already_set();
    return result == 1;
  }

  static bool precision_matches(pybind11::handle argument, int64_t expected) {
    if (expected == 2) return true;  // FP64 ignores the optional precision argument.
    if (argument.is_none()) return expected == 0;
    if (!PyUnicode_CheckExact(argument.ptr())) return false;
    const auto length = PyUnicode_GET_LENGTH(argument.ptr());
    if (length != 4 && length != 6) return false;
    char value[7] = {};
    for (Py_ssize_t q = 0; q < length; ++q) {
      auto c = PyUnicode_READ_CHAR(argument.ptr(), q);
      if (c > 127) return false;
      value[q] = static_cast<char>(c >= 'A' && c <= 'Z' ? c + 32 : c);
    }
    return expected == 1 ? std::strcmp(value, "fp32") == 0
                         : std::strcmp(value, "tf32") == 0 || std::strcmp(value, "tf32x3") == 0;
  }

  std::weak_ptr<Handle> plan_;
  pybind11::weakref descriptor_, weight_;
  pybind11::object tensor_type_, descriptor_type_, operand_type_, path_type_, array_type_;
  pybind11::object operands_, paths_, dimensions_, subscripts_, indices_, dtype_, forward_ad_, transforms_active_;
  std::array<pybind11::object, 3> dimension_sets_, segments_;
  std::array<pybind11::str, 3> names_{pybind11::str("i"), pybind11::str("u"), pybind11::str("v")};
  pybind11::str dimension_method_{"get_dimensions_dict"};
  std::vector<uint8_t> coefficient_;
  std::vector<int64_t> sizes_, strides_;
  bool eligible_ = false;
};
