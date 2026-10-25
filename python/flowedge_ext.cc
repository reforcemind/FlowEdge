#include "api/engine.h"
#include "protocol/deadline_flow.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#if defined(__MINGW32__) && defined(__clang__)
namespace std {
__thread void* __once_callable = nullptr;
__thread void (*__once_call)() = nullptr;
} // namespace std
#endif

namespace py = pybind11;

namespace {

template<typename T> class TypedBuffer
{
public:
  explicit TypedBuffer(py::handle object, bool writable = false)
  {
    const int flags = PyBUF_FORMAT | PyBUF_C_CONTIGUOUS | (writable ? PyBUF_WRITABLE : 0);
    if (PyObject_GetBuffer(object.ptr(), &view_, flags) != 0)
      throw py::error_already_set{};
    const bool format_matches =
        view_.format != nullptr &&
        (std::is_same_v<T, float> ? std::strcmp(view_.format, "f") == 0
         : std::is_same_v<T, std::uint8_t>
             ? (std::strcmp(view_.format, "B") == 0 || std::strcmp(view_.format, "b") == 0)
             : (std::strcmp(view_.format, "i") == 0 || std::strcmp(view_.format, "l") == 0));
    if (view_.itemsize != static_cast<Py_ssize_t>(sizeof(T)) || !format_matches) {
      PyBuffer_Release(&view_);
      view_.obj = nullptr;
      throw std::runtime_error(std::is_same_v<T, float> ? "expected a C-contiguous float32 buffer"
                               : std::is_same_v<T, std::uint8_t>
                                   ? "expected a C-contiguous uint8 buffer"
                                   : "expected a C-contiguous int32 buffer");
    }
  }

  ~TypedBuffer()
  {
    if (view_.obj != nullptr)
      PyBuffer_Release(&view_);
  }

  TypedBuffer(const TypedBuffer&) = delete;
  TypedBuffer& operator=(const TypedBuffer&) = delete;

  [[nodiscard]] const T* data() const noexcept { return static_cast<const T*>(view_.buf); }
  [[nodiscard]] T* mutable_data() noexcept { return static_cast<T*>(view_.buf); }
  [[nodiscard]] std::size_t size() const noexcept
  {
    return static_cast<std::size_t>(view_.len / view_.itemsize);
  }

private:
  Py_buffer view_{};
};

using FloatBuffer = TypedBuffer<float>;
using Int32Buffer = TypedBuffer<std::int32_t>;
using UInt8Buffer = TypedBuffer<std::uint8_t>;

// Minimal legacy DLPack ABI. Calling __dlpack__ without max_version requests
// this stable capsule layout and lets the producer synchronize its work to
// CUDA's legacy default stream (stream=1).
enum class DLDeviceType : std::int32_t
{
  kCuda = 2,
};

struct DLDevice
{
  DLDeviceType device_type;
  std::int32_t device_id;
};

struct DLDataType
{
  std::uint8_t code;
  std::uint8_t bits;
  std::uint16_t lanes;
};

struct DLTensor
{
  void* data;
  DLDevice device;
  std::int32_t ndim;
  DLDataType dtype;
  std::int64_t* shape;
  std::int64_t* strides;
  std::uint64_t byte_offset;
};

struct DLManagedTensor
{
  DLTensor dl_tensor;
  void* manager_ctx;
  void (*deleter)(DLManagedTensor* self);
};

class DLPackCudaTensor
{
public:
  explicit DLPackCudaTensor(py::handle object)
  {
    if (!py::hasattr(object, "__dlpack__") || !py::hasattr(object, "__dlpack_device__"))
      throw std::runtime_error("expected an object implementing the DLPack protocol");
    const py::tuple device = py::cast<py::tuple>(object.attr("__dlpack_device__")());
    if (device.size() != 2 || py::cast<int>(device[0]) != static_cast<int>(DLDeviceType::kCuda))
      throw std::runtime_error("expected a CUDA DLPack tensor");
    capsule_ = object.attr("__dlpack__")(py::arg("stream") = 1);
    managed_ = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule_.ptr(), "dltensor"));
    if (managed_ == nullptr)
      throw py::error_already_set{};
    const DLTensor& tensor = managed_->dl_tensor;
    if (tensor.device.device_type != DLDeviceType::kCuda || tensor.device.device_id < 0 ||
        tensor.dtype.code != 2u || tensor.dtype.bits != 32u || tensor.dtype.lanes != 1u ||
        tensor.ndim < 0 || tensor.shape == nullptr || tensor.data == nullptr)
      throw std::runtime_error("expected a dense CUDA float32 DLPack tensor");
    device_id_ = tensor.device.device_id;
    std::size_t expected_stride = 1uz;
    for (std::int32_t dimension = tensor.ndim; dimension-- > 0;) {
      const std::int64_t extent = tensor.shape[dimension];
      if (extent < 0 || (tensor.strides != nullptr &&
                         tensor.strides[dimension] != static_cast<std::int64_t>(expected_stride)))
        throw std::runtime_error("expected a C-contiguous CUDA DLPack tensor");
      if (expected_stride != 0uz && static_cast<std::uint64_t>(extent) >
                                        std::numeric_limits<std::size_t>::max() / expected_stride)
        throw std::runtime_error("DLPack tensor size overflows size_t");
      expected_stride *= static_cast<std::size_t>(extent);
    }
    size_ = expected_stride;
    data_ = reinterpret_cast<float*>(static_cast<std::byte*>(tensor.data) + tensor.byte_offset);
  }

  ~DLPackCudaTensor()
  {
    if (managed_ == nullptr)
      return;
    DLManagedTensor* const owned = managed_;
    managed_ = nullptr;
    if (PyCapsule_SetName(capsule_.ptr(), "used_dltensor") != 0)
      PyErr_Clear();
    if (owned->deleter != nullptr)
      owned->deleter(owned);
  }

  DLPackCudaTensor(const DLPackCudaTensor&) = delete;
  DLPackCudaTensor& operator=(const DLPackCudaTensor&) = delete;

  [[nodiscard]] const float* data() const noexcept { return data_; }
  [[nodiscard]] float* mutable_data() noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] int device_id() const noexcept { return device_id_; }

private:
  py::object capsule_;
  DLManagedTensor* managed_{};
  float* data_{};
  std::size_t size_{};
  int device_id_{-1};
};

py::object float_array(py::handle numpy, std::size_t size)
{
  return numpy.attr("empty")(py::int_(size), py::str("float32"));
}

py::object float_matrix(py::handle numpy, std::size_t rows, std::size_t columns)
{
  return numpy.attr("empty")(py::make_tuple(py::int_(rows), py::int_(columns)), py::str("float32"));
}

py::object contiguous_array(py::handle numpy, py::handle object, const char* dtype)
{
  return numpy.attr("ascontiguousarray")(object, py::str(dtype));
}

int method_id(std::string_view method)
{
  if (method == "euler")
    return FE_SOLVER_EULER;
  if (method == "heun")
    return FE_SOLVER_HEUN;
  if (method == "rk4")
    return FE_SOLVER_RK4;
  throw std::runtime_error("method must be one of: euler, heun, rk4");
}

int diffusion_scheduler_id(std::string_view scheduler)
{
  if (scheduler == "ddim")
    return FE_DIFFUSION_DDIM;
  if (scheduler == "ddpm")
    return FE_DIFFUSION_DDPM;
  throw std::runtime_error("scheduler must be one of: ddim, ddpm");
}

int diffusion_precision_id(std::string_view precision)
{
  if (precision == "exact-f32")
    return FE_DIFFUSION_PRECISION_EXACT_F32;
  if (precision == "approx-fp16-deep")
    return FE_DIFFUSION_PRECISION_APPROX_FP16_DEEP;
  throw std::runtime_error("diffusion precision must be one of: exact-f32, approx-fp16-deep");
}

// Mamba + flow-matching action head
class Engine
{
public:
  explicit Engine(const std::string& path, std::optional<unsigned> threads,
                  std::string_view diffusion_precision)
      : engine_{threads ? fe_engine_load_with_threads_and_diffusion_precision(
                              path.c_str(), *threads, diffusion_precision_id(diffusion_precision))
                        : fe_engine_load_with_diffusion_precision(path.c_str(),
                                                                  diffusion_precision_id(
                                                                      diffusion_precision))},
        numpy_{py::module_::import("numpy")}
  {
    if (engine_ == nullptr)
      throw std::runtime_error("FlowEdge: cannot load " + path + ": " + fe_engine_last_error());
  }
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;
  ~Engine() { fe_engine_free(engine_); }

  [[nodiscard]] std::size_t action_dim() const { return fe_engine_action_dim(engine_); }
  [[nodiscard]] std::size_t action_horizon() const { return fe_engine_action_horizon(engine_); }
  [[nodiscard]] std::size_t condition_dim() const { return fe_engine_condition_dim(engine_); }
  [[nodiscard]] unsigned thread_count() const { return fe_engine_thread_count(engine_); }
  [[nodiscard]] bool cuda_resident() const { return fe_engine_cuda_resident(engine_) != 0; }
  [[nodiscard]] int diffusion_cuda_device() const
  {
    return fe_engine_diffusion_cuda_device(engine_);
  }
  [[nodiscard]] std::string_view diffusion_precision() const
  {
    return fe_engine_diffusion_precision(engine_) == FE_DIFFUSION_PRECISION_APPROX_FP16_DEEP
               ? "approx-fp16-deep"
               : "exact-f32";
  }
  [[nodiscard]] py::dict diffusion_metadata() const
  {
    fe_diffusion_metadata metadata{};
    if (fe_engine_diffusion_metadata(engine_, &metadata) != 0)
      throw std::runtime_error(fe_engine_last_error());
    py::dict result;
    result["protocol_version"] = metadata.protocol_version;
    result["action_dim"] = metadata.action_dim;
    result["condition_dim"] = metadata.condition_dim;
    result["horizon"] = metadata.horizon;
    result["action_steps"] = metadata.action_steps;
    result["observation_steps"] = metadata.observation_steps;
    result["train_timesteps"] = metadata.train_timesteps;
    result["clip_sample"] = metadata.clip_sample != 0u;
    result["clip_sample_range"] = metadata.clip_sample_range;
    return result;
  }
  [[nodiscard]] py::dict model_metadata() const
  {
    fe_model_metadata metadata{};
    if (fe_engine_model_metadata(engine_, &metadata) != 0)
      throw std::runtime_error(fe_engine_last_error());
    py::dict result;
    result["protocol_version"] = metadata.protocol_version;
    result["architecture"] = metadata.architecture;
    result["precision"] = metadata.precision;
    result["model_digest"] = digest_hex(metadata.model_digest);
    result["d_model"] = metadata.d_model;
    result["n_layers"] = metadata.n_layers;
    result["action_dim"] = metadata.action_dim;
    result["action_horizon"] = action_horizon();
    result["condition_dim"] = metadata.condition_dim;
    result["decode_snapshot_bytes"] = metadata.decode_snapshot_bytes;
    return result;
  }
  [[nodiscard]] std::size_t d_model() const
  {
    std::size_t dm{0uz};
    std::size_t nl{0uz};
    fe_engine_dims(engine_, &dm, &nl);
    return dm;
  }

  // tokens -> hidden states [seq_len, d_model]
  py::object run(py::handle tokens_object)
  {
    py::object tokens_array = contiguous_array(numpy_, tokens_object, "int32");
    const Int32Buffer tokens{tokens_array};
    py::object out = float_matrix(numpy_, tokens.size(), d_model());
    FloatBuffer output{out, true};
    run_native(tokens, output);
    return out;
  }

  void run_into(py::handle tokens_object, py::handle output_object)
  {
    const Int32Buffer tokens{tokens_object};
    FloatBuffer output{output_object, true};
    run_native(tokens, output);
  }

  // External VLM/proprioception encoder embeddings -> hidden states.
  py::object run_embeddings(py::handle embeddings_object, py::object attention_mask)
  {
    py::object embeddings_array = contiguous_array(numpy_, embeddings_object, "float32");
    const FloatBuffer embeddings{embeddings_array};
    const std::size_t width = d_model();
    if (width == 0uz || embeddings.size() == 0uz || embeddings.size() % width != 0uz)
      throw std::runtime_error("embeddings must contain one or more complete d_model rows");
    py::object out = float_matrix(numpy_, embeddings.size() / width, width);
    FloatBuffer output{out, true};
    if (attention_mask.is_none())
      run_embeddings_native(embeddings, output);
    else {
      py::object mask_array = contiguous_array(numpy_, attention_mask, "uint8");
      const UInt8Buffer mask{mask_array};
      run_embeddings_masked_native(embeddings, mask, output);
    }
    return out;
  }

  void run_embeddings_into(py::handle embeddings_object, py::handle output_object,
                           py::object attention_mask)
  {
    const FloatBuffer embeddings{embeddings_object};
    FloatBuffer output{output_object, true};
    if (attention_mask.is_none())
      run_embeddings_native(embeddings, output);
    else {
      py::object mask_array = contiguous_array(numpy_, attention_mask, "uint8");
      const UInt8Buffer mask{mask_array};
      run_embeddings_masked_native(embeddings, mask, output);
    }
  }

  py::object smolvla_embed_suffix(py::handle noisy_actions_object, float timestep)
  {
    py::object actions_array = contiguous_array(numpy_, noisy_actions_object, "float32");
    const FloatBuffer actions{actions_array};
    fe_model_metadata metadata{};
    if (fe_engine_model_metadata(engine_, &metadata) != 0)
      throw std::runtime_error(fe_engine_last_error());
    if (metadata.architecture != FE_ARCH_SMOLVLA_ACTION_EXPERT || metadata.action_dim == 0uz ||
        metadata.d_inner == 0uz || actions.size() == 0uz ||
        actions.size() % metadata.action_dim != 0uz)
      throw std::runtime_error(
          "expected a SmolVLA checkpoint and [chunk_size, max_action_dim] float32 actions");
    const std::size_t chunk_size = actions.size() / metadata.action_dim;
    py::object out = float_matrix(numpy_, chunk_size, metadata.d_inner);
    FloatBuffer output{out, true};
    int rc{0};
    {
      py::gil_scoped_release release;
      rc = fe_engine_smolvla_embed_suffix(engine_, actions.data(), chunk_size, timestep,
                                          output.mutable_data());
    }
    if (rc != 0)
      throw std::runtime_error(fe_engine_last_error());
    return out;
  }

  py::object smolvla_run_expert(py::handle suffix_object, py::handle prefix_keys_object,
                                py::handle prefix_values_object, py::handle prefix_mask_object)
  {
    py::object suffix_array = contiguous_array(numpy_, suffix_object, "float32");
    py::object keys_array = contiguous_array(numpy_, prefix_keys_object, "float32");
    py::object values_array = contiguous_array(numpy_, prefix_values_object, "float32");
    py::object mask_array = contiguous_array(numpy_, prefix_mask_object, "uint8");
    const FloatBuffer suffix{suffix_array};
    const FloatBuffer prefix_keys{keys_array};
    const FloatBuffer prefix_values{values_array};
    const UInt8Buffer prefix_mask{mask_array};
    fe_model_metadata metadata{};
    if (fe_engine_model_metadata(engine_, &metadata) != 0)
      throw std::runtime_error(fe_engine_last_error());
    const std::size_t expert_width = static_cast<std::size_t>(metadata.d_inner);
    const std::size_t expert_layers = static_cast<std::size_t>(metadata.n_layers);
    if (metadata.architecture != FE_ARCH_SMOLVLA_ACTION_EXPERT || expert_width == 0uz ||
        expert_layers == 0uz || suffix.size() == 0uz || suffix.size() % expert_width != 0uz ||
        prefix_mask.size() == 0uz || prefix_keys.size() != prefix_values.size() ||
        prefix_mask.size() > std::numeric_limits<std::size_t>::max() / expert_layers ||
        prefix_mask.size() * expert_layers > std::numeric_limits<std::size_t>::max() / 320uz ||
        prefix_keys.size() != prefix_mask.size() * expert_layers * 320uz)
      throw std::runtime_error(
          "expected SmolVLA suffix [chunk, expert_width], matching layer-major F32 VLM K/V "
          "caches, and a nonempty uint8 prefix mask");
    const std::size_t chunk_size = suffix.size() / expert_width;
    py::object out = float_matrix(numpy_, chunk_size, expert_width);
    FloatBuffer output{out, true};
    int rc{0};
    {
      py::gil_scoped_release release;
      rc = fe_engine_smolvla_run_expert(engine_, suffix.data(), chunk_size, prefix_keys.data(),
                                        prefix_values.data(), prefix_mask.data(),
                                        prefix_mask.size(), output.mutable_data());
    }
    if (rc != 0)
      throw std::runtime_error(fe_engine_last_error());
    return out;
  }

  py::object smolvla_project_actions(py::handle hidden_object)
  {
    py::object hidden_array = contiguous_array(numpy_, hidden_object, "float32");
    const FloatBuffer hidden{hidden_array};
    fe_model_metadata metadata{};
    if (fe_engine_model_metadata(engine_, &metadata) != 0)
      throw std::runtime_error(fe_engine_last_error());
    const std::size_t expert_width = static_cast<std::size_t>(metadata.d_inner);
    const std::size_t action_dim = static_cast<std::size_t>(metadata.action_dim);
    if (metadata.architecture != FE_ARCH_SMOLVLA_ACTION_EXPERT || expert_width == 0uz ||
        action_dim == 0uz || hidden.size() == 0uz || hidden.size() % expert_width != 0uz)
      throw std::runtime_error(
          "expected a SmolVLA checkpoint and [chunk_size, expert_width] float32 hidden states");
    const std::size_t chunk_size = hidden.size() / expert_width;
    py::object out = float_matrix(numpy_, chunk_size, action_dim);
    FloatBuffer output{out, true};
    int rc{0};
    {
      py::gil_scoped_release release;
      rc = fe_engine_smolvla_project_actions(engine_, hidden.data(), chunk_size,
                                             output.mutable_data());
    }
    if (rc != 0)
      throw std::runtime_error(fe_engine_last_error());
    return out;
  }

  py::object smolvla_denoise(py::handle noisy_actions_object, float timestep,
                             py::handle prefix_keys_object, py::handle prefix_values_object,
                             py::handle prefix_mask_object)
  {
    py::object actions_array = contiguous_array(numpy_, noisy_actions_object, "float32");
    py::object keys_array = contiguous_array(numpy_, prefix_keys_object, "float32");
    py::object values_array = contiguous_array(numpy_, prefix_values_object, "float32");
    py::object mask_array = contiguous_array(numpy_, prefix_mask_object, "uint8");
    const FloatBuffer actions{actions_array};
    const FloatBuffer prefix_keys{keys_array};
    const FloatBuffer prefix_values{values_array};
    const UInt8Buffer prefix_mask{mask_array};
    fe_model_metadata metadata{};
    if (fe_engine_model_metadata(engine_, &metadata) != 0)
      throw std::runtime_error(fe_engine_last_error());
    const std::size_t action_dim = static_cast<std::size_t>(metadata.action_dim);
    const std::size_t expert_layers = static_cast<std::size_t>(metadata.n_layers);
    if (metadata.architecture != FE_ARCH_SMOLVLA_ACTION_EXPERT || action_dim == 0uz ||
        expert_layers == 0uz || actions.size() == 0uz || actions.size() % action_dim != 0uz ||
        prefix_mask.size() == 0uz || prefix_keys.size() != prefix_values.size() ||
        prefix_mask.size() > std::numeric_limits<std::size_t>::max() / expert_layers ||
        prefix_mask.size() * expert_layers > std::numeric_limits<std::size_t>::max() / 320uz ||
        prefix_keys.size() != prefix_mask.size() * expert_layers * 320uz)
      throw std::runtime_error(
          "expected SmolVLA actions [chunk, max_action_dim], matching layer-major F32 VLM K/V "
          "caches, and a nonempty uint8 prefix mask");
    const std::size_t chunk_size = actions.size() / action_dim;
    py::object out = float_matrix(numpy_, chunk_size, action_dim);
    FloatBuffer output{out, true};
    int rc{0};
    {
      py::gil_scoped_release release;
      rc = fe_engine_smolvla_denoise(engine_, actions.data(), chunk_size, timestep,
                                     prefix_keys.data(), prefix_values.data(), prefix_mask.data(),
                                     prefix_mask.size(), output.mutable_data());
    }
    if (rc != 0)
      throw std::runtime_error(fe_engine_last_error());
    return out;
  }

  py::object smolvla_sample(py::handle initial_noise_object, py::handle prefix_keys_object,
                            py::handle prefix_values_object, py::handle prefix_mask_object,
                            std::size_t steps)
  {
    py::object noise_array = contiguous_array(numpy_, initial_noise_object, "float32");
    py::object keys_array = contiguous_array(numpy_, prefix_keys_object, "float32");
    py::object values_array = contiguous_array(numpy_, prefix_values_object, "float32");
    py::object mask_array = contiguous_array(numpy_, prefix_mask_object, "uint8");
    const FloatBuffer noise{noise_array};
    const FloatBuffer prefix_keys{keys_array};
    const FloatBuffer prefix_values{values_array};
    const UInt8Buffer prefix_mask{mask_array};
    const std::size_t action_dim =
        validate_smolvla_sample(noise, prefix_keys, prefix_values, prefix_mask, steps);
    const std::size_t chunk_size = noise.size() / action_dim;
    py::object out = float_matrix(numpy_, chunk_size, action_dim);
    FloatBuffer output{out, true};
    smolvla_sample_native(noise, prefix_keys, prefix_values, prefix_mask, chunk_size, steps,
                          output);
    return out;
  }

  void smolvla_sample_into(py::handle initial_noise_object, py::handle prefix_keys_object,
                           py::handle prefix_values_object, py::handle prefix_mask_object,
                           py::handle output_object, std::size_t steps)
  {
    py::object noise_array = contiguous_array(numpy_, initial_noise_object, "float32");
    py::object keys_array = contiguous_array(numpy_, prefix_keys_object, "float32");
    py::object values_array = contiguous_array(numpy_, prefix_values_object, "float32");
    py::object mask_array = contiguous_array(numpy_, prefix_mask_object, "uint8");
    const FloatBuffer noise{noise_array};
    const FloatBuffer prefix_keys{keys_array};
    const FloatBuffer prefix_values{values_array};
    const UInt8Buffer prefix_mask{mask_array};
    FloatBuffer output{output_object, true};
    const std::size_t action_dim =
        validate_smolvla_sample(noise, prefix_keys, prefix_values, prefix_mask, steps);
    const std::size_t chunk_size = noise.size() / action_dim;
    smolvla_sample_native(noise, prefix_keys, prefix_values, prefix_mask, chunk_size, steps,
                          output);
  }

  void run_embeddings_masked_native(const FloatBuffer& embeddings, const UInt8Buffer& mask,
                                    FloatBuffer& output)
  {
    const std::size_t width = d_model();
    if (width == 0uz || embeddings.size() == 0uz || embeddings.size() % width != 0uz ||
        mask.size() != embeddings.size() / width || output.size() != embeddings.size())
      throw std::runtime_error(
          "embedding, attention_mask, and output shapes must be [seq_len, d_model], [seq_len], "
          "and [seq_len, d_model]");
    int rc{0};
    {
      py::gil_scoped_release release;
      rc = fe_engine_run_embeddings_masked(engine_, embeddings.data(), mask.data(),
                                           embeddings.size() / width, output.mutable_data());
    }
    if (rc != 0)
      throw std::runtime_error(fe_engine_last_error());
  }

  py::object step(std::int32_t token)
  {
    py::object out = float_array(numpy_, d_model());
    FloatBuffer output{out, true};
    step_native(token, output);
    return out;
  }

  void step_into(std::int32_t token, py::handle output_object)
  {
    FloatBuffer output{output_object, true};
    step_native(token, output);
  }

  void step_native(std::int32_t token, FloatBuffer& output)
  {
    if (output.size() != d_model())
      throw std::runtime_error("output length must equal d_model");
    int rc{0};
    {
      py::gil_scoped_release release;
      rc = fe_engine_step(engine_, token, output.mutable_data());
    }
    if (rc != 0)
      throw std::runtime_error(fe_engine_last_error());
  }

  void reset() { fe_engine_reset(engine_); }

  py::bytes decode_state() const
  {
    const std::size_t size = fe_engine_decode_state_bytes(engine_);
    if (size > static_cast<std::size_t>(std::numeric_limits<Py_ssize_t>::max()))
      throw std::runtime_error("decode snapshot is too large for a Python bytes object");
    py::bytes snapshot = py::reinterpret_steal<py::bytes>(
        PyBytes_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(size)));
    if (!snapshot)
      throw py::error_already_set{};
    if (fe_engine_export_decode_state(engine_, PyBytes_AS_STRING(snapshot.ptr()), size) != 0)
      throw std::runtime_error(fe_engine_last_error());
    return snapshot;
  }

  void restore_decode_state(const py::bytes& snapshot)
  {
    char* data = nullptr;
    Py_ssize_t size = 0;
    if (PyBytes_AsStringAndSize(snapshot.ptr(), &data, &size) != 0)
      throw py::error_already_set{};
    if (fe_engine_import_decode_state(engine_, data, static_cast<std::size_t>(size)) != 0)
      throw std::runtime_error(fe_engine_last_error());
  }

  // prefix conditions the SSM
  // ODE noise -> action [action_dim]
  py::object sample(py::handle prefix_object, py::handle noise_object, std::size_t steps,
                    std::string_view method)
  {
    py::object prefix_array = contiguous_array(numpy_, prefix_object, "int32");
    py::object noise_array = contiguous_array(numpy_, noise_object, "float32");
    const Int32Buffer prefix{prefix_array};
    const FloatBuffer noise{noise_array};
    py::object action = float_array(numpy_, action_dim());
    FloatBuffer output{action, true};
    sample_native(prefix, noise, output, steps, method);
    return action;
  }

  void sample_into(py::handle prefix_object, py::handle noise_object, py::handle output_object,
                   std::size_t steps, std::string_view method)
  {
    const Int32Buffer prefix{prefix_object};
    const FloatBuffer noise{noise_object};
    FloatBuffer output{output_object, true};
    sample_native(prefix, noise, output, steps, method);
  }

  void sample_condition(py::handle condition_object, py::handle noise_object,
                        py::handle output_object, std::size_t steps, std::string_view method)
  {
    const FloatBuffer condition{condition_object};
    const FloatBuffer noise{noise_object};
    FloatBuffer action{output_object, true};
    validate_vectors(condition.size(), noise.size());
    validate_action_output(action.size());
    const int m = method_id(method);
    int rc{0};
    {
      py::gil_scoped_release release;
      rc = fe_engine_sample_condition(engine_, condition.data(), noise.data(), steps, m,
                                      action.mutable_data());
    }
    if (rc != 0)
      throw std::runtime_error(fe_engine_last_error());
  }

  py::object sample_diffusion(py::handle condition_object, py::handle noise_object,
                              std::size_t steps, std::string_view scheduler, std::uint64_t seed)
  {
    py::object condition_array = contiguous_array(numpy_, condition_object, "float32");
    py::object noise_array = contiguous_array(numpy_, noise_object, "float32");
    const FloatBuffer condition{condition_array};
    const FloatBuffer noise{noise_array};
    py::object result = float_matrix(numpy_, action_horizon(), action_dim());
    FloatBuffer action{result, true};
    sample_diffusion_native(condition, noise, action, steps, scheduler, seed);
    return result;
  }

  py::object diffusion_denoise(py::handle condition_object, py::handle sample_object,
                               float timestep)
  {
    py::object condition_array = contiguous_array(numpy_, condition_object, "float32");
    py::object sample_array = contiguous_array(numpy_, sample_object, "float32");
    const FloatBuffer condition{condition_array};
    const FloatBuffer sample{sample_array};
    py::object result = float_matrix(numpy_, action_horizon(), action_dim());
    FloatBuffer predicted_noise{result, true};
    const std::size_t values = action_horizon() * action_dim();
    if (condition.size() != condition_dim() || sample.size() != values)
      throw std::runtime_error("invalid Diffusion Policy condition or sample shape");
    int rc{0};
    {
      py::gil_scoped_release release;
      rc = fe_engine_diffusion_denoise(engine_, condition.data(), sample.data(), timestep,
                                       predicted_noise.mutable_data());
    }
    if (rc != 0)
      throw std::runtime_error(fe_engine_last_error());
    return result;
  }

  void sample_diffusion_into(py::handle condition_object, py::handle noise_object,
                             py::handle output_object, std::size_t steps,
                             std::string_view scheduler, std::uint64_t seed)
  {
    const FloatBuffer condition{condition_object};
    const FloatBuffer noise{noise_object};
    FloatBuffer action{output_object, true};
    sample_diffusion_native(condition, noise, action, steps, scheduler, seed);
  }

  void sample_diffusion_dlpack_into(py::handle condition_object, py::handle noise_object,
                                    py::handle output_object, std::size_t steps,
                                    std::string_view scheduler, std::uint64_t seed,
                                    std::size_t action_start)
  {
    DLPackCudaTensor condition{condition_object};
    DLPackCudaTensor noise{noise_object};
    DLPackCudaTensor action{output_object};
    const int engine_device = diffusion_cuda_device();
    if (engine_device < 0)
      throw std::runtime_error("CUDA tensor sampling requires a device-resident diffusion head");
    if (condition.device_id() != engine_device || noise.device_id() != engine_device ||
        action.device_id() != engine_device)
      throw std::runtime_error("DLPack tensors must use the diffusion head's CUDA device");
    const std::size_t values = action_horizon() * action_dim();
    if (condition.size() != condition_dim())
      throw std::runtime_error("condition length must equal condition_dim");
    if (noise.size() != values)
      throw std::runtime_error("noise size must equal action_horizon * action_dim");
    if (action_dim() == 0uz || action.size() == 0uz || action.size() % action_dim() != 0uz)
      throw std::runtime_error("output size must be a non-empty multiple of action_dim");
    const std::size_t action_steps = action.size() / action_dim();
    if (action_start > action_horizon() || action_steps > action_horizon() - action_start)
      throw std::runtime_error("output action slice is outside the diffusion horizon");
    const int scheduler_id = diffusion_scheduler_id(scheduler);
    int rc{0};
    {
      py::gil_scoped_release release;
      rc = fe_engine_sample_diffusion_device(engine_, condition.data(), noise.data(), steps,
                                             scheduler_id, seed, action_start, action_steps,
                                             action.mutable_data());
    }
    if (rc != FE_STATUS_OK)
      throw std::runtime_error(fe_engine_last_error());
  }

  void flow_begin(py::handle condition_object, py::handle noise_object, std::size_t steps,
                  std::string_view method, std::optional<std::uint64_t> generation,
                  std::uint64_t timestamp_ns, std::uint64_t deadline_ns)
  {
    const FloatBuffer condition{condition_object};
    const FloatBuffer noise{noise_object};
    validate_vectors(condition.size(), noise.size());
    const int m = method_id(method);
    int rc{0};
    {
      py::gil_scoped_release release;
      if (generation) {
        fe_condition_metadata metadata{};
        rc = fe_engine_make_condition_metadata(engine_, timestamp_ns, deadline_ns, *generation,
                                               steps, m, &metadata);
        if (rc == 0)
          rc = fe_engine_flow_begin_request(engine_, condition.data(), noise.data(), &metadata);
      } else {
        rc = fe_engine_flow_begin(engine_, condition.data(), noise.data(), steps, m);
      }
    }
    if (rc != 0)
      throw std::runtime_error(fe_engine_last_error());
  }

  void cancel_before(std::uint64_t generation) { fe_engine_cancel_before(engine_, generation); }

  [[nodiscard]] py::dict flow_metadata() const
  {
    fe_action_metadata metadata{};
    if (fe_engine_flow_action_metadata(engine_, &metadata) != 0)
      throw std::runtime_error(fe_engine_last_error());
    py::dict result;
    result["protocol_version"] = metadata.protocol_version;
    result["solver"] = metadata.solver;
    result["status"] = metadata.status;
    result["model_digest"] = digest_hex(metadata.model_digest);
    result["timestamp_ns"] = metadata.timestamp_ns;
    result["deadline_ns"] = metadata.deadline_ns;
    result["generation"] = metadata.generation;
    result["condition_dim"] = metadata.condition_dim;
    result["action_dim"] = metadata.action_dim;
    result["remaining_nfe"] = metadata.remaining_nfe;
    return result;
  }

  std::size_t flow_advance(py::handle output_object, std::size_t step_budget)
  {
    FloatBuffer action{output_object, true};
    validate_action_output(action.size());
    std::size_t remaining{0uz};
    int rc{0};
    {
      py::gil_scoped_release release;
      rc = fe_engine_flow_advance(engine_, step_budget, action.mutable_data(), &remaining);
    }
    if (rc != 0)
      throw std::runtime_error(fe_engine_last_error());
    return remaining;
  }

private:
  [[nodiscard]] std::size_t validate_smolvla_sample(const FloatBuffer& noise,
                                                    const FloatBuffer& prefix_keys,
                                                    const FloatBuffer& prefix_values,
                                                    const UInt8Buffer& prefix_mask,
                                                    std::size_t steps) const
  {
    fe_model_metadata metadata{};
    if (fe_engine_model_metadata(engine_, &metadata) != 0)
      throw std::runtime_error(fe_engine_last_error());
    const std::size_t action_dim = static_cast<std::size_t>(metadata.action_dim);
    const std::size_t expert_layers = static_cast<std::size_t>(metadata.n_layers);
    if (metadata.architecture != FE_ARCH_SMOLVLA_ACTION_EXPERT || action_dim == 0uz ||
        expert_layers == 0uz || steps == 0uz || steps > 100uz || noise.size() == 0uz ||
        noise.size() % action_dim != 0uz || prefix_mask.size() == 0uz ||
        prefix_keys.size() != prefix_values.size() ||
        prefix_mask.size() > std::numeric_limits<std::size_t>::max() / expert_layers ||
        prefix_mask.size() * expert_layers > std::numeric_limits<std::size_t>::max() / 320uz ||
        prefix_keys.size() != prefix_mask.size() * expert_layers * 320uz)
      throw std::runtime_error(
          "expected SmolVLA initial noise [chunk, max_action_dim], 1..100 Euler steps, matching "
          "layer-major F32 VLM K/V caches, and a nonempty uint8 prefix mask");
    return action_dim;
  }

  void smolvla_sample_native(const FloatBuffer& noise, const FloatBuffer& prefix_keys,
                             const FloatBuffer& prefix_values, const UInt8Buffer& prefix_mask,
                             std::size_t chunk_size, std::size_t steps, FloatBuffer& output)
  {
    if (output.size() != noise.size())
      throw std::runtime_error("output size must equal initial noise size");
    int rc{0};
    {
      py::gil_scoped_release release;
      rc = fe_engine_smolvla_sample(engine_, noise.data(), chunk_size, steps, prefix_keys.data(),
                                    prefix_values.data(), prefix_mask.data(), prefix_mask.size(),
                                    output.mutable_data());
    }
    if (rc != 0)
      throw std::runtime_error(fe_engine_last_error());
  }

  [[nodiscard]] static std::string digest_hex(const fe_model_digest& digest)
  {
    constexpr char alphabet[] = "0123456789abcdef";
    std::string result(2uz * FE_MODEL_DIGEST_BYTES, '0');
    for (std::size_t i{0uz}; i < FE_MODEL_DIGEST_BYTES; ++i) {
      result[2uz * i] = alphabet[digest.bytes[i] >> 4u];
      result[(2uz * i) + 1uz] = alphabet[digest.bytes[i] & 0x0fu];
    }
    return result;
  }

  void run_native(const Int32Buffer& tokens, FloatBuffer& output)
  {
    if (tokens.size() == 0uz)
      throw std::runtime_error("tokens must not be empty");
    const std::size_t model_width = d_model();
    if (model_width == 0uz ||
        tokens.size() > (std::numeric_limits<std::size_t>::max() / model_width) ||
        output.size() != tokens.size() * model_width)
      throw std::runtime_error("output size must equal token_count * d_model");
    int rc{0};
    {
      py::gil_scoped_release release;
      rc = fe_engine_run(engine_, tokens.data(), tokens.size(), output.mutable_data());
    }
    if (rc != 0)
      throw std::runtime_error(fe_engine_last_error());
  }

  void run_embeddings_native(const FloatBuffer& embeddings, FloatBuffer& output)
  {
    const std::size_t width = d_model();
    if (width == 0uz || embeddings.size() == 0uz || embeddings.size() % width != 0uz ||
        output.size() != embeddings.size())
      throw std::runtime_error("embedding input and output must be [seq_len, d_model]");
    int rc{0};
    {
      py::gil_scoped_release release;
      rc = fe_engine_run_embeddings(engine_, embeddings.data(), embeddings.size() / width,
                                    output.mutable_data());
    }
    if (rc != 0)
      throw std::runtime_error(fe_engine_last_error());
  }

  void sample_native(const Int32Buffer& prefix, const FloatBuffer& noise, FloatBuffer& output,
                     std::size_t steps, std::string_view method)
  {
    if (action_dim() == 0uz)
      throw std::runtime_error("checkpoint has no flow head");
    if (prefix.size() == 0uz)
      throw std::runtime_error("prefix must not be empty");
    if (noise.size() != action_dim())
      throw std::runtime_error("noise length must equal action_dim");
    validate_action_output(output.size());
    const int m = method_id(method);
    int rc{0};
    {
      py::gil_scoped_release release;
      rc = fe_engine_sample(engine_, prefix.data(), prefix.size(), noise.data(), steps, m,
                            output.mutable_data());
    }
    if (rc != 0)
      throw std::runtime_error(fe_engine_last_error());
  }

  void validate_vectors(std::size_t condition_size, std::size_t noise_size) const
  {
    if (condition_size != condition_dim())
      throw std::runtime_error("condition length must equal condition_dim");
    if (noise_size != action_dim())
      throw std::runtime_error("noise length must equal action_dim");
  }

  void validate_action_output(std::size_t output_size) const
  {
    if (output_size != action_dim())
      throw std::runtime_error("output length must equal action_dim");
  }

  void sample_diffusion_native(const FloatBuffer& condition, const FloatBuffer& noise,
                               FloatBuffer& action, std::size_t steps, std::string_view scheduler,
                               std::uint64_t seed)
  {
    if (action_horizon() <= 1uz)
      throw std::runtime_error("checkpoint has no Diffusion Policy head");
    if (condition.size() != condition_dim())
      throw std::runtime_error("condition length must equal condition_dim");
    const std::size_t values = action_horizon() * action_dim();
    if (noise.size() != values)
      throw std::runtime_error("noise size must equal action_horizon * action_dim");
    if (action.size() != values)
      throw std::runtime_error("output size must equal action_horizon * action_dim");
    const int scheduler_id = diffusion_scheduler_id(scheduler);
    int rc{0};
    {
      py::gil_scoped_release release;
      rc = fe_engine_sample_diffusion(engine_, condition.data(), noise.data(), steps, scheduler_id,
                                      seed, action.mutable_data());
    }
    if (rc != 0)
      throw std::runtime_error(fe_engine_last_error());
  }

  fe_engine* engine_;
  py::object numpy_;
};

} // namespace

PYBIND11_MODULE(flowedge, m)
{
#if defined(__clang__)
  m.attr("compiler") = "Clang " __clang_version__;
#elif defined(__GNUC__)
  m.attr("compiler") = "GCC " __VERSION__;
#elif defined(_MSC_VER)
  m.attr("compiler") = std::string{"MSVC "} + std::to_string(_MSC_VER);
#else
  m.attr("compiler") = "unknown";
#endif
#if defined(NDEBUG)
  m.attr("build_type") = "Release";
#else
  m.attr("build_type") = "Debug";
#endif
#ifdef FLOWEDGE_CUDA
  m.attr("cuda") = true;
#else
  m.attr("cuda") = false;
#endif
  py::class_<fe::ExecutionPlan>(m, "ExecutionPlan")
      .def(py::init<>())
      .def_readwrite("id", &fe::ExecutionPlan::id)
      .def_readwrite("steps", &fe::ExecutionPlan::steps)
      .def_readwrite("solver", &fe::ExecutionPlan::solver)
      .def_readwrite("precision", &fe::ExecutionPlan::precision)
      .def_readwrite("trace_id", &fe::ExecutionPlan::trace_id)
      .def_readwrite("hardware_mask", &fe::ExecutionPlan::hardware_mask)
      .def_readwrite("latency_ns", &fe::ExecutionPlan::latency_ns)
      .def_readwrite("memory_bytes", &fe::ExecutionPlan::memory_bytes)
      .def_readwrite("quality", &fe::ExecutionPlan::quality);
  py::class_<fe::PlanContext>(m, "PlanContext")
      .def(py::init<>())
      .def_readwrite("slack_ns", &fe::PlanContext::slack_ns)
      .def_readwrite("buffered_actions", &fe::PlanContext::buffered_actions)
      .def_readwrite("period_ns", &fe::PlanContext::period_ns)
      .def_readwrite("reserve_ns", &fe::PlanContext::reserve_ns)
      .def_readwrite("switch_ns", &fe::PlanContext::switch_ns)
      .def_readwrite("available_memory_bytes", &fe::PlanContext::available_memory_bytes)
      .def_readwrite("hardware_mask", &fe::PlanContext::hardware_mask)
      .def_readwrite("minimum_quality", &fe::PlanContext::minimum_quality)
      .def_readwrite("healthy", &fe::PlanContext::healthy);
  py::class_<fe::DeadlineFlow>(m, "DeadlineFlow")
      .def(py::init([](const std::vector<fe::ExecutionPlan>& plans) {
        auto selector = fe::DeadlineFlow::create(plans);
        if (!selector)
          throw std::invalid_argument("expected 1-32 unique, calibrated execution plans");
        return *selector;
      }))
      .def("select", &fe::DeadlineFlow::select)
      .def("observe", &fe::DeadlineFlow::observe);
  m.doc() = "FlowEdge: flow-matching action-head inference";

  py::class_<Engine>(m, "Engine")
      .def(py::init<const std::string&, std::optional<unsigned>, std::string_view>(),
           py::arg("path"), py::arg("threads") = py::none(),
           py::arg("diffusion_precision") = "exact-f32")
      .def_property_readonly("action_dim", &Engine::action_dim)
      .def_property_readonly("action_horizon", &Engine::action_horizon)
      .def_property_readonly("condition_dim", &Engine::condition_dim)
      .def_property_readonly("d_model", &Engine::d_model)
      .def_property_readonly("thread_count", &Engine::thread_count)
      .def_property_readonly("cuda_resident", &Engine::cuda_resident)
      .def_property_readonly("diffusion_cuda_device", &Engine::diffusion_cuda_device)
      .def_property_readonly("diffusion_precision", &Engine::diffusion_precision)
      .def_property_readonly("model_metadata", &Engine::model_metadata)
      .def_property_readonly("diffusion_metadata", &Engine::diffusion_metadata)
      .def("run", &Engine::run, py::arg("tokens"))
      .def("run_into", &Engine::run_into, py::arg("tokens"), py::arg("output"))
      .def("run_embeddings", &Engine::run_embeddings, py::arg("embeddings"),
           py::arg("attention_mask") = py::none())
      .def("run_embeddings_into", &Engine::run_embeddings_into, py::arg("embeddings"),
           py::arg("output"), py::arg("attention_mask") = py::none())
      .def("smolvla_embed_suffix", &Engine::smolvla_embed_suffix,
           "embed a padded SmolVLA action chunk before the external expert", py::arg("actions"),
           py::arg("timestep"))
      .def("smolvla_run_expert", &Engine::smolvla_run_expert,
           "run the SmolVLA action expert from a captured VLM K/V cache", py::arg("suffix"),
           py::arg("prefix_keys"), py::arg("prefix_values"), py::arg("prefix_mask"))
      .def("smolvla_project_actions", &Engine::smolvla_project_actions,
           "project SmolVLA expert hidden states into padded action coordinates", py::arg("hidden"))
      .def("smolvla_denoise", &Engine::smolvla_denoise,
           "run one complete SmolVLA action-expert denoise step from a captured VLM K/V cache",
           py::arg("actions"), py::arg("timestep"), py::arg("prefix_keys"),
           py::arg("prefix_values"), py::arg("prefix_mask"))
      .def("smolvla_sample", &Engine::smolvla_sample,
           "sample a SmolVLA action chunk with Euler from a captured VLM K/V cache",
           py::arg("initial_noise"), py::arg("prefix_keys"), py::arg("prefix_values"),
           py::arg("prefix_mask"), py::arg("steps") = 10uz)
      .def("smolvla_sample_into", &Engine::smolvla_sample_into,
           "sample a SmolVLA action chunk into caller-owned contiguous float32 storage",
           py::arg("initial_noise"), py::arg("prefix_keys"), py::arg("prefix_values"),
           py::arg("prefix_mask"), py::arg("output"), py::arg("steps") = 10uz)
      .def("step", &Engine::step, py::arg("token"))
      .def("step_into", &Engine::step_into, py::arg("token"), py::arg("output"))
      .def("reset", &Engine::reset)
      .def("decode_state", &Engine::decode_state, "snapshot the streaming SSM state")
      .def("restore_decode_state", &Engine::restore_decode_state, "restore a streaming SSM state",
           py::arg("snapshot"))
      .def("sample", &Engine::sample, "sample action trajectory", py::arg("prefix"),
           py::arg("noise"), py::arg("steps") = 10uz, py::arg("method") = "euler")
      .def("sample_into", &Engine::sample_into, "sample into caller-owned storage",
           py::arg("prefix"), py::arg("noise"), py::arg("output"), py::arg("steps") = 10uz,
           py::arg("method") = "euler")
      .def("sample_condition", &Engine::sample_condition,
           "sample from an external condition vector", py::arg("condition"), py::arg("noise"),
           py::arg("output"), py::arg("steps") = 10uz, py::arg("method") = "euler")
      .def("sample_diffusion", &Engine::sample_diffusion,
           "sample an un-normalized Diffusion Policy action horizon", py::arg("condition"),
           py::arg("noise"), py::arg("steps") = 10uz, py::arg("scheduler") = "ddim",
           py::arg("seed") = 0u)
      .def("diffusion_denoise", &Engine::diffusion_denoise,
           "run one normalized ConditionalUnet1D epsilon prediction", py::arg("condition"),
           py::arg("sample"), py::arg("timestep"))
      .def("sample_diffusion_into", &Engine::sample_diffusion_into,
           "sample Diffusion Policy into caller-owned storage", py::arg("condition"),
           py::arg("noise"), py::arg("output"), py::arg("steps") = 10uz,
           py::arg("scheduler") = "ddim", py::arg("seed") = 0u)
      .def("sample_diffusion_dlpack_into", &Engine::sample_diffusion_dlpack_into,
           "sample a Diffusion Policy action slice between CUDA DLPack tensors",
           py::arg("condition"), py::arg("noise"), py::arg("output"), py::arg("steps") = 10uz,
           py::arg("scheduler") = "ddim", py::arg("seed") = 0u, py::arg("action_start") = 0uz)
      .def("flow_begin", &Engine::flow_begin, "begin a resumable flow solve", py::arg("condition"),
           py::arg("noise"), py::arg("steps") = 10uz, py::arg("method") = "euler",
           py::arg("generation") = py::none(), py::arg("timestamp_ns") = 0u,
           py::arg("deadline_ns") = 0u)
      .def("flow_advance", &Engine::flow_advance, "advance a resumable flow solve",
           py::arg("output"), py::arg("step_budget") = 1uz)
      .def("cancel_before", &Engine::cancel_before, py::arg("generation"),
           "cancel an active request from an older generation")
      .def_property_readonly("flow_metadata", &Engine::flow_metadata);
}
