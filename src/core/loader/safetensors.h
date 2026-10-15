#pragma once

#include "../arena/arena.h"
#include "../protocol/deployment_profile.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace fe {

struct TensorView
{
  enum class Dtype : std::uint8_t
  {
    F32,
    BF16,
    INT8,
    Unsupported
  };

  const void* data{};
  Dtype dtype{};
  std::array<std::size_t, 4> shape{};
  std::array<char, 128> name{}; // null-terminated; supports long policy keys
  std::size_t bytes{};
  std::uint8_t ndim{};

  [[nodiscard]] std::string_view name_view() const noexcept { return {name.data()}; }

  [[nodiscard]] bool is_f32() const noexcept { return dtype == Dtype::F32; }
  [[nodiscard]] bool is_bf16() const noexcept { return dtype == Dtype::BF16; }

  [[nodiscard]] const float* as_f32() const noexcept
  {
    return is_f32() ? static_cast<const float*>(data) : nullptr;
  }
  [[nodiscard]] const uint16_t* as_bf16() const noexcept
  {
    return is_bf16() ? static_cast<const uint16_t*>(data) : nullptr;
  }
  [[nodiscard]] const uint8_t* as_i8() const noexcept
  {
    return (dtype == Dtype::INT8) ? static_cast<const uint8_t*>(data) : nullptr;
  }
};

struct WeightView
{
  const void* data{};
  TensorView::Dtype dtype{};
};

struct TensorMetadata
{
  TensorView::Dtype dtype{TensorView::Dtype::Unsupported};
  std::array<std::size_t, 4> shape{};
  std::array<char, 128> name{};
  std::size_t bytes{};
  std::uint8_t ndim{};

  [[nodiscard]] std::string_view name_view() const noexcept { return {name.data()}; }
  [[nodiscard]] bool supported() const noexcept
  {
    return dtype == TensorView::Dtype::F32 || dtype == TensorView::Dtype::BF16;
  }
};

// First tensor whose name matches, or nullptr.
[[nodiscard]] inline const TensorView* find_tensor(std::span<const TensorView> ts,
                                                   std::string_view name) noexcept
{
  for (const auto& t : ts)
    if (t.name_view() == name)
      return &t;
  return nullptr;
}

// Parse an F32/BF16 .safetensors file; weights stored as-is
[[nodiscard]] std::expected<void, const char*> load_safetensors(
    std::string_view path, Arena& arena, std::span<TensorView> out,
    std::size_t& tensors_loaded) noexcept;

// returns actual stored bytes
[[nodiscard]] std::size_t safetensors_weight_bytes(std::string_view path) noexcept;

// shape of a named tensor from the header, for pre-load arena sizing
[[nodiscard]] std::array<std::size_t, 4> safetensors_tensor_shape(std::string_view path,
                                                                  std::string_view name) noexcept;

// Read one string value from the safetensors __metadata__ map.
[[nodiscard]] std::expected<std::optional<std::string>, const char*> safetensors_metadata_value(
    std::string_view path, std::string_view key) noexcept;

// Inspect tensor headers without copying weight data or constructing a runtime.
[[nodiscard]] std::expected<void, const char*> inspect_safetensors(
    std::string_view path, std::span<TensorMetadata> out, std::size_t& tensors_loaded,
    std::size_t& total_bytes, std::size_t& unsupported_tensors) noexcept;

// One immutable, reference-counted checkpoint store. Engine instances retain a
// shared reference while keeping all scratch, sampler, and decode state private.
class ModelWeights
{
public:
  [[nodiscard]] static std::expected<std::shared_ptr<const ModelWeights>, const char*> open(
      std::string_view path) noexcept;

  ModelWeights(const ModelWeights&) = delete;
  ModelWeights& operator=(const ModelWeights&) = delete;
  ModelWeights(ModelWeights&&) = delete;
  ModelWeights& operator=(ModelWeights&&) = delete;
  ~ModelWeights() = default;

  [[nodiscard]] std::span<const TensorView> tensors() const noexcept
  {
    return {views_.data(), tensor_count_};
  }
  [[nodiscard]] std::size_t weight_bytes() const noexcept { return weight_bytes_; }
  [[nodiscard]] std::size_t mapped_file_bytes() const noexcept { return mapping_bytes_; }

  [[nodiscard]] const DeploymentProfile* deployment_profile() const noexcept
  {
    return deployment_profile_ ? &*deployment_profile_ : nullptr;
  }

private:
  static constexpr std::size_t kMaxTensors = 1024uz;

  ModelWeights(std::shared_ptr<const std::byte> mapping, std::size_t mapping_bytes,
               std::optional<DeploymentProfile> deployment_profile);

  std::shared_ptr<const std::byte> mapping_{};
  std::size_t mapping_bytes_{};
  std::array<TensorView, kMaxTensors> views_{};
  std::size_t tensor_count_{};
  std::size_t weight_bytes_{};
  std::optional<DeploymentProfile> deployment_profile_{};
};

} // namespace fe
