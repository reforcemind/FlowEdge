#include "loader/safetensors.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace {

class TemporarySafetensors
{
public:
  TemporarySafetensors(std::string_view json, std::size_t data_bytes,
                       std::uint64_t header_length = std::numeric_limits<std::uint64_t>::max())
      : path_(std::filesystem::temp_directory_path() /
              ("flowedge-safetensors-" + std::to_string(next_id()) + ".safetensors"))
  {
    if (header_length == std::numeric_limits<std::uint64_t>::max())
      header_length = json.size();
    std::vector<std::byte> bytes(8uz + json.size() + data_bytes);
    for (std::size_t index{}; index < sizeof(header_length); ++index)
      bytes[index] = static_cast<std::byte>((header_length >> (index * 8uz)) & 0xffu);
    std::copy(json.begin(), json.end(), reinterpret_cast<char*>(bytes.data() + 8uz));
    std::ofstream file(path_, std::ios::binary);
    file.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  }

  explicit TemporarySafetensors(std::size_t truncated_bytes)
      : path_(std::filesystem::temp_directory_path() /
              ("flowedge-safetensors-" + std::to_string(next_id()) + ".safetensors"))
  {
    std::vector<std::byte> bytes(truncated_bytes);
    std::ofstream file(path_, std::ios::binary);
    file.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  }

  ~TemporarySafetensors()
  {
    std::error_code ignored;
    std::filesystem::remove(path_, ignored);
  }

  TemporarySafetensors(const TemporarySafetensors&) = delete;
  TemporarySafetensors& operator=(const TemporarySafetensors&) = delete;

  [[nodiscard]] std::string string() const { return path_.string(); }

private:
  static std::uint64_t next_id() noexcept
  {
    static std::atomic_uint64_t id{0u};
    return id.fetch_add(1u, std::memory_order_relaxed);
  }

  std::filesystem::path path_;
};

TEST(SafetensorsHardening, RejectsTruncatedAndOversizedHeaders)
{
  const TemporarySafetensors truncated{7uz};
  EXPECT_EQ(fe::safetensors_weight_bytes(truncated.string()), 0uz);
  EXPECT_FALSE(fe::ModelWeights::open(truncated.string()));

  const TemporarySafetensors oversized{"{}", 0uz, 100uz};
  EXPECT_EQ(fe::safetensors_weight_bytes(oversized.string()), 0uz);
  EXPECT_FALSE(fe::ModelWeights::open(oversized.string()));
}

TEST(SafetensorsHardening, RejectsReversedAndOverlappingTensorSpans)
{
  const TemporarySafetensors reversed{
      R"({"weight":{"dtype":"F32","shape":[2],"data_offsets":[8,0]}})", 8uz};
  EXPECT_EQ(fe::safetensors_weight_bytes(reversed.string()), 0uz);

  const TemporarySafetensors overlapping{
      R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}})",
      8uz};
  EXPECT_EQ(fe::safetensors_weight_bytes(overlapping.string()), 0uz);
  EXPECT_FALSE(fe::ModelWeights::open(overlapping.string()));
}

TEST(SafetensorsHardening, RejectsUnsupportedRankAndShapeByteMismatch)
{
  const TemporarySafetensors rank{
      R"({"weight":{"dtype":"F32","shape":[1,2,3,4,5],"data_offsets":[0,8]}})", 8uz};
  EXPECT_EQ(fe::safetensors_weight_bytes(rank.string()), 0uz);

  const TemporarySafetensors mismatch{
      R"({"weight":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}})", 8uz};
  EXPECT_EQ(fe::safetensors_weight_bytes(mismatch.string()), 0uz);
}

TEST(SafetensorsHardening, InspectionRejectsOutOfBoundsAndOverlappingSpans)
{
  const TemporarySafetensors out_of_bounds{
      R"({"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,16]}})", 8uz};
  std::array<fe::TensorMetadata, 2> metadata{};
  std::size_t loaded{}, total{}, unsupported{};
  EXPECT_FALSE(
      fe::inspect_safetensors(out_of_bounds.string(), metadata, loaded, total, unsupported));

  const TemporarySafetensors overlapping{
      R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}})",
      8uz};
  EXPECT_FALSE(fe::inspect_safetensors(overlapping.string(), metadata, loaded, total, unsupported));
}

TEST(SafetensorsHardening, InspectionRejectsShapeByteMismatch)
{
  const TemporarySafetensors mismatch{
      R"({"weight":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}})", 8uz};
  std::array<fe::TensorMetadata, 1> metadata{};
  std::size_t loaded{}, total{}, unsupported{};
  EXPECT_FALSE(fe::inspect_safetensors(mismatch.string(), metadata, loaded, total, unsupported));
}

TEST(SafetensorsHardening, InspectionReportsUnsupportedDtypes)
{
  const TemporarySafetensors unsupported{
      R"({"weight":{"dtype":"I64","shape":[1],"data_offsets":[0,8]}})", 8uz};
  std::array<fe::TensorMetadata, 1> metadata{};
  std::size_t loaded{}, total{}, unsupported_count{};
  ASSERT_TRUE(
      fe::inspect_safetensors(unsupported.string(), metadata, loaded, total, unsupported_count));
  EXPECT_EQ(loaded, 0uz);
  EXPECT_EQ(total, 0uz);
  EXPECT_EQ(unsupported_count, 1uz);
}

TEST(SafetensorsStorage, KeepsValidatedTensorViewsInOneFileMapping)
{
  constexpr std::string_view json =
      R"({"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})";
  const TemporarySafetensors checkpoint{json, 8uz};
  const auto weights = fe::ModelWeights::open(checkpoint.string());
  ASSERT_TRUE(weights);
  EXPECT_EQ((*weights)->mapped_file_bytes(), 8uz + json.size() + 8uz);
  EXPECT_EQ((*weights)->weight_bytes(), 8uz);
  ASSERT_EQ((*weights)->tensors().size(), 1uz);
  EXPECT_EQ((*weights)->tensors().front().bytes, 8uz);
  EXPECT_NE((*weights)->tensors().front().data, nullptr);
}

} // namespace
