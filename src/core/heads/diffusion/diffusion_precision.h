#pragma once

#include <cstdint>

namespace fe {

enum class DiffusionPrecision : std::uint8_t
{
  ExactF32 = 0,
  ApproxFp16Deep = 1,
};

} // namespace fe
