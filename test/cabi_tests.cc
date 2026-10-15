#include "api/engine.h"
#include "test/test_utils.h"
#ifdef FLOWEDGE_CUDA
#include "kernels/cuda/kernels_cuda_api.h"
#endif

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <gtest/gtest.h>
#include <string>
#include <thread>
#include <vector>

namespace {

TEST(CAbiContract, StatusValuesAreStable)
{
  EXPECT_EQ(FE_STATUS_OK, 0);
  EXPECT_EQ(FE_STATUS_INVALID_ARGUMENT, 1);
  EXPECT_EQ(FE_STATUS_RESOURCE_EXHAUSTED, 2);
  EXPECT_EQ(FE_STATUS_TOKEN_OUT_OF_RANGE, 3);
  EXPECT_EQ(FE_STATUS_UNSUPPORTED_MODEL, 4);
  EXPECT_EQ(FE_STATUS_INVALID_SOLVER, 5);
  EXPECT_EQ(FE_STATUS_NO_ACTIVE_OPERATION, 6);
  EXPECT_EQ(FE_STATUS_STATE_UNAVAILABLE, 7);
  EXPECT_EQ(FE_STATUS_CANCELLED, 8);
  EXPECT_EQ(FE_STATUS_INCOMPATIBLE_STATE, 9);
  EXPECT_EQ(FE_STATUS_CORRUPT_STATE, 10);
  EXPECT_EQ(FE_DIFFUSION_PRECISION_EXACT_F32, 0);
  EXPECT_EQ(FE_DIFFUSION_PRECISION_APPROX_FP16_DEEP, 1);
}

TEST(CAbiContract, NullHandlesReportInvalidArguments)
{
  std::int32_t token{};
  float value{};
  std::size_t remaining{};

  EXPECT_EQ(fe_engine_run(nullptr, &token, 1uz, &value), FE_STATUS_INVALID_ARGUMENT);
  EXPECT_STREQ(fe_engine_last_error(), "Invalid arguments to fe_engine_run");
  EXPECT_EQ(fe_engine_run_embeddings(nullptr, &value, 1uz, &value), FE_STATUS_INVALID_ARGUMENT);
  EXPECT_STREQ(fe_engine_last_error(), "Invalid arguments to fe_engine_run_embeddings");
  const std::uint8_t mask = 1u;
  EXPECT_EQ(fe_engine_run_embeddings_masked(nullptr, &value, &mask, 1uz, &value),
            FE_STATUS_INVALID_ARGUMENT);
  EXPECT_STREQ(fe_engine_last_error(), "Invalid arguments to fe_engine_run_embeddings_masked");
  EXPECT_EQ(fe_engine_smolvla_embed_suffix(nullptr, &value, 1uz, 0.5F, &value),
            FE_STATUS_INVALID_ARGUMENT);
  EXPECT_STREQ(fe_engine_last_error(), "Invalid arguments to fe_engine_smolvla_embed_suffix");
  EXPECT_EQ(fe_engine_smolvla_run_expert(nullptr, &value, 1uz, &value, &value, &mask, 1uz, &value),
            FE_STATUS_INVALID_ARGUMENT);
  EXPECT_STREQ(fe_engine_last_error(), "Invalid arguments to fe_engine_smolvla_run_expert");
  EXPECT_EQ(fe_engine_smolvla_project_actions(nullptr, &value, 1uz, &value),
            FE_STATUS_INVALID_ARGUMENT);
  EXPECT_STREQ(fe_engine_last_error(), "Invalid arguments to fe_engine_smolvla_project_actions");
  EXPECT_EQ(fe_engine_smolvla_denoise(nullptr, &value, 1uz, 0.5F, &value, &value, &mask, 1uz,
                                      &value),
            FE_STATUS_INVALID_ARGUMENT);
  EXPECT_STREQ(fe_engine_last_error(), "Invalid arguments to fe_engine_smolvla_denoise");
  EXPECT_EQ(fe_engine_smolvla_sample(nullptr, &value, 1uz, 10uz, &value, &value, &mask, 1uz,
                                     &value),
            FE_STATUS_INVALID_ARGUMENT);
  EXPECT_STREQ(fe_engine_last_error(), "Invalid arguments to fe_engine_smolvla_sample");
  EXPECT_EQ(fe_engine_step(nullptr, token, &value), FE_STATUS_INVALID_ARGUMENT);
  EXPECT_STREQ(fe_engine_last_error(), "Invalid null arguments to fe_engine_step");
  EXPECT_EQ(fe_engine_flow_advance(nullptr, 1uz, &value, &remaining), FE_STATUS_INVALID_ARGUMENT);
  EXPECT_STREQ(fe_engine_last_error(), "Invalid arguments to fe_engine_flow_advance");
  EXPECT_EQ(fe_engine_export_decode_state(nullptr, &value, sizeof(value)),
            FE_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(fe_engine_import_decode_state(nullptr, &value, sizeof(value)),
            FE_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(fe_engine_decode_state_bytes(nullptr), 0uz);
  EXPECT_EQ(fe_engine_action_dim(nullptr), 0uz);
  EXPECT_EQ(fe_engine_condition_dim(nullptr), 0uz);
  EXPECT_EQ(fe_engine_cuda_resident(nullptr), 0);
  EXPECT_EQ(fe_engine_diffusion_cuda_device(nullptr), -1);
  EXPECT_EQ(fe_engine_diffusion_precision(nullptr), FE_DIFFUSION_PRECISION_EXACT_F32);
  EXPECT_EQ(fe_engine_sample_diffusion_device(nullptr, nullptr, nullptr, 0uz, FE_DIFFUSION_DDIM, 0u,
                                              0uz, 0uz, nullptr),
            FE_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(fe_weights_size_bytes(nullptr), 0uz);
}

TEST(CAbiContract, RejectsUnknownDiffusionPrecision)
{
  EXPECT_EQ(fe_engine_load_with_threads_and_diffusion_precision(nullptr, 0u, 99), nullptr);
  EXPECT_NE(std::string{fe_engine_last_error()}.find("precision"), std::string::npos);
}

TEST(CAbiContract, LastErrorIsThreadLocal)
{
  ASSERT_EQ(fe_engine_load(nullptr), nullptr);
  const std::string main_error{fe_engine_last_error()};
  std::string worker_error;
  std::thread worker{[&] {
    static_cast<void>(fe_engine_step(nullptr, 0, nullptr));
    worker_error = fe_engine_last_error();
  }};
  worker.join();

  EXPECT_EQ(main_error, "Invalid null model path");
  EXPECT_EQ(worker_error, "Invalid null arguments to fe_engine_step");
  EXPECT_STREQ(fe_engine_last_error(), main_error.c_str());
}

#ifdef FLOWEDGE_SOURCE_DIR
TEST(CAbiContract, DecodeBuffersRejectUndersizedStorage)
{
  const std::filesystem::path model =
      std::filesystem::path{FLOWEDGE_SOURCE_DIR} / "models" / "mamba_flow.safetensors";
  if (!std::filesystem::exists(model))
    GTEST_SKIP() << "models/mamba_flow.safetensors is not available";

  fe_engine* engine = fe_engine_load_with_threads(model.string().c_str(), 0u);
  ASSERT_NE(engine, nullptr) << fe_engine_last_error();
  const std::size_t bytes = fe_engine_decode_state_bytes(engine);
  ASSERT_GT(bytes, 1uz);
  std::vector<std::byte> undersized(bytes - 1uz);
  EXPECT_EQ(fe_engine_export_decode_state(engine, undersized.data(), undersized.size()),
            FE_STATUS_INVALID_ARGUMENT);
  EXPECT_NE(std::string{fe_engine_last_error()}.find("too small"), std::string::npos);
  EXPECT_EQ(fe_engine_import_decode_state(engine, undersized.data(), undersized.size()),
            FE_STATUS_INVALID_ARGUMENT);
  EXPECT_NE(std::string{fe_engine_last_error()}.find("truncated"), std::string::npos);
  fe_engine_free(engine);
}

TEST(CAbiContract, CudaRequiredFailsClosedOnCpuBinary)
{
  const std::filesystem::path model =
      std::filesystem::path{FLOWEDGE_SOURCE_DIR} / "models" / "mamba_flow.safetensors";
  if (!std::filesystem::exists(model))
    GTEST_SKIP() << "models/mamba_flow.safetensors is not available";

#ifndef FLOWEDGE_CUDA
  const fe::test::EnvOverride require{"FLOWEDGE_CUDA_REQUIRED", "1"};
  EXPECT_EQ(fe_engine_load_with_threads(model.string().c_str(), 0u), nullptr);
  EXPECT_NE(std::string{fe_engine_last_error()}.find("no CUDA kernels"), std::string::npos);
#else
  const fe::test::EnvOverride host{"FLOWEDGE_CUDA_FORCE_HOST", "1"};
  const fe::test::EnvOverride require{"FLOWEDGE_CUDA_REQUIRED", "1"};
  EXPECT_EQ(fe_engine_load_with_threads(model.string().c_str(), 0u), nullptr);
  EXPECT_NE(std::string{fe_engine_last_error()}.find("did not attach"), std::string::npos);
#endif
}

TEST(CAbiContract, ApproximateDiffusionPrecisionRejectsNonDiffusionModels)
{
  const std::filesystem::path model =
      std::filesystem::path{FLOWEDGE_SOURCE_DIR} / "models" / "mamba_flow.safetensors";
  if (!std::filesystem::exists(model))
    GTEST_SKIP() << "models/mamba_flow.safetensors is not available";

  EXPECT_EQ(fe_engine_load_with_threads_and_diffusion_precision(
                model.string().c_str(), 0u, FE_DIFFUSION_PRECISION_APPROX_FP16_DEEP),
            nullptr);
  EXPECT_NE(std::string{fe_engine_last_error()}.find("CUDA-resident diffusion head"),
            std::string::npos);
}

TEST(CAbiContract, LoadKeepsCpuKernelsWhenCudaDoesNotAttach)
{
  const std::filesystem::path model =
      std::filesystem::path{FLOWEDGE_SOURCE_DIR} / "models" / "mamba_flow.safetensors";
  if (!std::filesystem::exists(model))
    GTEST_SKIP() << "models/mamba_flow.safetensors is not available";

#ifdef FLOWEDGE_CUDA
  const fe::test::EnvOverride host{"FLOWEDGE_CUDA_FORCE_HOST", "1"};
#endif
  fe_engine* engine = fe_engine_load_with_threads(model.string().c_str(), 0u);
  ASSERT_NE(engine, nullptr) << fe_engine_last_error();
  EXPECT_EQ(fe_engine_cuda_resident(engine), 0);
  std::vector<std::int32_t> tokens{1, 2, 3, 4};
  std::size_t d_model{};
  fe_engine_dims(engine, &d_model, nullptr);
  ASSERT_GT(d_model, 0uz);
  std::vector<float> hidden(tokens.size() * d_model);
  EXPECT_EQ(fe_engine_run(engine, tokens.data(), tokens.size(), hidden.data()), FE_STATUS_OK);
  fe_engine_free(engine);
}

TEST(CAbiContract, CudaResidentReportsDeviceAttach)
{
  const std::filesystem::path model =
      std::filesystem::path{FLOWEDGE_SOURCE_DIR} / "models" / "mamba_flow.safetensors";
  if (!std::filesystem::exists(model))
    GTEST_SKIP() << "models/mamba_flow.safetensors is not available";

  fe_engine* engine = fe_engine_load_with_threads(model.string().c_str(), 0u);
  ASSERT_NE(engine, nullptr) << fe_engine_last_error();
#ifdef FLOWEDGE_CUDA
  EXPECT_EQ(fe_engine_cuda_resident(engine), fe::cuda_ops::device_available() ? 1 : 0);
#else
  EXPECT_EQ(fe_engine_cuda_resident(engine), 0);
#endif
  fe_engine_free(engine);
}
#endif

} // namespace
