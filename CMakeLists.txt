cmake_minimum_required(VERSION 3.21)
project(FlowEdge LANGUAGES C CXX)

find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

function(flowedge_executable target source)
  add_executable(${target} ${source})
  target_link_libraries(${target} PRIVATE ${ARGN})
endfunction()

option(FLOWEDGE_TESTS "Build the GoogleTest suite (dev/CI-only)" OFF)
option(FLOWEDGE_BENCH "Build the Google Benchmark microbenchmarks (dev-only)" OFF)
option(FLOWEDGE_PYTHON "Build the pybind11 Python module (dev/verification)" OFF)
option(FLOWEDGE_RELAY "Build the local real-time inference relay" OFF)
set(FLOWEDGE_RELAY_MAX_CONDITION_DIM "4096" CACHE STRING
    "Maximum condition width stored in a Relay message")
set(FLOWEDGE_RELAY_MAX_ACTION_DIM "1024" CACHE STRING
    "Maximum action width stored in a Relay message")
set(FLOWEDGE_RELAY_MAX_JOB_PAYLOAD_BYTES "65536" CACHE STRING
    "Maximum opaque payload stored in a generic Relay job message")

# Core compute kernels. CPU auto-selects the host ISA; scalar is also explicit
# for compatibility testing and targets where deploying AVX2 is undesirable.
# cuda requires nvcc. Host-span kernels still copy in and out; flow and
# Diffusion Policy heads keep weights and solver scratch on device after load.
set(FLOWEDGE_BACKEND "cpu" CACHE STRING "Kernel backend: cpu, avx512, scalar, or cuda")
set_property(CACHE FLOWEDGE_BACKEND PROPERTY STRINGS cpu avx512 scalar cuda)
if(FLOWEDGE_BACKEND STREQUAL "cuda")
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR
      "FLOWEDGE_BACKEND=cuda requires nvcc. Install the CUDA toolkit, or keep FLOWEDGE_BACKEND=cpu.")
  endif()
  enable_language(CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
  endif()
endif()

# CORE
add_subdirectory(src/core)
if(FLOWEDGE_RELAY)
  add_subdirectory(src/relay)
endif()

# EXAMPLES
foreach(ex flow_sample mamba_forward external_flow_sample streaming_snapshot diffusion_sample)
  flowedge_executable(${ex} examples/core/${ex}.cc FlowEdge::Core)
endforeach()
# The same checkpoint/input runner supports Transformer and Mamba backbones. Keep
# the historical mamba_forward target while giving Transformer documentation a
# model-neutral executable name.
flowedge_executable(transformer_forward examples/core/mamba_forward.cc FlowEdge::Core)
flowedge_executable(transformer_latency tools/transformer_latency.cc FlowEdge::Core)
flowedge_executable(flowedge-inspect tools/checkpoint_inspect.cc FlowEdge::Core)
flowedge_executable(flowedge_model_lifecycle_check tools/model_lifecycle_check.cc FlowEdge::Core)
if(FLOWEDGE_BACKEND STREQUAL "cuda")
  flowedge_executable(flowedge_cuda_dp_mix tools/cuda_dp_mix.cc FlowEdge::Core)
endif()
if(FLOWEDGE_RELAY)
  foreach(ex relay_client_sample cooperative_job_sample routed_job_sample mamba_relay_stream
             action_delivery_sample)
    flowedge_executable(${ex} examples/relay/${ex}.cc FlowEdge::Relay)
  endforeach()
endif()

# BENCHMARKS
if(FLOWEDGE_BENCH)
  include(FetchContent)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(googlebenchmark)

  foreach(t benchmark benchmark_main)
    if(TARGET ${t})
      target_compile_options(${t} PRIVATE -Wno-c2y-extensions)
    endif()
  endforeach()

  flowedge_executable(flowedge_kernels_bench tools/benchmark/kernels/kernels_bench.cc FlowEdge::Core
                      benchmark::benchmark)
  flowedge_executable(flowedge_diffusion_breakdown tools/benchmark/kernels/diffusion_breakdown.cc
                      FlowEdge::Core)
  execute_process(
    COMMAND git rev-parse --short=12 HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE FLOWEDGE_GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(NOT FLOWEDGE_GIT_COMMIT)
    set(FLOWEDGE_GIT_COMMIT "unknown")
  endif()
  flowedge_executable(flowedge-profile tools/benchmark/runtime/deadline_profile.cc FlowEdge::Core)
  target_compile_definitions(flowedge-profile PRIVATE
    FLOWEDGE_GIT_COMMIT="${FLOWEDGE_GIT_COMMIT}")

  flowedge_executable(flowedge_threaded_matmul_bench tools/benchmark/kernels/threaded_matmul_bench.cc
                      FlowEdge::Core benchmark::benchmark_main Threads::Threads)

  if(FLOWEDGE_RELAY)
    flowedge_executable(flowedge_relay_bench tools/benchmark/runtime/relay_bench.cc FlowEdge::Relay)
    flowedge_executable(flowedge_relay_pool_bench tools/benchmark/runtime/relay_pool_bench.cc FlowEdge::Relay)
    flowedge_executable(flowedge_cooperative_job_bench tools/benchmark/runtime/cooperative_job_bench.cc
                        FlowEdge::Relay)
    flowedge_executable(flowedge_job_queue_bench tools/benchmark/runtime/job_queue_bench.cc FlowEdge::Relay)
    flowedge_executable(flowedge_job_qos_bench tools/benchmark/runtime/job_qos_bench.cc FlowEdge::Relay)
    flowedge_executable(flowedge_mamba_stream_bench tools/benchmark/runtime/mamba_stream_bench.cc FlowEdge::Relay)
    flowedge_executable(flowedge_worker_drain_bench tools/benchmark/runtime/worker_drain_bench.cc FlowEdge::Relay)
    flowedge_executable(flowedge_action_delivery_bench tools/benchmark/runtime/action_delivery_bench.cc
                        FlowEdge::Relay)
  endif()
endif()

# TESTS
if(FLOWEDGE_TESTS)
  enable_testing()
  include(FetchContent)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(googletest)

  add_executable(flowedge_tests test/kernel_tests.cc test/core_model_tests.cc test/transformer_tests.cc
                 test/cabi_tests.cc test/deployment_profile_tests.cc test/deployment_profile_runtime_tests.cc
                 test/deadline_profile_tests.cc test/safetensors_tests.cc)
  target_link_libraries(flowedge_tests PRIVATE FlowEdge::Core GTest::gtest_main Threads::Threads)
  target_include_directories(flowedge_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(flowedge_tests PRIVATE FLOWEDGE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

  if(FLOWEDGE_RELAY)
    add_executable(flowedge_job_service_fixture test/job_service_fixture.cc)
    target_link_libraries(flowedge_job_service_fixture PRIVATE FlowEdge::Relay)

    add_executable(flowedge_relay_tests test/relay_protocol_tests.cc test/relay_worker_tests.cc
                                         test/relay_runtime_tests.cc test/relay_process_tests.cc
                                         test/relay_parse_tests.cc)
    target_link_libraries(flowedge_relay_tests PRIVATE FlowEdge::Relay GTest::gtest_main)
    target_compile_definitions(flowedge_relay_tests PRIVATE
      FLOWEDGE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
      FLOWEDGE_RELAYD_PATH="$<TARGET_FILE:flowedge-relayd>"
      FLOWEDGE_JOBD_PATH="$<TARGET_FILE:flowedge-jobd>"
      FLOWEDGE_JOB_SERVICE_FIXTURE_PATH="$<TARGET_FILE:flowedge_job_service_fixture>")
    add_dependencies(flowedge_relay_tests flowedge-relayd flowedge-jobd flowedge_job_service_fixture)
  endif()
  
  include(GoogleTest)
  gtest_discover_tests(flowedge_tests)
  add_test(NAME Tools.InspectHelp COMMAND flowedge-inspect --help)
  add_test(NAME Tools.TransformerLatencyHelp COMMAND transformer_latency --help)
  if(FLOWEDGE_BENCH)
    add_test(NAME Tools.ProfileHelp COMMAND flowedge-profile --help)
  endif()
  if(FLOWEDGE_RELAY)
    gtest_discover_tests(flowedge_relay_tests)
    add_test(NAME RelayTools.ClientHelp COMMAND flowedge-relayctl --help)
    add_test(NAME RelayTools.JobClientHelp COMMAND flowedge-jobctl --help)
    add_test(NAME RelayTools.JobDaemonHelp COMMAND flowedge-jobd --help)
    add_test(NAME RelayTools.TraceHelp COMMAND flowedge-relay-trace --help)
  endif()
endif()

# PYTHON (Pybind11)
if(FLOWEDGE_PYTHON)
  set(PYBIND11_FINDPYTHON ON)
  find_package(pybind11 CONFIG QUIET)
  if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG v2.13.6
      GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(pybind11)
  endif()

  # nvcc device-link LTO bytecode is not compatible with host GCC -flto.
  if(FLOWEDGE_BACKEND STREQUAL "cuda")
    pybind11_add_module(flowedge NO_EXTRAS python/flowedge_ext.cc)
  else()
    pybind11_add_module(flowedge python/flowedge_ext.cc)
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND NOT MSVC)
    target_compile_options(flowedge PRIVATE -pthread)
  endif()
  if(WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang"
     AND NOT CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
    # MinGW/GNU-ABI Clang: statically link the C++ runtime and ship winpthread.
    target_link_options(flowedge PRIVATE
      -static-libgcc -static-libstdc++ "-Wl,--allow-multiple-definition")
    target_link_libraries(flowedge PRIVATE winpthread)
    find_file(FLOWEDGE_WINPTHREAD_DLL NAMES libwinpthread-1.dll HINTS $ENV{PATH})
    if(NOT FLOWEDGE_WINPTHREAD_DLL)
      message(FATAL_ERROR "libwinpthread-1.dll is required to package the MinGW Python extension")
    endif()
    install(FILES "${FLOWEDGE_WINPTHREAD_DLL}" DESTINATION .)
    add_custom_command(TARGET flowedge POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${FLOWEDGE_WINPTHREAD_DLL}" "$<TARGET_FILE_DIR:flowedge>")
  endif()
  target_link_libraries(flowedge PRIVATE FlowEdge::Core)
  install(TARGETS flowedge LIBRARY DESTINATION . RUNTIME DESTINATION .) # wheel root: pip install .
endif()

if(NOT SKBUILD)
include(GNUInstallDirs)
set(FLOWEDGE_INSTALL_TARGETS flowedge_core flowedge_engine)
if(TARGET flowedge-inspect)
  list(APPEND FLOWEDGE_INSTALL_TARGETS flowedge-inspect)
endif()
if(TARGET flowedge_relay)
  list(APPEND FLOWEDGE_INSTALL_TARGETS flowedge_relay)
endif()
install(TARGETS ${FLOWEDGE_INSTALL_TARGETS}
        EXPORT FlowEdgeTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/core/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h")
if(TARGET flowedge_relay)
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/relay/
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/relay
          FILES_MATCHING PATTERN "*.h")
endif()

set(FLOWEDGE_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/FlowEdge)
install(EXPORT FlowEdgeTargets
        FILE FlowEdgeTargets.cmake
        NAMESPACE FlowEdge::
        DESTINATION ${FLOWEDGE_CMAKEDIR})

include(CMakePackageConfigHelpers)
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FlowEdgeConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/FlowEdgeConfig.cmake"
  INSTALL_DESTINATION ${FLOWEDGE_CMAKEDIR})
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/FlowEdgeConfigVersion.cmake"
  VERSION 0.1.1
  COMPATIBILITY AnyNewerVersion)
install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/FlowEdgeConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/FlowEdgeConfigVersion.cmake"
        DESTINATION ${FLOWEDGE_CMAKEDIR})
endif()
