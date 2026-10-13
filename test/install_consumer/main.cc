#include "api/engine.h"
#include "protocol/deadline_flow.h"
#include "relay/adapters/mamba_stream_adapter.h"
#include "relay/adapters/routed_adapters.h"
#include "relay/client/action_delivery.h"
#include "relay/client/job_client.h"
#include "relay/client/job_control_client.h"
#include "relay/scheduler/edf_scheduler.h"
#include "relay/telemetry/metrics.h"
#include "relay/worker/job_control_service.h"
#include "relay/worker/job_service.h"
#include "relay/worker/job_worker_pool.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <span>
#include <string>
#include <thread>
#include <utility>

namespace {

struct InstalledBackend
{
  [[nodiscard]] bool prepare(std::span<const std::byte> request) noexcept
  {
    return request.empty();
  }
  [[nodiscard]] bool begin() noexcept { return true; }
  [[nodiscard]] fe::relay::BackendAdvance advance(std::size_t) noexcept
  {
    return {fe::relay::BackendStep::kComplete, 1uz};
  }
  void cancel() noexcept {}
  [[nodiscard]] std::size_t state_bytes() const noexcept { return 0uz; }
  [[nodiscard]] bool save_state(std::span<std::byte> state) const noexcept { return state.empty(); }
  [[nodiscard]] bool load_state(std::span<const std::byte> state) noexcept { return state.empty(); }
  [[nodiscard]] std::span<const std::byte> result() const noexcept { return {}; }
};

static_assert(fe::relay::CooperativeBackend<InstalledBackend>);
static_assert(fe::relay::RoutedBackend<InstalledBackend>);
static_assert(fe::relay::RoutedBackend<fe::relay::MambaStreamAdapter>);

} // namespace

int main()
{
  fe_engine_free(nullptr);
  fe_diffusion_metadata diffusion{};
  if (fe_engine_action_horizon(nullptr) != 0uz || fe_engine_diffusion_cuda_device(nullptr) != -1 ||
      fe_engine_diffusion_metadata(nullptr, &diffusion) != 1 ||
      fe_engine_sample_diffusion(nullptr, nullptr, nullptr, 0uz, FE_DIFFUSION_DDIM, 0u, nullptr) !=
          1 ||
      fe_engine_sample_diffusion_device(nullptr, nullptr, nullptr, 0uz, FE_DIFFUSION_DDIM, 0u, 0uz,
                                        0uz, nullptr) != 1 ||
      fe_engine_diffusion_denoise(nullptr, nullptr, nullptr, 0.0F, nullptr) != 1)
    return 1;
  fe_model_metadata delivery_model{};
  delivery_model.struct_size = sizeof(delivery_model);
  delivery_model.protocol_version = FE_PROTOCOL_VERSION;
  delivery_model.action_dim = 2u;
  delivery_model.model_digest.bytes[0] = 7u;
  constexpr std::array delivery_lower{-1.0F, -1.0F};
  constexpr std::array delivery_upper{1.0F, 1.0F};
  constexpr std::array delivery_delta{0.5F, 0.5F};
  auto delivery = fe::relay::ActionDeliveryGate::create(
      delivery_model, fe::relay::ActionDeliveryConfig{.control_dim = 2uz, .step_period_ns = 1u},
      fe::relay::ActionSafetyLimits{.lower = delivery_lower,
                                    .upper = delivery_upper,
                                    .max_delta_per_step = delivery_delta});
  if (!delivery)
    return 1;
  fe::relay::ActionMessage action{};
  action.envelope.sequence = 1u;
  action.envelope.session_id = 1u;
  action.metadata.struct_size = sizeof(action.metadata);
  action.metadata.protocol_version = FE_PROTOCOL_VERSION;
  action.metadata.status = FE_ACTION_COMPLETE;
  action.metadata.model_digest = delivery_model.model_digest;
  action.metadata.timestamp_ns = 1u;
  action.metadata.generation = 1u;
  action.metadata.action_dim = delivery_model.action_dim;
  action.envelope.struct_size = static_cast<std::uint32_t>(fe::relay::wire_size(action));
  if (delivery->accept(action, 1u, 1u) != fe::relay::ActionChunkAcceptResult::kAccepted ||
      !delivery->next(1u))
    return 1;
  const fe::relay::EdfScheduler scheduler{1uz};
  InstalledBackend backend{};
  fe::relay::JobDescriptor descriptor{};
  descriptor.kind = fe::relay::JobKind::kIterative;
  descriptor.model_digest[0] = 1u;
  descriptor.state_schema = 1u;
  descriptor.total_work_units = 1u;
  auto job = fe::relay::make_iterative_job(backend, descriptor);
  if (!job || !job->start())
    return 1;
  const auto complete = job->advance(1uz);
  std::array<fe::relay::JobAdapterRegistration, 1> entries{};
  fe::relay::JobAdapterRegistry registry{entries};
  if (!registry.add(
          fe::relay::make_routed_adapter(backend, fe::relay::job_route(descriptor), 0uz, 0uz)))
    return 1;
  registry.freeze();
  fe::relay::JobRequestMessage request{};
  if (fe::relay::make_job_request(request, 1u, descriptor, 0u, {},
                                  fe::relay::JobServiceClass::kCritical) !=
      fe::relay::ProtocolResult::kSuccess)
    return 1;
  auto routed = registry.bind(request);
  if (!routed || !routed->start())
    return 1;
  const auto routed_complete = routed->advance(1uz);
  std::array<fe::relay::JobAdapterRegistry, 1> lanes{registry};
  auto events = fe::relay::JobEventBuffer::create(4uz);
  if (!events)
    return 1;
  auto created = fe::relay::JobWorkerPool::create(lanes, 1uz, {}, 1uz, 1uz,
                                                  fe::relay::WorkerPlacement::kNone, &*events);
  if (!created)
    return 1;
  const std::string suffix =
      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  const std::string request_name = "flowedge-install-request-" + suffix;
  const std::string result_name = "flowedge-install-result-" + suffix;
  const std::string control_name = "flowedge-install-control-" + suffix;
  const std::string status_name = "flowedge-install-status-" + suffix;
  auto client_result = fe::relay::JobClient::create(request_name, result_name, 2u);
  auto service_result = fe::relay::JobService::connect(request_name, result_name, *created);
  auto control_client_result = fe::relay::JobControlClient::create(control_name, status_name, 2u);
  if (!client_result || !service_result || !control_client_result)
    return 1;
  fe::relay::JobClient client = std::move(*client_result);
  fe::relay::JobService service = std::move(*service_result);
  auto control_service_result =
      fe::relay::JobControlService::connect(control_name, status_name, *created, service.stats());
  if (!control_service_result)
    return 1;
  fe::relay::JobControlClient control_client = std::move(*control_client_result);
  fe::relay::JobControlService control_service = std::move(*control_service_result);
  const fe::relay::JobControlRequest status_request =
      fe::relay::make_job_control_request(1u, descriptor.session_id,
                                          fe::relay::JobControlOperation::kStatus);
  fe::relay::JobControlResponse status_response{};
  if (control_client.try_submit(status_request) != fe::relay::ClientResult::kSuccess ||
      control_service.poll() != fe::relay::JobControlServiceResult::kProgress ||
      control_client.try_receive(status_response) != fe::relay::ClientResult::kSuccess ||
      status_response.metadata.status.worker_count != 1u)
    return 1;
  if (client.try_submit(request) != fe::relay::ClientResult::kSuccess)
    return 1;
  fe::relay::JobResultMessage pooled{};
  fe::relay::ClientResult received{fe::relay::ClientResult::kEmpty};
  while (received == fe::relay::ClientResult::kEmpty) {
    const fe::relay::JobServiceResult pumped = service.poll();
    if (pumped == fe::relay::JobServiceResult::kCorruptInput ||
        pumped == fe::relay::JobServiceResult::kTransportError)
      return 1;
    received = client.try_receive(pooled);
    if (received == fe::relay::ClientResult::kEmpty)
      std::this_thread::yield();
  }
  const bool pooled_complete =
      received == fe::relay::ClientResult::kSuccess &&
      fe::relay::job_result_code(pooled) == fe::relay::JobResultCode::kComplete &&
      pooled.metadata.service_class == fe::relay::JobServiceClass::kCritical;
  fe::relay::JobMetrics metrics{};
  while (const fe::relay::JobEventMessage* event = events->front()) {
    metrics.record(*event);
    events->pop();
  }
  if (client.try_shutdown(2u, descriptor.session_id) != fe::relay::ClientResult::kSuccess ||
      service.poll() != fe::relay::JobServiceResult::kStopped)
    return 1;
  const bool session_released = created->release_session(descriptor.session_id);
  const bool drain_started =
      created->request_worker_drain(0uz) == fe::relay::WorkerDrainResult::kStarted;
  const bool drained = created->worker_drained(0uz) && created->accepting_worker_count() == 0uz;
  const bool resumed = created->resume_worker(0uz);
  const std::array plans{fe::ExecutionPlan{1u, 4u, 0u, 0u, 0u, 1u, 20u, 0u, 0.6}};
  auto selector = fe::DeadlineFlow::create(plans);
  const bool plan_selected = selector && selector->select(fe::PlanContext{.slack_ns = 100u}) == 1u;
  return plan_selected && scheduler.capacity() == 1uz && complete && routed_complete &&
                 pooled_complete && session_released && drain_started && drained && resumed &&
                 metrics.counters().completed == 1u &&
                 fe::relay::mamba_stream_request_bytes(4uz) == 4uz * sizeof(std::int32_t) &&
                 complete->state == fe::relay::JobState::kComplete &&
                 routed_complete->state == fe::relay::JobState::kComplete
             ? 0
             : 1;
}
