import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from flowedge_lerobot.cli import _parser, main, run_simulator


class FakePolicy:
    metadata = type("Metadata", (), {"condition_dim": 3, "horizon": 4})()
    action_dim = 2
    cuda_resident = True

    def __init__(self):
        self.calls = 0

    def reset(self):
        pass

    def select_action_into(self, condition, noise, output, **kwargs):
        self.calls += 1
        output.fill(float(condition[0] + noise[0, 0] + kwargs["seed"]))


class CliTests(unittest.TestCase):
    def test_parser_defaults(self):
        args = _parser().parse_args(["policy.safetensors"])
        self.assertEqual(args.steps, 10)
        self.assertEqual(args.diffusion_steps, 10)
        self.assertEqual(args.scheduler, "ddim")
        self.assertIsNone(args.period_ms)
        self.assertEqual(args.on_miss, "hold")
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.precision, "exact-f32")

    def test_simulator_is_bounded_and_stops(self):
        policy = FakePolicy()
        result = run_simulator(
            policy, steps=3, diffusion_steps=2, scheduler="ddim", seed=7
        )
        self.assertEqual(result.steps, 3)
        self.assertTrue(result.stopped)
        self.assertEqual(policy.calls, 3)

    def test_main_loads_checkpoint_and_emits_result(self):
        with patch(
            "flowedge_lerobot.cli.FlowEdgeDiffusionPolicy.from_checkpoint"
        ) as load:
            load.return_value = FakePolicy()
            with patch("builtins.print") as print_result:
                self.assertEqual(main(["policy.safetensors", "--steps", "2"]), 0)
        load.assert_called_once_with(
            "policy.safetensors",
            threads=None,
            device="cpu",
            precision="exact-f32",
        )
        payload = json.loads(print_result.call_args.args[0])
        self.assertEqual(payload["steps"], 2)
        self.assertTrue(payload["stopped"])
        self.assertIn("p99_ms", payload)
        self.assertGreaterEqual(payload["startup_ms"], 0.0)
        self.assertIsInstance(payload["peak_rss_bytes"], (int, type(None)))
        self.assertTrue(payload["platform"])
        self.assertTrue(payload["machine"])
        self.assertEqual(payload["device"], "cpu")
        self.assertEqual(payload["precision"], "exact-f32")

    def test_cuda_fails_closed_without_native_backend(self):
        with patch.dict(sys.modules, {"flowedge": SimpleNamespace(cuda=False)}):
            with patch(
                "flowedge_lerobot.cli.FlowEdgeDiffusionPolicy.from_checkpoint"
            ) as load:
                with self.assertRaises(SystemExit):
                    main(["policy.safetensors", "--device", "cuda"])
            load.assert_not_called()

    def test_cuda_fails_closed_without_resident_weights(self):
        policy = FakePolicy()
        policy.cuda_resident = False
        with patch.dict(sys.modules, {"flowedge": SimpleNamespace(cuda=True)}):
            with patch(
                "flowedge_lerobot.cli.FlowEdgeDiffusionPolicy.from_checkpoint"
            ) as load:
                load.return_value = policy
                with self.assertRaises(SystemExit):
                    main(["policy.safetensors", "--device", "cuda"])

    def test_cuda_records_device_and_limitations(self):
        with patch.dict(sys.modules, {"flowedge": SimpleNamespace(cuda=True)}):
            with patch(
                "flowedge_lerobot.cli.FlowEdgeDiffusionPolicy.from_checkpoint"
            ) as load:
                load.return_value = FakePolicy()
                with patch(
                    "flowedge_lerobot.cli._cuda_device_name",
                    return_value="NVIDIA GeForce GTX 1650",
                ):
                    with patch("builtins.print") as print_result:
                        self.assertEqual(
                            main(
                                [
                                    "policy.safetensors",
                                    "--device",
                                    "cuda",
                                    "--steps",
                                    "1",
                                ]
                            ),
                            0,
                        )
        payload = json.loads(print_result.call_args.args[0])
        self.assertEqual(payload["device"], "cuda")
        self.assertEqual(payload["cuda_device"], "NVIDIA GeForce GTX 1650")
        self.assertTrue(payload["limitations"])

    def test_host_facts_mark_edge_rollout_json(self):
        with patch(
            "flowedge_lerobot.cli.FlowEdgeDiffusionPolicy.from_checkpoint"
        ) as load:
            load.return_value = FakePolicy()
            with patch(
                "flowedge_lerobot.cli._jetson_facts",
                return_value={"detected": True, "model": "Jetson AGX Orin"},
            ):
                with patch("builtins.print") as print_result:
                    self.assertEqual(
                        main(["policy.safetensors", "--steps", "1", "--host-facts"]),
                        0,
                    )
        payload = json.loads(print_result.call_args.args[0])
        self.assertEqual(payload["evaluation_kind"], "edge_dp_rollout")
        self.assertTrue(payload["checkpoint"])
        self.assertIn("processor", payload)
        self.assertIn("python", payload)
        self.assertEqual(payload["jetson"]["model"], "Jetson AGX Orin")

    def test_cuda_host_facts_do_not_mislabel_non_jetson_run(self):
        with patch.dict(sys.modules, {"flowedge": SimpleNamespace(cuda=True)}):
            with patch(
                "flowedge_lerobot.cli.FlowEdgeDiffusionPolicy.from_checkpoint",
                return_value=FakePolicy(),
            ):
                with patch(
                    "flowedge_lerobot.cli._jetson_facts",
                    return_value={"detected": False},
                ):
                    with patch("builtins.print") as print_result:
                        self.assertEqual(
                            main(
                                [
                                    "policy.safetensors",
                                    "--device",
                                    "cuda",
                                    "--steps",
                                    "1",
                                    "--host-facts",
                                ]
                            ),
                            0,
                        )
        payload = json.loads(print_result.call_args.args[0])
        self.assertIn("No physical Jetson", payload["limitations"][-1])


if __name__ == "__main__":
    unittest.main()
