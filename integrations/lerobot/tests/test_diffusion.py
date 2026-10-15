import numpy as np
import unittest

from flowedge_lerobot import DiffusionActionContract, FlowEdgeDiffusionPolicy


class FakeEngine:
    diffusion_metadata = {
        "condition_dim": 4,
        "action_dim": 2,
        "horizon": 6,
        "action_steps": 3,
        "observation_steps": 2,
    }
    diffusion_precision = "approx-fp16-deep"

    def __init__(self):
        self.calls = []
        self.into_calls = []
        self.dlpack_calls = []

    def sample_diffusion(self, condition, noise, steps, scheduler, seed):
        self.calls.append((condition, noise, steps, scheduler, seed))
        return np.arange(12, dtype=np.float32).reshape(6, 2)

    def sample_diffusion_into(self, condition, noise, output, steps, scheduler, seed):
        self.into_calls.append((condition, noise, steps, scheduler, seed))
        output[...] = np.arange(12, dtype=np.float32).reshape(6, 2)

    def sample_diffusion_dlpack_into(
        self, condition, noise, output, steps, scheduler, seed, action_start
    ):
        self.dlpack_calls.append(
            (condition, noise, output, steps, scheduler, seed, action_start)
        )


class LegacyFakeEngine:
    diffusion_metadata = FakeEngine.diffusion_metadata

    def sample_diffusion(self, condition, noise, steps, scheduler, seed):
        del condition, steps, scheduler, seed
        return noise


class DiffusionAdapterTests(unittest.TestCase):
    def test_adapter_reports_resolved_precision(self):
        self.assertEqual(
            FlowEdgeDiffusionPolicy(FakeEngine()).precision, "approx-fp16-deep"
        )
        self.assertEqual(
            FlowEdgeDiffusionPolicy(LegacyFakeEngine()).precision, "exact-f32"
        )

    def test_contract_slices_the_le_robot_action_chunk(self):
        contract = DiffusionActionContract.from_metadata(FakeEngine.diffusion_metadata)
        self.assertEqual(contract.action_start, 1)
        self.assertEqual(contract.action_steps, 3)

    def test_adapter_forwards_contiguous_inputs_and_slices_horizon(self):
        engine = FakeEngine()
        policy = FlowEdgeDiffusionPolicy(engine)
        condition = np.arange(4, dtype=np.float64).reshape(1, 4)
        noise = np.zeros((6, 2), dtype=np.float64)

        actions = policy.predict_action_chunk(condition, noise, steps=7, seed=11)

        np.testing.assert_array_equal(actions, [[2, 3], [4, 5], [6, 7]])
        self.assertEqual(engine.into_calls[0][0].dtype, np.float32)
        self.assertTrue(engine.into_calls[0][0].flags.c_contiguous)
        self.assertEqual(engine.into_calls[0][1].dtype, np.float32)
        self.assertTrue(engine.into_calls[0][1].flags.c_contiguous)
        self.assertEqual(engine.into_calls[0][2:], (7, "ddim", 11))

    def test_into_writes_caller_owned_storage(self):
        engine = FakeEngine()
        policy = FlowEdgeDiffusionPolicy(engine)
        output = np.empty((3, 2), dtype=np.float32)
        returned = policy.predict_action_chunk_into(
            np.zeros(4), np.zeros((6, 2)), output
        )

        self.assertIs(returned, output)
        np.testing.assert_array_equal(output, [[2, 3], [4, 5], [6, 7]])

    def test_legacy_engine_falls_back_to_returning_horizon(self):
        policy = FlowEdgeDiffusionPolicy(LegacyFakeEngine())
        noise = np.ones((6, 2), dtype=np.float32)
        output = np.empty((3, 2), dtype=np.float32)

        policy.predict_action_chunk_into(np.zeros(4), noise, output)

        np.testing.assert_array_equal(output, noise[1:4])

    def test_adapter_rejects_shape_mismatch(self):
        cases = (
            (np.zeros(3), np.zeros((6, 2)), "condition"),
            (np.zeros(4), np.zeros((5, 2)), "noise"),
        )
        for condition, noise, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                FlowEdgeDiffusionPolicy(FakeEngine()).predict_action_chunk(
                    condition, noise
                )

    def test_dlpack_path_writes_only_the_executable_slice(self):
        engine = FakeEngine()
        policy = FlowEdgeDiffusionPolicy(engine)
        condition = np.zeros(4, dtype=np.float32)
        noise = np.zeros((6, 2), dtype=np.float32)
        output = np.empty((3, 2), dtype=np.float32)

        returned = policy.predict_action_chunk_dlpack_into(
            condition, noise, output, steps=7, seed=11
        )

        self.assertIs(returned, output)
        self.assertEqual(engine.dlpack_calls[0][3:], (7, "ddim", 11, 1))
