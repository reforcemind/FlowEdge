import unittest
from typing import ClassVar

import numpy as np
from flowedge_lerobot import (
    FlowEdgeSmolVLACachedExpert,
    SmolVLAActionContract,
    SmolVLAKVCache,
)


class FakeEngine:
    model_metadata: ClassVar[dict[str, int]] = {
        "architecture": 7,
        "action_dim": 4,
        "action_horizon": 3,
        "n_layers": 2,
    }

    def __init__(self):
        self.calls = []

    def smolvla_sample(self, noise, prefix_keys, prefix_values, prefix_mask, steps):
        self.calls.append(
            (noise.copy(), prefix_keys, prefix_values, prefix_mask, steps)
        )
        return np.arange(12, dtype=np.float32).reshape(3, 4)

    def smolvla_sample_into(
        self, noise, prefix_keys, prefix_values, prefix_mask, output, steps
    ):
        self.calls.append(
            (noise.copy(), prefix_keys, prefix_values, prefix_mask, output, steps)
        )
        np.copyto(output, np.arange(12, dtype=np.float32).reshape(3, 4))


def cache():
    return SmolVLAKVCache(
        keys=np.zeros((2, 3, 320), dtype=np.float32),
        values=np.zeros((2, 3, 320), dtype=np.float32),
        mask=np.array([1, 1, 0], dtype=np.uint8),
    )


class SmolVLAAdapterTests(unittest.TestCase):
    def test_contract_rejects_non_smolvla_metadata(self):
        with self.assertRaisesRegex(ValueError, "not a FlowEdge SmolVLA"):
            SmolVLAActionContract.from_metadata(
                {"architecture": 1, "action_dim": 4, "action_horizon": 3, "n_layers": 2}
            )

    def test_refill_consumes_a_seeded_action_chunk(self):
        engine = FakeEngine()
        policy = FlowEdgeSmolVLACachedExpert(
            engine, action_dim=2, action_steps=2, seed=7
        )

        policy.refill(cache(), steps=9)
        self.assertIs(engine.calls[0][-2], policy._padded_actions)
        np.testing.assert_array_equal(policy.take_action(), [0, 1])
        output = np.empty(2, dtype=np.float32)
        self.assertIs(policy.take_action_into(output), output)
        np.testing.assert_array_equal(output, [4, 5])
        self.assertEqual(policy.remaining_actions, 0)
        with self.assertRaisesRegex(RuntimeError, "no sampled SmolVLA actions remain"):
            policy.take_action()
        self.assertEqual(engine.calls[0][-1], 9)

        policy.reset()
        policy.refill(cache())
        np.testing.assert_array_equal(engine.calls[0][0], engine.calls[1][0])

    def test_prediction_truncates_padded_actions_and_accepts_explicit_noise(self):
        engine = FakeEngine()
        policy = FlowEdgeSmolVLACachedExpert(engine, action_dim=3, action_steps=2)
        noise = np.ones((3, 4), dtype=np.float32)

        actions = policy.predict_action_chunk(cache(), noise=noise)

        np.testing.assert_array_equal(actions, [[0, 1, 2], [4, 5, 6]])
        np.testing.assert_array_equal(engine.calls[0][0], noise)

    def test_legacy_engine_uses_result_returning_sampler(self):
        engine = FakeEngine()
        engine.smolvla_sample_into = None
        policy = FlowEdgeSmolVLACachedExpert(engine, action_dim=2)

        actions = policy.predict_action_chunk(cache())

        np.testing.assert_array_equal(actions, [[0, 1], [4, 5], [8, 9]])
        self.assertEqual(len(engine.calls), 1)

    def test_cache_and_noise_contracts_accept_sparse_source_masks(self):
        policy = FlowEdgeSmolVLACachedExpert(FakeEngine(), action_dim=2)
        sparse_cache = SmolVLAKVCache(
            keys=np.zeros((2, 3, 320), dtype=np.float32),
            values=np.zeros((2, 3, 320), dtype=np.float32),
            mask=np.array([1, 0, 1], dtype=np.uint8),
        )
        policy.predict_action_chunk(sparse_cache)
        self.assertEqual(len(policy._engine.calls), 1)

        for mask in (np.zeros(3, dtype=np.uint8), np.array([1, 2, 0], dtype=np.uint8)):
            with self.subTest(mask=mask):
                invalid_mask = SmolVLAKVCache(
                    keys=np.zeros((2, 3, 320), dtype=np.float32),
                    values=np.zeros((2, 3, 320), dtype=np.float32),
                    mask=mask,
                )
                with self.assertRaisesRegex(ValueError, "binary array"):
                    policy.predict_action_chunk(invalid_mask)
        with self.assertRaisesRegex(ValueError, "full padded action shape"):
            policy.predict_action_chunk(
                cache(), noise=np.zeros((3, 2), dtype=np.float32)
            )


if __name__ == "__main__":
    unittest.main()
