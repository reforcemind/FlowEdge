import importlib.util
import unittest

import numpy as np
from flowedge_lerobot import LeRobotSmolVLACacheProvider


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
class SmolVLACachePackingTests(unittest.TestCase):
    def test_packing_cuda_cache_matches_cpu_values(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("requires CUDA-enabled PyTorch and a CUDA device")
        provider = LeRobotSmolVLACacheProvider(None, expert_layers=2, key_value_width=4)
        source = torch.arange(12, dtype=torch.float32, device="cuda").reshape(
            1, 3, 2, 2
        )
        cache = {
            0: {"key_states": source, "value_states": source + 1},
            1: {"key_states": source + 2, "value_states": source + 3},
        }
        keys, values = provider._pack_cache(cache)
        np.testing.assert_array_equal(keys[1], np.arange(12).reshape(3, 4) + 2)
        np.testing.assert_array_equal(values[0], np.arange(12).reshape(3, 4) + 1)

    def test_packing_copies_layers_without_aliasing_and_rejects_mismatch(self):
        import torch

        provider = LeRobotSmolVLACacheProvider(None, expert_layers=2, key_value_width=4)
        cache = {
            layer: {
                "key_states": torch.full((1, 3, 2, 2), layer, dtype=torch.float64),
                "value_states": torch.full((1, 3, 2, 2), layer + 10),
            }
            for layer in range(2)
        }
        keys, values = provider._pack_cache(cache)
        self.assertEqual(keys.shape, (2, 3, 4))
        self.assertEqual(keys.dtype, np.float32)
        self.assertTrue(keys.flags.c_contiguous)
        np.testing.assert_array_equal(keys[1], 1.0)
        np.testing.assert_array_equal(values[0], 10.0)
        cache[1]["key_states"].zero_()
        np.testing.assert_array_equal(keys[1], 1.0)
        cache[1]["value_states"] = torch.zeros(1, 2, 2, 2)
        with self.assertRaisesRegex(ValueError, "matching prefix lengths"):
            provider._pack_cache(cache)


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("lerobot"),
    "requires torch and LeRobot",
)
class SmolVLASourceProviderTests(unittest.TestCase):
    def test_releasing_source_expert_keeps_prefix_model(self):
        from types import SimpleNamespace

        import torch

        class Bridge(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.vlm = torch.nn.Linear(2, 2)
                self.lm_expert = torch.nn.Module()
                self.lm_expert.layers = torch.nn.ModuleList(
                    [torch.nn.Linear(2, 2) for _ in range(3)]
                )

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.vlm_with_expert = Bridge()
                self.action_in_proj = torch.nn.Linear(2, 2)
                self.action_out_proj = torch.nn.Linear(2, 2)
                self.action_time_mlp_in = torch.nn.Linear(2, 2)
                self.action_time_mlp_out = torch.nn.Linear(2, 2)
                self.state_proj = torch.nn.Linear(2, 2)

        source = SimpleNamespace(model=Model())
        prefix = source.model.vlm_with_expert.vlm
        state = source.model.state_proj
        before = sum(p.numel() for p in source.model.parameters())
        LeRobotSmolVLACacheProvider.release_source_expert(source)
        self.assertIs(source.model.vlm_with_expert.vlm, prefix)
        self.assertIs(source.model.state_proj, state)
        self.assertEqual(source.model.vlm_with_expert.lm_expert.layers, [None] * 3)
        self.assertIsNone(source.model.action_in_proj)
        self.assertLess(sum(p.numel() for p in source.model.parameters()), before)

    def test_provider_replays_source_prefix_and_flattens_cache(self):
        import torch

        testcase = self

        class VlmWithExpert:
            def forward(self, **kwargs):
                testcase.assertTrue(
                    torch.equal(
                        kwargs["attention_mask"],
                        torch.tensor(
                            [
                                [
                                    [True, False, True],
                                    [False, False, False],
                                    [True, False, True],
                                ]
                            ]
                        ),
                    )
                )
                cache = {
                    layer: {
                        "key_states": torch.full((1, 3, 5, 64), float(layer)),
                        "value_states": torch.full((1, 3, 5, 64), float(layer + 1)),
                    }
                    for layer in range(2)
                }
                return None, cache

        class Model:
            vlm_with_expert = VlmWithExpert()

            def embed_prefix(self, images, image_masks, tokens, masks, state):
                testcase.assertTrue(masks.dtype is torch.bool)
                del images, image_masks, tokens, masks, state
                return (
                    torch.zeros((1, 3, 8)),
                    torch.tensor([[True, False, True]]),
                    torch.zeros((1, 3), dtype=torch.bool),
                )

        class SourcePolicy:
            def __init__(self):
                self.model = Model()
                self._queues = {}

            def _prepare_batch(self, batch):
                return batch

            def prepare_images(self, batch):
                del batch
                return [torch.zeros((1, 1, 2, 2))], [torch.ones((1,), dtype=torch.bool)]

            def prepare_state(self, batch):
                del batch
                return torch.zeros((1, 4))

            def reset(self):
                self._queues = {}

        batch = {
            "observation.language.tokens": torch.zeros((1, 2), dtype=torch.long),
            "observation.language.attention_mask": torch.ones((1, 2), dtype=torch.long),
        }
        provider = LeRobotSmolVLACacheProvider(SourcePolicy(), expert_layers=2)
        result = provider(batch)

        self.assertEqual(result.keys.shape, (2, 3, 320))
        self.assertEqual(result.values.shape, (2, 3, 320))
        np.testing.assert_array_equal(result.keys[1], 1.0)
        np.testing.assert_array_equal(result.values[0], 1.0)
        np.testing.assert_array_equal(result.mask, [1, 0, 1])
