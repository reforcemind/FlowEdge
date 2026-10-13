"""LeRobot Diffusion Policy action contract over the FlowEdge Python API.

Observation encoding and LeRobot feature preprocessing stay outside this package. The adapter
owns only the deployment boundary: validating a flattened condition, invoking FlowEdge, and
slicing the action horizon according to LeRobot's metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class DiffusionEngine(Protocol):
    """Minimum engine surface required by the adapter."""

    @property
    def diffusion_metadata(self) -> dict[str, Any]: ...

    def sample_diffusion(
        self,
        condition: np.ndarray,
        noise: np.ndarray,
        steps: int,
        scheduler: str,
        seed: int,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class DiffusionActionContract:
    """Fixed-shape action boundary shared with a converted FlowEdge checkpoint."""

    condition_dim: int
    action_dim: int
    horizon: int
    action_steps: int
    observation_steps: int

    @property
    def action_start(self) -> int:
        """Return the first horizon element consumed by LeRobot."""
        return self.observation_steps - 1

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> DiffusionActionContract:
        required = (
            "condition_dim",
            "action_dim",
            "horizon",
            "action_steps",
            "observation_steps",
        )
        missing = [key for key in required if key not in metadata]
        if missing:
            raise ValueError(f"missing diffusion metadata: {', '.join(missing)}")
        contract = cls(*(int(metadata[key]) for key in required))
        if (
            contract.action_start < 0
            or contract.action_start + contract.action_steps > contract.horizon
        ):
            raise ValueError("LeRobot action slice is outside the diffusion horizon")
        if (
            min(
                contract.condition_dim,
                contract.action_dim,
                contract.horizon,
                contract.action_steps,
            )
            <= 0
        ):
            raise ValueError("diffusion dimensions must be positive")
        if contract.observation_steps <= 0:
            raise ValueError("observation_steps must be positive")
        return contract


class FlowEdgeDiffusionPolicy:
    """Deployment adapter for a converted LeRobot Diffusion Policy checkpoint.

    The returned action chunk is already in dataset action units. LeRobot's image/state encoder,
    feature normalization, and robot safety limits remain the caller's responsibility.
    """

    def __init__(self, engine: DiffusionEngine):
        self._engine = engine
        self.contract = DiffusionActionContract.from_metadata(engine.diffusion_metadata)
        self._horizon = np.empty(
            (self.contract.horizon, self.contract.action_dim), dtype=np.float32
        )
        self._sample_diffusion_into = getattr(engine, "sample_diffusion_into", None)
        self._sample_diffusion_dlpack_into = getattr(
            engine, "sample_diffusion_dlpack_into", None
        )
        self._sample_diffusion = engine.sample_diffusion

    @classmethod
    def from_checkpoint(
        cls, path: str, *, threads: int | None = None, device: str = "cpu"
    ) -> FlowEdgeDiffusionPolicy:
        """Load a converted checkpoint without making FlowEdge a LeRobot dependency."""
        import flowedge

        from .device import require_native_device

        require_native_device(device, flowedge)
        engine = flowedge.Engine(path, threads=threads)
        require_native_device(device, flowedge, engine=engine)
        return cls(engine)

    @property
    def metadata(self) -> DiffusionActionContract:
        return self.contract

    @property
    def cuda_resident(self) -> bool:
        return bool(getattr(self._engine, "cuda_resident", False))

    @property
    def action_dim(self) -> int:
        return self.contract.action_dim

    @property
    def action_steps(self) -> int:
        return self.contract.action_steps

    @property
    def supports_cuda_tensors(self) -> bool:
        return self._sample_diffusion_dlpack_into is not None

    def reset(self) -> None:
        """Reset the deployment adapter; the current diffusion engine is stateless per call."""

    def predict_action_chunk(
        self,
        condition: np.ndarray,
        noise: np.ndarray,
        *,
        steps: int = 10,
        scheduler: str = "ddim",
        seed: int = 0,
    ) -> np.ndarray:
        """Return LeRobot's executable action chunk from a flattened condition."""
        actions = np.empty(
            (self.contract.action_steps, self.contract.action_dim), dtype=np.float32
        )
        self.predict_action_chunk_into(
            condition, noise, actions, steps=steps, scheduler=scheduler, seed=seed
        )
        return actions

    def predict_action_chunk_into(
        self,
        condition: np.ndarray,
        noise: np.ndarray,
        output: np.ndarray,
        *,
        steps: int = 10,
        scheduler: str = "ddim",
        seed: int = 0,
    ) -> np.ndarray:
        """Write the executable action chunk into caller-owned float32 storage."""
        expected_actions = (self.contract.action_steps, self.contract.action_dim)
        self._validate_output(output, expected_actions)
        self._sample_horizon(condition, noise, steps, scheduler, seed)
        start = self.contract.action_start
        stop = start + self.contract.action_steps
        np.copyto(output, self._horizon[start:stop])
        return output

    def predict_action_chunk_dlpack_into(
        self,
        condition: Any,
        noise: Any,
        output: Any,
        *,
        steps: int = 10,
        scheduler: str = "ddim",
        seed: int = 0,
    ) -> Any:
        """Write an action chunk without leaving caller-owned CUDA storage."""
        if self._sample_diffusion_dlpack_into is None:
            raise RuntimeError(
                "the loaded FlowEdge module has no CUDA DLPack entry point"
            )
        expected_noise = (self.contract.horizon, self.contract.action_dim)
        expected_output = (self.contract.action_steps, self.contract.action_dim)
        if tuple(getattr(condition, "shape", ())) not in {
            (self.contract.condition_dim,),
            (1, self.contract.condition_dim),
        }:
            raise ValueError(
                f"condition must contain {self.contract.condition_dim} float32 values"
            )
        if tuple(getattr(noise, "shape", ())) != expected_noise:
            raise ValueError(f"noise must have shape {expected_noise}")
        if tuple(getattr(output, "shape", ())) != expected_output:
            raise ValueError(f"output must have shape {expected_output}")
        if steps <= 0:
            raise ValueError("steps must be positive")
        if scheduler not in {"ddim", "ddpm"}:
            raise ValueError("scheduler must be one of: ddim, ddpm")
        self._sample_diffusion_dlpack_into(
            condition,
            noise,
            output,
            steps,
            scheduler,
            int(seed),
            self.contract.action_start,
        )
        return output

    def select_action(
        self,
        condition: np.ndarray,
        noise: np.ndarray,
        *,
        steps: int = 10,
        scheduler: str = "ddim",
        seed: int = 0,
    ) -> np.ndarray:
        """Return the first action in the executable chunk for single-step loops."""
        action = np.empty(self.contract.action_dim, dtype=np.float32)
        return self.select_action_into(
            condition, noise, action, steps=steps, scheduler=scheduler, seed=seed
        )

    def select_action_into(
        self,
        condition: np.ndarray,
        noise: np.ndarray,
        output: np.ndarray,
        *,
        steps: int = 10,
        scheduler: str = "ddim",
        seed: int = 0,
    ) -> np.ndarray:
        """Write the first action into caller-owned float32 storage."""
        self._validate_output(output, (self.contract.action_dim,))
        self._sample_horizon(condition, noise, steps, scheduler, seed)
        np.copyto(output, self._horizon[self.contract.action_start])
        return output

    @staticmethod
    def _validate_output(output: np.ndarray, shape: tuple[int, ...]) -> None:
        if (
            not isinstance(output, np.ndarray)
            or output.shape != shape
            or output.dtype != np.float32
            or not output.flags.c_contiguous
        ):
            raise ValueError(
                f"output must be a C-contiguous float32 array with shape {shape}"
            )

    def _sample_horizon(
        self,
        condition: np.ndarray,
        noise: np.ndarray,
        steps: int,
        scheduler: str,
        seed: int,
    ) -> None:
        condition = np.ascontiguousarray(condition, dtype=np.float32).reshape(-1)
        noise = np.ascontiguousarray(noise, dtype=np.float32)
        expected_noise = self._horizon.shape
        if condition.size != self.contract.condition_dim:
            raise ValueError(
                f"condition must contain {self.contract.condition_dim} float32 values"
            )
        if noise.shape != expected_noise:
            raise ValueError(f"noise must have shape {expected_noise}")
        if steps <= 0:
            raise ValueError("steps must be positive")
        if scheduler not in {"ddim", "ddpm"}:
            raise ValueError("scheduler must be one of: ddim, ddpm")
        if self._sample_diffusion_into is not None:
            self._sample_diffusion_into(
                condition, noise, self._horizon, steps, scheduler, int(seed)
            )
            return
        horizon = np.asarray(
            self._sample_diffusion(condition, noise, steps, scheduler, int(seed)),
            dtype=np.float32,
        )
        if horizon.shape != self._horizon.shape:
            raise ValueError(
                f"FlowEdge returned horizon with shape {horizon.shape}, expected {self._horizon.shape}"
            )
        np.copyto(self._horizon, horizon)
