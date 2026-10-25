"""Fail-closed CPU/CUDA selection for the LeRobot adapter."""


def config_device(config):
    return str(getattr(config, "device", "cpu") or "cpu")


def require_torch_cuda(device, torch_module=None):
    """Require the requested source-model CUDA device to exist in PyTorch."""
    requested = str(device)
    if not requested.startswith("cuda"):
        return
    if requested != "cuda" and not (
        requested.startswith("cuda:") and requested[5:].isdecimal()
    ):
        raise ValueError("device must be cpu or cuda[:index]")
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as exc:
            raise RuntimeError(
                "CUDA VLM execution requires CUDA-enabled PyTorch"
            ) from exc
    if not torch_module.cuda.is_available():
        raise RuntimeError(
            "CUDA VLM execution requested but PyTorch CUDA is unavailable"
        )
    selected = torch_module.device(requested)
    index = selected.index
    if index is not None and index >= torch_module.cuda.device_count():
        raise RuntimeError(f"requested PyTorch CUDA device is unavailable: {requested}")


def require_native_device(device, flowedge_module, engine=None):
    if device in (None, "cpu"):
        return
    if device != "cuda" and not str(device).startswith("cuda"):
        raise ValueError("device must be cpu or cuda")
    if not getattr(flowedge_module, "cuda", False):
        raise RuntimeError(
            "CUDA requested but FlowEdge was built without FLOWEDGE_BACKEND=cuda. "
            "Release CPU wheels are not on PyPI; CUDA is a source build with nvcc."
        )
    if engine is not None and not getattr(engine, "cuda_resident", False):
        raise RuntimeError(
            "CUDA requested but this engine is running CPU kernels "
            "(device-resident load failed, 4 GB VRAM OOM, no GPU, or "
            "FLOWEDGE_CUDA_FORCE_HOST=1). Use device=cpu, or set "
            "FLOWEDGE_CUDA_REQUIRED=1 to fail at load."
        )
