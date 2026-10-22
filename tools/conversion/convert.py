#!/usr/bin/env python3
"""Convert a torch / HuggingFace checkpoint into FlowEdge .safetensors layout.

    python tools/conversion/convert.py <source> <out.safetensors> [--arch mamba|transformer|diffusion]
        [--component all|backbone|head] [--dtype f32|bf16]

<source> is a .safetensors, .pt, .pth or .bin (a state_dict).
FlowEdge's tensor convention mirrors HF Mamba (`backbone.*`)
+ an optional `flow.*` action head,
so converting a Mamba checkpoint is a rename + normalize pass.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from safetensors import SafetensorError
from safetensors.torch import load_file, save_file

# used to validate output.
REQUIRED_BACKBONE = [
    "backbone.embeddings.weight",
    "backbone.layers.0.mixer.A_log",
    "backbone.layers.0.mixer.conv1d.weight",
    "backbone.layers.0.mixer.x_proj.weight",
    "backbone.norm_f.weight",
]
REQUIRED_FLOW = [
    "flow.in_proj.weight",
    "flow.time_proj.weight",
    "flow.cond_proj.weight",
    "flow.out_proj.weight",
]
REQUIRED_DIFFUSION = [
    "dp.meta",
    "dp.dims",
    "dp.action_min",
    "dp.action_max",
    "dp.te1.w",
    "dp.te2.w",
    "dp.f.o.w",
]
REQUIRED_TRANSFORMER = [
    "transformer.config",
    "transformer.embeddings.token.weight",
    "transformer.embeddings.position.weight",
    "transformer.layers.0.attn.qkv.weight",
    "transformer.layers.0.mlp.fc_in.weight",
    "transformer.norm_f.weight",
]


def _load(path):
    path = str(path)
    if path.endswith(".safetensors"):
        return load_file(path)
    obj = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("state_dict", "model", "module"):  # unwrap common training wrappers
        if isinstance(obj, dict) and isinstance(obj.get(key), dict):
            obj = obj[key]
    return obj


def mamba(sd):
    """HF / state-spaces Mamba -> FlowEdge. Names align; normalize the embedding key and
    keep only backbone.*/flow.* (drops lm_head, tokenizer, optimizer state, ...)."""
    out = {}
    for k, v in sd.items():
        k = k.replace("backbone.embedding.weight", "backbone.embeddings.weight")
        if k.startswith("backbone.") or k.startswith("flow."):
            out[k] = v
    return out


def _transformer_config(config, key):
    value = config.get(key)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"unsupported Transformer config: {key} must be a positive integer")
    return value


def transformer(sd, config):
    """Map a GPT-2 decoder into FlowEdge's fixed causal contract.

    Hugging Face GPT-2 stores Conv1D projections as [in, out]; FlowEdge uses
    [out, in]. Tokenization, LM-head generation, and other model families are
    deliberately out of scope.
    """
    if config is None:
        raise ValueError("Transformer conversion requires config.json")
    if config.get("model_type") != "gpt2":
        raise ValueError(
            "Transformer conversion currently supports model_type 'gpt2' only "
            "(sshleifer/tiny-gpt2 CI fixture or openai-community/gpt2)"
        )
    if config.get("activation_function", "gelu_new") not in {"gelu", "gelu_new"}:
        raise ValueError("Transformer conversion supports GELU/GELU-new MLP activations only")

    vocab = _transformer_config(config, "vocab_size")
    d_model = _transformer_config(config, "n_embd")
    n_layers = _transformer_config(config, "n_layer")
    n_heads = _transformer_config(config, "n_head")
    max_sequence = _transformer_config(config, "n_positions")
    if d_model % n_heads:
        raise ValueError("unsupported Transformer config: n_embd must divide n_head")
    n_inner = config.get("n_inner") or (4 * d_model)
    if not isinstance(n_inner, int) or n_inner <= 0:
        raise ValueError("unsupported Transformer config: n_inner must be a positive integer")

    def get(name, expected_shape):
        value = sd.get(name)
        if value is None:
            raise ValueError(f"checkpoint is missing required tensor {name!r}")
        if tuple(value.shape) != tuple(expected_shape):
            raise ValueError(
                f"tensor {name!r} has shape {tuple(value.shape)}, expected {tuple(expected_shape)}"
            )
        return value

    out = {
        "transformer.config": torch.tensor(
            [vocab, d_model, n_layers, n_heads, max_sequence, n_inner], dtype=torch.float32
        ),
        "transformer.embeddings.token.weight": get("transformer.wte.weight", (vocab, d_model)),
        "transformer.embeddings.position.weight": get(
            "transformer.wpe.weight", (max_sequence, d_model)
        ),
    }
    for layer in range(n_layers):
        source, destination = f"transformer.h.{layer}", f"transformer.layers.{layer}"
        out[f"{destination}.ln_1.weight"] = get(f"{source}.ln_1.weight", (d_model,))
        out[f"{destination}.ln_1.bias"] = get(f"{source}.ln_1.bias", (d_model,))
        out[f"{destination}.attn.qkv.weight"] = get(
            f"{source}.attn.c_attn.weight", (d_model, 3 * d_model)
        ).transpose(0, 1).contiguous()
        out[f"{destination}.attn.qkv.bias"] = get(f"{source}.attn.c_attn.bias", (3 * d_model,))
        out[f"{destination}.attn.out_proj.weight"] = get(
            f"{source}.attn.c_proj.weight", (d_model, d_model)
        ).transpose(0, 1).contiguous()
        out[f"{destination}.attn.out_proj.bias"] = get(f"{source}.attn.c_proj.bias", (d_model,))
        out[f"{destination}.ln_2.weight"] = get(f"{source}.ln_2.weight", (d_model,))
        out[f"{destination}.ln_2.bias"] = get(f"{source}.ln_2.bias", (d_model,))
        out[f"{destination}.mlp.fc_in.weight"] = get(
            f"{source}.mlp.c_fc.weight", (d_model, n_inner)
        ).transpose(0, 1).contiguous()
        out[f"{destination}.mlp.fc_in.bias"] = get(f"{source}.mlp.c_fc.bias", (n_inner,))
        out[f"{destination}.mlp.fc_out.weight"] = get(
            f"{source}.mlp.c_proj.weight", (n_inner, d_model)
        ).transpose(0, 1).contiguous()
        out[f"{destination}.mlp.fc_out.bias"] = get(f"{source}.mlp.c_proj.bias", (d_model,))
    out["transformer.norm_f.weight"] = get("transformer.ln_f.weight", (d_model,))
    out["transformer.norm_f.bias"] = get("transformer.ln_f.bias", (d_model,))
    return out


def _config_error(message):
    raise ValueError(f"unsupported Diffusion Policy config: {message}")


def _canonical_json(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _observation_schema_hash(config, condition_dim, observation_steps):
    """Hash the exported condition contract, not model weights or normalization stats."""
    features = config.get("input_features", {})
    if not isinstance(features, dict):
        features = {}
    schema = {
        "condition_dim": condition_dim,
        "input_features": features,
        "observation_steps": observation_steps,
    }
    return "sha256:" + hashlib.sha256(_canonical_json(schema).encode()).hexdigest()


def _deployment_profile(config, tensors, condition_dim):
    action_min = tensors["dp.action_min"].detach().cpu().reshape(-1).tolist()
    action_max = tensors["dp.action_max"].detach().cpu().reshape(-1).tolist()
    action_horizon = int(config["horizon"])
    train_timesteps = int(config["num_train_timesteps"])
    return {
        "profile_version": 1,
        "model_compatibility_version": 1,
        "observation_schema_hash": _observation_schema_hash(
            config, condition_dim, int(config["n_obs_steps"])
        ),
        "action_dim": int(config["output_features"]["action"]["shape"][0]),
        "action_horizon": action_horizon,
        # The runtime returns actions after MIN_MAX un-normalization.
        "action_units": "physical",
        "normalization_type": "minmax",
        "normalization_parameters": {"min": action_min, "max": action_max},
        "solver_default": "euler",
        "solver_min_steps": 1,
        "solver_max_steps": min(train_timesteps, 4096),
    }


def _processor_action_stats(path):
    """Load MIN_MAX action statistics from a LeRobot postprocessor sidecar."""
    try:
        processor = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read processor config {path}: {exc}") from exc

    step = _unnormalizer_step(processor)
    normalization, state_path = _processor_state_contract(step)
    try:
        stats = _load(path.parent / state_path)
    except (OSError, SafetensorError, ValueError) as exc:
        raise ValueError(
            f"cannot read processor state {path.parent / state_path}: {exc}"
        ) from exc
    action_min, action_max = stats.get("action.min"), stats.get("action.max")
    if action_min is None or action_max is None:
        _config_error("processor state must contain action.min and action.max")
    return normalization, action_min, action_max


def _unnormalizer_step(processor):
    if not isinstance(processor, dict):
        _config_error("processor config must be an object")
    steps = processor.get("steps")
    if not isinstance(steps, list):
        _config_error("processor config steps must be a list")
    step = next(
        (
            candidate
            for candidate in steps
            if isinstance(candidate, dict)
            and candidate.get("registry_name") == "unnormalizer_processor"
        ),
        None,
    )
    return step or _config_error("processor config has no unnormalizer_processor step")


def _processor_state_contract(step):
    step_config = step.get("config")
    if not isinstance(step_config, dict):
        _config_error("unnormalizer_processor config must be an object")
    norm_map = step_config.get("norm_map")
    if not isinstance(norm_map, dict):
        _config_error("unnormalizer_processor norm_map must be an object")
    normalization = norm_map.get("ACTION")
    if normalization != "MIN_MAX":
        _config_error(
            f"processor ACTION normalization must be 'MIN_MAX', got {normalization!r}"
        )
    state_file = step.get("state_file")
    if not isinstance(state_file, str) or not state_file:
        _config_error("unnormalizer_processor state_file is required")
    state_path = Path(state_file)
    if state_path.is_absolute() or ".." in state_path.parts:
        _config_error("processor state_file must stay beside the processor config")
    return normalization, state_path


def _diffusion_parameters(config, processor_stats):
    if config is None:
        _config_error(
            "config.json is required (pass --config when it is not beside the model)"
        )
    checks = {
        "type": (config.get("type"), "diffusion"),
        "beta_schedule": (config.get("beta_schedule"), "squaredcos_cap_v2"),
        "prediction_type": (config.get("prediction_type"), "epsilon"),
        "use_film_scale_modulation": (config.get("use_film_scale_modulation"), True),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            _config_error(f"{name} must be {expected!r}, got {actual!r}")
    config_normalization = config.get("normalization_mapping", {}).get("ACTION")
    processor_normalization = processor_stats[0] if processor_stats else None
    if (
        config_normalization
        and processor_normalization
        and config_normalization != processor_normalization
    ):
        _config_error(
            "config and processor ACTION normalization disagree "
            f"({config_normalization!r} vs {processor_normalization!r})"
        )
    normalization = processor_normalization or config_normalization
    if normalization != "MIN_MAX":
        _config_error(f"ACTION normalization must be 'MIN_MAX', got {normalization!r}")

    down_dims = [int(value) for value in config["down_dims"]]
    if not 2 <= len(down_dims) <= 8 or any(value <= 0 for value in down_dims):
        _config_error("down_dims must contain between 2 and 8 positive dimensions")
    action_dim = int(config["output_features"]["action"]["shape"][0])
    horizon = int(config["horizon"])
    action_steps = int(config["n_action_steps"])
    observation_steps = int(config["n_obs_steps"])
    kernel = int(config["kernel_size"])
    groups = int(config["n_groups"])
    timestep_dim = int(config["diffusion_step_embed_dim"])
    train_timesteps = int(config["num_train_timesteps"])
    _validate_diffusion_geometry(
        down_dims,
        action_dim,
        horizon,
        action_steps,
        observation_steps,
        kernel,
        groups,
        timestep_dim,
        train_timesteps,
    )
    clip_range = float(config.get("clip_sample_range", 1.0))
    if not math.isfinite(clip_range) or clip_range <= 0:
        _config_error("clip_sample_range must be finite and positive")
    return (
        down_dims,
        action_dim,
        horizon,
        action_steps,
        observation_steps,
        kernel,
        groups,
        timestep_dim,
        train_timesteps,
        clip_range,
    )


def _validate_diffusion_geometry(
    down_dims,
    action_dim,
    horizon,
    action_steps,
    observation_steps,
    kernel,
    groups,
    timestep_dim,
    train_timesteps,
):
    checks = (
        (
            action_dim > 0
            and horizon > 0
            and horizon % (1 << (len(down_dims) - 1)) == 0,
            "action dimension/horizon is invalid for the configured U-Net depth",
        ),
        (0 < observation_steps <= horizon, "n_obs_steps must be between 1 and horizon"),
        (
            0 < action_steps <= horizon - observation_steps + 1,
            "n_action_steps does not fit after the observation prefix",
        ),
        (kernel > 0 and kernel % 2, "kernel_size must be positive and odd"),
        (
            timestep_dim >= 4 and timestep_dim % 2 == 0,
            "diffusion_step_embed_dim must be an even value of at least 4",
        ),
        (
            groups > 0 and not any(value % groups for value in down_dims),
            "every down dimension must be divisible by n_groups",
        ),
        (
            down_dims[0] % 8 == 0,
            "the first down dimension must be divisible by LeRobot's final 8 groups",
        ),
        (train_timesteps > 0, "num_train_timesteps must be positive"),
    )
    for valid, message in checks:
        if not valid:
            _config_error(message)


class _TensorMapper:
    """Checked checkpoint-to-runtime tensor mapping."""

    def __init__(self, state_dict):
        self.source, self.output = state_dict, {}

    def put(self, destination, source, shape):
        tensor = self.source.get(source)
        if tensor is None:
            raise ValueError(f"checkpoint is missing required tensor {source!r}")
        if tuple(tensor.shape) != tuple(shape):
            raise ValueError(
                f"tensor {source!r} has shape {tuple(tensor.shape)}, expected {tuple(shape)}"
            )
        self.output[destination] = tensor

    def conv(self, destination, source, out_channels, in_channels, size):
        self.put(
            f"{destination}.w", f"{source}.weight", (out_channels, in_channels, size)
        )
        self.put(f"{destination}.b", f"{source}.bias", (out_channels,))

    def linear(self, destination, source, out_features, in_features):
        self.put(f"{destination}.w", f"{source}.weight", (out_features, in_features))
        self.put(f"{destination}.b", f"{source}.bias", (out_features,))

    def norm(self, destination, source, channels):
        self.put(f"{destination}.w", f"{source}.weight", (channels,))
        self.put(f"{destination}.b", f"{source}.bias", (channels,))

    def residual(
        self, destination, source, in_channels, out_channels, kernel, film_width
    ):
        self.conv(
            f"{destination}.c1",
            f"{source}.conv1.block.0",
            out_channels,
            in_channels,
            kernel,
        )
        self.norm(f"{destination}.n1", f"{source}.conv1.block.1", out_channels)
        self.linear(
            f"{destination}.film",
            f"{source}.cond_encoder.1",
            2 * out_channels,
            film_width,
        )
        self.conv(
            f"{destination}.c2",
            f"{source}.conv2.block.0",
            out_channels,
            out_channels,
            kernel,
        )
        self.norm(f"{destination}.n2", f"{source}.conv2.block.1", out_channels)
        if in_channels != out_channels:
            self.conv(
                f"{destination}.res",
                f"{source}.residual_conv",
                out_channels,
                in_channels,
                1,
            )


def diffusion(sd, config, processor_stats=None):
    """LeRobot ConditionalUnet1D -> short FlowEdge `dp.*` tensor names."""
    (
        down_dims,
        action_dim,
        horizon,
        action_steps,
        observation_steps,
        kernel,
        groups,
        timestep_dim,
        train_timesteps,
        clip_range,
    ) = _diffusion_parameters(config, processor_stats)

    root = "diffusion.unet"
    first_film = f"{root}.down_modules.0.0.cond_encoder.1.weight"
    if first_film not in sd or sd[first_film].ndim != 2:
        raise ValueError(f"checkpoint is missing required tensor {first_film!r}")
    condition_dim = int(sd[first_film].shape[1]) - timestep_dim
    if condition_dim <= 0:
        _config_error("U-Net FiLM width does not contain an observation condition")

    mapper = _TensorMapper(sd)
    _map_diffusion(
        mapper, root, down_dims, action_dim, kernel, timestep_dim, condition_dim
    )
    _map_action_statistics(mapper, processor_stats, action_dim)
    out = mapper.output
    action_min, action_max = out["dp.action_min"], out["dp.action_max"]
    if not torch.isfinite(action_min).all() or not torch.isfinite(action_max).all():
        _config_error("action normalization statistics must be finite")
    if torch.any(action_max <= action_min):
        _config_error("action normalization max must be strictly greater than min")
    out["dp.dims"] = torch.tensor(down_dims, dtype=torch.float32)
    out["dp.meta"] = torch.tensor(
        [
            1,
            action_dim,
            horizon,
            action_steps,
            observation_steps,
            condition_dim,
            len(down_dims),
            kernel,
            groups,
            timestep_dim,
            train_timesteps,
            0,
            0,
            int(bool(config.get("clip_sample", True))),
            clip_range,
            0,
        ],
        dtype=torch.float32,
    )
    return out


def _map_diffusion(
    mapper, root, down_dims, action_dim, kernel, timestep_dim, condition_dim
):
    mapper.linear(
        "dp.te1", f"{root}.diffusion_step_encoder.1", timestep_dim * 4, timestep_dim
    )
    mapper.linear(
        "dp.te2", f"{root}.diffusion_step_encoder.3", timestep_dim, timestep_dim * 4
    )

    for stage, out_channels in enumerate(down_dims):
        in_channels = action_dim if stage == 0 else down_dims[stage - 1]
        mapper.residual(
            f"dp.d{stage}.r0",
            f"{root}.down_modules.{stage}.0",
            in_channels,
            out_channels,
            kernel,
            timestep_dim + condition_dim,
        )
        mapper.residual(
            f"dp.d{stage}.r1",
            f"{root}.down_modules.{stage}.1",
            out_channels,
            out_channels,
            kernel,
            timestep_dim + condition_dim,
        )
        if stage + 1 < len(down_dims):
            mapper.conv(
                f"dp.d{stage}.ds",
                f"{root}.down_modules.{stage}.2",
                out_channels,
                out_channels,
                3,
            )

    largest = down_dims[-1]
    for block in range(2):
        mapper.residual(
            f"dp.m{block}",
            f"{root}.mid_modules.{block}",
            largest,
            largest,
            kernel,
            timestep_dim + condition_dim,
        )

    for stage in range(len(down_dims) - 1):
        high = down_dims[-1 - stage]
        low = down_dims[-2 - stage]
        mapper.residual(
            f"dp.u{stage}.r0",
            f"{root}.up_modules.{stage}.0",
            2 * high,
            low,
            kernel,
            timestep_dim + condition_dim,
        )
        mapper.residual(
            f"dp.u{stage}.r1",
            f"{root}.up_modules.{stage}.1",
            low,
            low,
            kernel,
            timestep_dim + condition_dim,
        )
        # LeRobot uses equal input/output widths here, so its ConvTranspose1d
        # storage [in,out,k] is also [low,low,4].
        mapper.conv(f"dp.u{stage}.us", f"{root}.up_modules.{stage}.2", low, low, 4)

    mapper.conv(
        "dp.f.c", f"{root}.final_conv.0.block.0", down_dims[0], down_dims[0], kernel
    )
    mapper.norm("dp.f.n", f"{root}.final_conv.0.block.1", down_dims[0])
    mapper.conv("dp.f.o", f"{root}.final_conv.1", action_dim, down_dims[0], 1)


def _map_action_statistics(mapper, processor_stats, action_dim):
    if processor_stats:
        action_min, action_max = processor_stats[1:]
        if tuple(action_min.shape) != (action_dim,) or tuple(action_max.shape) != (
            action_dim,
        ):
            raise ValueError(
                "processor action statistics must have shape "
                f"({action_dim},), got {tuple(action_min.shape)} and {tuple(action_max.shape)}"
            )
        mapper.output["dp.action_min"] = action_min
        mapper.output["dp.action_max"] = action_max
    else:
        min_key = "unnormalize_outputs.buffer_action.min"
        max_key = "unnormalize_outputs.buffer_action.max"
        if min_key not in mapper.source or max_key not in mapper.source:
            min_key = "normalize_targets.buffer_action.min"
            max_key = "normalize_targets.buffer_action.max"
        mapper.put("dp.action_min", min_key, (action_dim,))
        mapper.put("dp.action_max", max_key, (action_dim,))


ARCH = {"mamba": mamba, "transformer": transformer, "diffusion": diffusion}


def _convert_dtype(name, tensor, weight_dtype):
    """Only 2-D matmul weights may be BF16 in the current runtime."""
    bf16_eligible = (
        tensor.ndim == 2
        and name != "backbone.embeddings.weight"
        and not name.endswith("mixer.A_log")
    )
    dtype = weight_dtype if bf16_eligible else torch.float32
    return tensor.to(dtype).contiguous()


def _require(sd, names, component):
    missing = [name for name in names if name not in sd]
    if missing:
        sys.exit(f"error: {component} is missing required tensors: {missing}")


def _arguments():
    ap = argparse.ArgumentParser(
        description="Convert a torch/HF checkpoint to FlowEdge .safetensors"
    )
    ap.add_argument("source")
    ap.add_argument("out")
    ap.add_argument("--arch", choices=sorted(ARCH), default="mamba")
    ap.add_argument("--component", choices=("all", "backbone", "head"), default="all")
    ap.add_argument("--dtype", choices=("f32", "bf16"), default="f32")
    ap.add_argument(
        "--config",
        help="architecture config.json (auto-detected beside source; required for Transformer)",
    )
    ap.add_argument(
        "--processor",
        help="LeRobot policy_postprocessor.json (auto-detected for a source directory)",
    )
    return ap.parse_args()


def _diffusion_inputs(args, source, default_config):
    if args.component == "backbone":
        sys.exit("error: diffusion conversion exports an action head, not a backbone")
    config_path = Path(args.config) if args.config else default_config
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        sys.exit(f"error: cannot read diffusion config {config_path}: {exc}")
    processor_path = Path(args.processor) if args.processor else None
    candidate = source / "policy_postprocessor.json"
    if processor_path is None and source.is_dir() and candidate.is_file():
        processor_path = candidate
    try:
        return config, _processor_action_stats(
            processor_path
        ) if processor_path else None
    except (TypeError, ValueError) as exc:
        sys.exit(f"error: {exc}")


def _component(state_dict, component):
    prefixes = {
        "backbone": ("backbone.", "transformer."),
        "head": ("flow.", "dp."),
    }
    return (
        state_dict
        if component == "all"
        else {
            key: value
            for key, value in state_dict.items()
            if key.startswith(prefixes[component])
        }
    )


def main():
    args = _arguments()

    source = Path(args.source)
    if source.is_dir():
        model_source = source / "model.safetensors"
        if not model_source.is_file():
            model_source = source / "pytorch_model.bin"
        default_config = source / "config.json"
    else:
        model_source = source
        default_config = source.with_name("config.json")
    config = None
    processor_stats = None
    if args.arch == "diffusion":
        config, processor_stats = _diffusion_inputs(args, source, default_config)
    if args.arch == "transformer":
        if args.component == "head":
            sys.exit("error: Transformer conversion exports a backbone, not an action head")
        config_path = Path(args.config) if args.config else default_config
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except OSError as exc:
            sys.exit(f"error: cannot read Transformer config {config_path}: {exc}")
    try:
        loaded = _load(model_source)
        sd = (
            diffusion(loaded, config, processor_stats)
            if args.arch == "diffusion"
            else transformer(loaded, config)
            if args.arch == "transformer"
            else mamba(loaded)
        )
    except (KeyError, TypeError, ValueError) as exc:
        sys.exit(f"error: {exc}")
    sd = _component(sd, args.component)
    if not sd:
        sys.exit("error: no tensors found for the selected architecture and component")

    dt = torch.float32 if args.dtype == "f32" else torch.bfloat16
    sd = {key: _convert_dtype(key, value, dt) for key, value in sd.items()}

    if args.arch == "mamba" and args.component != "head":
        _require(sd, REQUIRED_BACKBONE, "backbone")
    if args.arch == "transformer":
        _require(sd, REQUIRED_TRANSFORMER, "Transformer backbone")
    if (
        args.arch == "diffusion"
        or args.component == "head"
        or any(key.startswith("flow.") for key in sd)
    ):
        required = REQUIRED_DIFFUSION if args.arch == "diffusion" else REQUIRED_FLOW
        _require(sd, required, f"{args.arch} head")

    metadata = {}
    if args.arch == "diffusion":
        digest = hashlib.sha256()
        with model_source.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        metadata["flowedge.source_sha256"] = digest.hexdigest()
        metadata["flowedge.source_config_sha256"] = hashlib.sha256(
            _canonical_json(config).encode("utf-8")
        ).hexdigest()
        film_width = int(sd["dp.d0.r0.film.w"].shape[1])
        timestep_dim = int(config["diffusion_step_embed_dim"])
        condition_dim = film_width - timestep_dim
        metadata["flowedge.deployment_profile"] = _canonical_json(
            _deployment_profile(config, sd, condition_dim)
        )
    save_file(sd, args.out, metadata=metadata)
    layer_prefix = "transformer.layers." if args.arch == "transformer" else "backbone.layers."
    layer_ids = [int(key.split(".")[2]) for key in sd if key.startswith(layer_prefix)]
    layers = 1 + max(layer_ids) if layer_ids else 0
    head = (
        "diffusion"
        if any(k.startswith("dp.") for k in sd)
        else ("flow" if any(k.startswith("flow.") for k in sd) else "none")
    )
    print(
        f"wrote {args.out}: {len(sd)} tensors, {layers} layers, head={head}, dtype={args.dtype}"
    )


if __name__ == "__main__":
    main()
