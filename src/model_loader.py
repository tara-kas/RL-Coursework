"""
Default model path and save/load helpers for AlphaZero weights.
"""
import torch

DEFAULT_WEIGHTS_PATH = "weights/best.pt"

def _unwrap_compiled(model: torch.nn.Module) -> torch.nn.Module:
    """
    torch.compile() may wrap modules such that parameters live under model._orig_mod.
    We always save/load the underlying module weights to keep checkpoints compatible
    across compiled and non-compiled runs.
    """
    orig = getattr(model, "_orig_mod", None)
    return orig if isinstance(orig, torch.nn.Module) else model


def save_weights(model: torch.nn.Module, path: str) -> None:
    """Save a model's state dict to path."""
    model_to_save = _unwrap_compiled(model)
    torch.save(model_to_save.state_dict(), path)


def load_weights(
    model: torch.nn.Module,
    path: str,
    device: torch.device | str | None = None,
) -> None:
    """Load state dict from path into model. If device is given, map loaded tensors to it."""
    state = torch.load(path, map_location=device)
    model_to_load = _unwrap_compiled(model)

    # Backwards compatibility: older checkpoints might have been saved from a compiled wrapper,
    # producing keys like "_orig_mod.xxx". Strip that prefix if present.
    if isinstance(state, dict) and state:
        any_orig_mod = any(isinstance(k, str) and k.startswith("_orig_mod.") for k in state.keys())
        if any_orig_mod:
            state = {
                (k[len("_orig_mod.") :] if isinstance(k, str) and k.startswith("_orig_mod.") else k): v
                for k, v in state.items()
            }

    model_to_load.load_state_dict(state, strict=True)
