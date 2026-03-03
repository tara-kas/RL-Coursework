"""
Default model path and save/load helpers for AlphaZero weights.
"""
import torch

DEFAULT_WEIGHTS_PATH = "weights/best.pt"


def save_weights(model: torch.nn.Module, path: str) -> None:
    """Save a model's state dict to path."""
    torch.save(model.state_dict(), path)


def load_weights(
    model: torch.nn.Module,
    path: str,
    device: torch.device | str | None = None,
) -> None:
    """Load state dict from path into model. If device is given, map loaded tensors to it."""
    state = torch.load(path, map_location=device)
    model.load_state_dict(state, strict=True)
