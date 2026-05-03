"""
ReAct (Rectified Activations) for OOD Detection.

Implements the ReAct algorithm (Sun et al., 2021) which truncates penultimate-layer
activations at a threshold value `c` to reduce the effect of noisy high activations
on OOD inputs, thereby improving energy-based OOD detection.

Reference: https://arxiv.org/abs/2111.12797
"""

import torch


def apply_react_hook(model, threshold_c=1.0):
    """Register a forward pre-hook on `model.fc` to clamp activations.

    The hook truncates the input features to the final FC layer so that no
    activation exceeds `threshold_c`.  This is the core idea of ReAct.

    Args:
        model: A PyTorch model with a `.fc` attribute (e.g. ResNet).
        threshold_c: Maximum activation value (default: 1.0).

    Returns:
        The hook handle (use it to remove the hook later).
    """
    def _react_hook(module, args):
        # args is a tuple; args[0] is the input tensor to the FC layer
        clamped = torch.clamp(args[0], max=threshold_c)
        return (clamped,)

    handle = model.fc.register_forward_pre_hook(_react_hook)
    print(f"  [ReAct] Hook registered on model.fc (threshold_c={threshold_c})")
    return handle


def remove_react_hook(handle):
    """Safely remove a previously registered hook.

    Args:
        handle: The hook handle returned by `apply_react_hook`.
    """
    if handle is not None:
        handle.remove()
        print("  [ReAct] Hook removed")
