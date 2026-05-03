"""
ReAct (Rectified Activations) for OOD Detection.

Implements the ReAct algorithm (Sun et al., 2021) which truncates penultimate-layer
activations at a threshold value `c` to reduce the effect of noisy high activations
on OOD inputs, thereby improving energy-based OOD detection.

Reference: https://arxiv.org/abs/2111.12797
"""

import numpy as np
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Penultimate feature extraction
# ---------------------------------------------------------------------------

def extract_penultimate_features(model, dataloader, device):
    """Extract the penultimate-layer features (input to model.fc).

    Temporarily attaches a forward hook on `model.fc` to capture the
    activations flowing into the final classification head.

    Args:
        model: A PyTorch model with a `.fc` attribute.
        dataloader: DataLoader to iterate over.
        device: Target device.

    Returns:
        np.ndarray of shape (N, feature_dim).
    """
    features = []

    def _capture_hook(module, input, output):
        # input is a tuple; input[0] is the feature tensor
        features.append(input[0].detach().cpu())

    handle = model.fc.register_forward_hook(_capture_hook)

    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Extracting features", leave=False):
            images = images.to(device)
            model(images)

    handle.remove()
    return torch.cat(features, dim=0).numpy()


# ---------------------------------------------------------------------------
# Automatic threshold calculation
# ---------------------------------------------------------------------------

def calculate_react_threshold(model, id_loader, device, percentile=90):
    """Compute the ReAct clipping threshold from ID data activations.

    Performs a forward pass of `id_loader` through `model`, captures the
    penultimate features (just before model.fc), and returns the given
    percentile of all activation values.

    Args:
        model: A PyTorch model with a `.fc` attribute.
        id_loader: DataLoader for the in-distribution dataset (e.g. CIFAR-10).
        device: Target device.
        percentile: Percentile of activations to use as threshold (default: 90).

    Returns:
        float — the threshold value `c`.
    """
    print(f"  [ReAct] Calculating threshold (percentile={percentile})...")
    feats = extract_penultimate_features(model, id_loader, device)
    threshold_c = float(np.percentile(feats, percentile))
    print(f"  [ReAct] Threshold c = {threshold_c:.4f}")
    return threshold_c


# ---------------------------------------------------------------------------
# Hook management
# ---------------------------------------------------------------------------

def apply_react_hook(model, threshold_c=1.0):
    """Register a forward pre-hook on `model.fc` to clamp activations.

    The hook truncates the input features to the final FC layer so that no
    activation exceeds `threshold_c`.  This is the core idea of ReAct.

    Args:
        model: A PyTorch model with a `.fc` attribute (e.g. ResNet).
        threshold_c: Maximum activation value.

    Returns:
        The hook handle (use it to remove the hook later).
    """
    def _react_hook(module, args):
        # args is a tuple; args[0] is the input tensor to the FC layer
        clamped = torch.clamp(args[0], max=threshold_c)
        return (clamped,)

    handle = model.fc.register_forward_pre_hook(_react_hook)
    print(f"  [ReAct] Hook registered on model.fc (threshold_c={threshold_c:.4f})")
    return handle


def remove_react_hook(handle):
    """Safely remove a previously registered hook.

    Args:
        handle: The hook handle returned by `apply_react_hook`.
    """
    if handle is not None:
        handle.remove()
        print("  [ReAct] Hook removed")
