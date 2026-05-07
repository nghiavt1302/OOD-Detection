"""
Model loader for Architecture-Driven Ablation Study.

Supports three pretrained CIFAR-10 architectures from a single stable repo:
  - resnet20   → chenyaofo/pytorch-cifar-models (Base/Edge)
  - resnet56   → chenyaofo/pytorch-cifar-models (Deep)
  - vgg16_bn   → chenyaofo/pytorch-cifar-models (Wide/Large Capacity)
"""

import torch

# Thay wideresnet bằng vgg16_bn
SUPPORTED_MODELS = ["resnet20", "resnet56", "vgg16_bn"]


def load_pretrained_model(model_name, device=None):
    """Load a pretrained CIFAR-10 model from PyTorch Hub.

    Args:
        model_name: One of "resnet20", "resnet56", "vgg16_bn".
        device: Target device (auto-detected if None).

    Returns:
        Tuple of (model, device).
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model_name='{model_name}'. "
            f"Choose from {SUPPORTED_MODELS}"
        )

    # --- Auto-detect device --------------------------------------------------
    if device is None:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    print(f"  Device: {device}")

    # --- Load from PyTorch Hub -----------------------------------------------
    hub_repo = "chenyaofo/pytorch-cifar-models"
    hub_model_name = f"cifar10_{model_name}"
    
    print(f"  Downloading pretrained {hub_model_name} from {hub_repo}...")
    model = torch.hub.load(
        hub_repo,
        hub_model_name,
        pretrained=True,
    )

    # --- Move to device, freeze gradients ------------------------------------
    model = model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded pretrained {model_name} ({total_params:,} params) — eval mode, gradients frozen")

    return model, device


if __name__ == "__main__":
    for name in SUPPORTED_MODELS:
        print(f"\n{'='*50}")
        print(f"Testing: {name}")
        print(f"{'='*50}")
        model, device = load_pretrained_model(model_name=name)
        # Quick forward pass sanity check
        dummy = torch.randn(1, 3, 32, 32).to(device)
        with torch.no_grad():
            out = model(dummy)
        print(f"  Output shape: {out.shape}")