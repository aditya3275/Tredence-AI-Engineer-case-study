"""
src/evaluate.py
---------------
Standalone inference script for saved PrunableNetwork checkpoints.

Usage
-----
  # Evaluate the final pruned model:
  python -m src.evaluate --model outputs/models/pruned_model.pt

  # Evaluate a mid-training checkpoint:
  python -m src.evaluate --model outputs/models/checkpoint_epoch_30.pt

  # Override threshold for sparsity measurement:
  python -m src.evaluate --model outputs/models/pruned_model.pt --threshold 0.05

Output (example)
----------------
  ┌──────────────────────────────────────────────────────┐
  │  Model   : outputs/models/pruned_model.pt            │
  │  Device  : mps                                       │
  ├──────────────────────────────────────────────────────┤
  │  Accuracy        :  58.34 %                          │
  │  Sparsity        :  41.20 %                          │
  │  Remaining Params:  1,020,313  /  1,736,704  gated   │
  │  Pruned Params   :    716,391  (41.20%)               │
  └──────────────────────────────────────────────────────┘
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.model import PrunableLinear, PrunableNetwork

# ─────────────────────────────────────────────
#  Constants — match train.py values exactly
# ─────────────────────────────────────────────

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

TOTAL_GATED_WEIGHTS = (3072 * 512) + (512 * 256) + (256 * 128)  # 1,736,704


# ─────────────────────────────────────────────
#  Loader
# ─────────────────────────────────────────────


def get_test_loader(
    data_dir: str = "./data", batch_size: int = 256, num_workers: int = 2
) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )


# ─────────────────────────────────────────────
#  Load model from file
# ─────────────────────────────────────────────


def load_model(path: str, device: torch.device) -> tuple[PrunableNetwork, dict]:
    """
    Loads a PrunableNetwork from a .pt file.

    Handles two save formats:
      1. Full checkpoint dict  — saved by the training loop every 10 epochs
         Keys: model_state, epoch, lambda_val, sparsity_pct, train_loss, optimizer_state
      2. Bare state_dict       — saved by torch.save(model.state_dict(), path)
         This is what pruned_model.pt uses.

    Returns the model and a metadata dict (empty if bare state_dict).
    """
    raw = torch.load(path, map_location=device, weights_only=False)

    model = PrunableNetwork(num_classes=10).to(device)

    if isinstance(raw, dict) and "model_state" in raw:
        # Full checkpoint
        model.load_state_dict(raw["model_state"])
        meta = {k: v for k, v in raw.items() if k not in ("model_state", "optimizer_state")}
    else:
        # Bare state dict (pruned_model.pt)
        model.load_state_dict(raw)
        meta = {}

    model.eval()
    return model, meta


# ─────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────


@torch.no_grad()
def compute_accuracy(
    model: PrunableNetwork, loader: DataLoader, device: torch.device
) -> float:
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def compute_remaining_params(model: PrunableNetwork, threshold: float) -> tuple[int, int]:
    """
    Returns (remaining_gated, total_gated) counts.
    A weight is 'remaining' if its gate >= threshold (not pruned).
    """
    total = 0
    pruned = 0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores).detach()
            pruned += int((gates < threshold).sum().item())
            total += gates.numel()
    remaining = total - pruned
    return remaining, total


# ─────────────────────────────────────────────
#  Pretty Print Summary
# ─────────────────────────────────────────────


def print_summary(
    model_path: str,
    device: torch.device,
    accuracy: float,
    sparsity: float,
    remaining: int,
    total_gated: int,
    meta: dict,
    threshold: float,
) -> None:
    width = 56
    bar = "─" * width

    pruned = total_gated - remaining
    acc_str = f"{accuracy * 100:.2f} %"
    sparse_str = f"{sparsity * 100:.2f} %"
    rem_str = f"{remaining:,}  /  {total_gated:,}  gated"
    pru_str = f"{pruned:,}  ({sparsity*100:.2f}%)"

    print(f"\n  ┌{bar}┐")
    print(f"  │  {'Model':<14}: {os.path.basename(model_path):<{width-18}}│")
    print(f"  │  {'Device':<14}: {str(device):<{width-18}}│")
    print(f"  │  {'Threshold':<14}: {threshold:<{width-18}}│")

    if meta:
        ep = meta.get("epoch", "—")
        lam = meta.get("lambda_val", "—")
        print(f"  │  {'Checkpoint Ep':<14}: {str(ep):<{width-18}}│")
        print(
            f"  │  {'Lambda at ckpt':<14}: {str(round(lam, 5)) if isinstance(lam, float) else lam:<{width-18}}│"
        )

    print(f"  ├{bar}┤")
    print(f"  │  {'Accuracy':<14}: {acc_str:<{width-18}}│")
    print(f"  │  {'Sparsity':<14}: {sparse_str:<{width-18}}│")
    print(f"  │  {'Remaining':<14}: {rem_str:<{width-18}}│")
    print(f"  │  {'Pruned':<14}: {pru_str:<{width-18}}│")
    print(f"  └{bar}┘\n")


# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved PrunableNetwork checkpoint."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/models/pruned_model.pt",
        help="Path to .pt file (state_dict or full checkpoint)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="CIFAR-10 data directory (auto-downloaded if missing)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-2,
        help="Gate threshold below which a weight is considered pruned (default: 0.01)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for inference (default: 256)",
    )
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # ── Validate path ─────────────────────────────────────────────────────
    if not os.path.exists(args.model):
        print(f"\n  [ERROR] Model file not found: {args.model}")
        print(f"  Run 'make train' first to generate checkpoints.\n")
        raise SystemExit(1)

    print(f"\n  Loading  : {args.model}")
    print(f"  Device   : {device}")

    # ── Load & evaluate ───────────────────────────────────────────────────
    model, meta = load_model(args.model, device)

    test_loader = get_test_loader(data_dir=args.data_dir, batch_size=args.batch_size)

    print("  Running inference on CIFAR-10 test set (10,000 samples)...")
    accuracy = compute_accuracy(model, test_loader, device)
    sparsity = model.get_sparsity_level(threshold=args.threshold)
    remaining, total_gated = compute_remaining_params(model, threshold=args.threshold)

    print_summary(
        model_path=args.model,
        device=device,
        accuracy=accuracy,
        sparsity=sparsity,
        remaining=remaining,
        total_gated=total_gated,
        meta=meta,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
