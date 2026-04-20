"""
src/utils.py
------------
Contains:
  - verify_gradient_flow : proves gradients reach both weight and gate_scores
  - set_seed             : full reproducibility across all λ runs
  - plot_gate_distribution: log-scale histogram of final gate values
  - plot_training_curves  : loss + sparsity % over epochs per λ
  - print_results_table   : formatted summary of λ sweep results
"""

import random

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

from src.model import PrunableLinear, PrunableNetwork

# ─────────────────────────────────────────────
#  Reproducibility
# ─────────────────────────────────────────────


def set_seed(seed: int = 42) -> None:
    """
    Seeds Python, NumPy, and PyTorch (CPU + CUDA) for full reproducibility.
    Call this before each λ run so results are comparable and re-runnable.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensures deterministic ops where available (slight perf cost, worth it)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────
#  Gradient Flow Verification
# ─────────────────────────────────────────────


def verify_gradient_flow(device: torch.device) -> None:
    """
    Unit-test style check that proves gradients flow correctly through
    PrunableLinear into BOTH `weight` and `gate_scores`.

    Method
    ------
    1. Build a minimal PrunableLinear(4, 4).
    2. Run a forward pass with a dummy input.
    3. Backpropagate a scalar loss.
    4. Assert that .grad is not None AND not all-zero for both parameters.

    Why this matters
    ----------------
    If gate_scores had been created with torch.tensor() instead of
    nn.Parameter(), or if we had used torch.no_grad() anywhere in forward(),
    the gradient would be None here — and the gates would never learn.
    """
    print("\n" + "=" * 55)
    print("  GRADIENT FLOW VERIFICATION")
    print("=" * 55)

    layer = PrunableLinear(in_features=4, out_features=4).to(device)

    # Dummy batch: 2 samples, 4 features
    x = torch.randn(2, 4, device=device)
    out = layer(x)
    loss = out.sum()  # trivial scalar loss
    loss.backward()

    # ── weight ────────────────────────────────
    assert (
        layer.weight.grad is not None
    ), "FAIL: weight.grad is None — gradient not flowing into weight"
    assert layer.weight.grad.abs().sum().item() > 0, "FAIL: weight.grad is all zeros"

    # ── gate_scores ───────────────────────────
    assert (
        layer.gate_scores.grad is not None
    ), "FAIL: gate_scores.grad is None — gradient not flowing into gates"
    assert (
        layer.gate_scores.grad.abs().sum().item() > 0
    ), "FAIL: gate_scores.grad is all zeros"

    print(
        f"  weight.grad      : shape {list(layer.weight.grad.shape)}"
        f"  |  norm = {layer.weight.grad.norm().item():.4f}"
    )
    print(
        f"  gate_scores.grad : shape {list(layer.gate_scores.grad.shape)}"
        f"  |  norm = {layer.gate_scores.grad.norm().item():.4f}"
    )
    print("\n  ✅  Both weight and gate_scores receive gradients correctly.")
    print("=" * 55 + "\n")


# ─────────────────────────────────────────────
#  Gate Distribution Plot
# ─────────────────────────────────────────────


def plot_gate_distribution(
    model: PrunableNetwork, lambda_val: float, save_path: str = "gate_distribution.png"
) -> None:
    """
    Plots a log-scale histogram of ALL gate values across every PrunableLinear
    layer in the model after training.

    A successful pruning run shows:
      - A large spike at 0   (pruned / dead weights)
      - A second cluster away from 0 (surviving weights)

    Log-scale y-axis is used so the surviving cluster is visible even when
    the spike at zero dominates by orders of magnitude.
    """
    all_gates = []

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores).detach().cpu().numpy()
            all_gates.append(gates.flatten())

    all_gates = np.concatenate(all_gates)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        all_gates,
        bins=100,
        color="#2563EB",
        edgecolor="white",
        linewidth=0.3,
        log=True,  # log-scale y-axis
    )

    ax.axvline(
        x=1e-2,
        color="#DC2626",
        linestyle="--",
        linewidth=1.4,
        label="Prune threshold (1e-2)",
    )

    # Annotation: % pruned
    pruned_pct = (all_gates < 1e-2).mean() * 100
    ax.text(
        0.05,
        0.92,
        f"Sparsity: {pruned_pct:.1f}% of gates below threshold",
        transform=ax.transAxes,
        fontsize=10,
        color="#DC2626",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="#DC2626", alpha=0.8
        ),
    )

    ax.set_xlabel("Gate Value  g = σ(s)", fontsize=12)
    ax.set_ylabel("Count (log scale)", fontsize=12)
    ax.set_title(f"Gate Value Distribution  |  λ = {lambda_val}", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(-0.02, 1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [plot] Gate distribution saved → {save_path}")


# ─────────────────────────────────────────────
#  Training Curves Plot
# ─────────────────────────────────────────────


def plot_training_curves(history: dict, save_path: str = "training_curves.png") -> None:
    """
    Plots two subplots per λ value:
      Left  — Total training loss over epochs
      Right — Sparsity level (%) over epochs

    Args:
        history : dict shaped as
            {
              lambda_val (float) : {
                  "train_loss"    : [float, ...],   # per epoch
                  "sparsity_pct"  : [float, ...],   # per epoch
              }
            }
        save_path : output file path
    """
    lambdas = sorted(history.keys())
    colors = ["#2563EB", "#16A34A", "#DC2626"]  # blue, green, red

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for lam, color in zip(lambdas, colors):
        epochs = range(1, len(history[lam]["train_loss"]) + 1)
        train_loss = history[lam]["train_loss"]
        sparsity_pct = history[lam]["sparsity_pct"]

        axes[0].plot(epochs, train_loss, color=color, label=f"λ={lam}", linewidth=1.8)
        axes[1].plot(epochs, sparsity_pct, color=color, label=f"λ={lam}", linewidth=1.8)

    # ── Left: Loss ────────────────────────────
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Total Loss", fontsize=12)
    axes[0].set_title("Training Loss vs Epoch", fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # ── Right: Sparsity ───────────────────────
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Sparsity (%)", fontsize=12)
    axes[1].set_title("Sparsity Level vs Epoch", fontsize=13)
    axes[1].yaxis.set_major_formatter(ticker.PercentFormatter())
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        "Self-Pruning Network — Training Dynamics",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Training curves saved  → {save_path}")


# ─────────────────────────────────────────────
#  Results Table
# ─────────────────────────────────────────────


def print_results_table(results: list[dict]) -> None:
    """
    Prints a formatted summary table to stdout.

    Args:
        results : list of dicts, each with keys:
            lambda_val, test_accuracy, sparsity_level
    """
    header = f"{'Lambda':<12} {'Test Accuracy':>15} {'Sparsity Level':>16}"
    sep = "─" * len(header)

    print("\n" + sep)
    print(header)
    print(sep)

    for r in results:
        print(
            f"{r['lambda_val']:<12} "
            f"{r['test_accuracy']*100:>13.2f}%  "
            f"{r['sparsity_level']*100:>14.2f}%"
        )

    print(sep + "\n")
