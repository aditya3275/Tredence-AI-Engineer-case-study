"""
src/train.py
------------
Contains:
  - get_dataloaders         : CIFAR-10 train/test loaders
  - AdaptiveLambdaController: P-controller that hunts for target sparsity
  - train_one_epoch         : single epoch, uses controller's current lambda
  - evaluate                : test accuracy evaluation
  - run_experiment          : full adaptive training loop for one run
  - main                    : entry point
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.model import PrunableNetwork
from src.utils import (
    plot_gate_distribution,
    plot_training_curves,
    print_results_table,
    set_seed,
    verify_gradient_flow,
)

CONFIG = {
    "seed": 42,
    "batch_size": 128,
    "epochs": 50,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    # ── P-controller settings ─────────────────────────────────────────────
    "lambda_init": 0.045,  # Start at the known phase-transition edge
    "lambda_alpha": 0.0005,  # 10x smaller step — gentle nudge, not a shove
    "lambda_min": 0.0,  # hard floor — never let lambda go negative
    "lambda_max": 0.055,  # safety cap — collapses observed above 0.06
    "target_sparsity": 0.40,  # 40 % of gates below threshold is the goal
    # Warm-start: controller is frozen for the first N epochs so weights
    # can settle before pruning pressure is applied.
    "warmup_epochs": 5,
    # ─────────────────────────────────────────────────────────────────────
    "threshold": 1e-2,
    "data_dir": "./data",
    "output_dir": "./outputs",
    "checkpoint_dir": "./outputs/models",
    "num_workers": 2,
}


# ─────────────────────────────────────────────
#  Data Pipeline
# ─────────────────────────────────────────────


def get_dataloaders(batch_size: int, data_dir: str, num_workers: int):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, test_loader


# ─────────────────────────────────────────────
#  Adaptive Lambda P-Controller
# ─────────────────────────────────────────────


class AdaptiveLambdaController:
    """
    Proportional controller that adjusts lambda each epoch to chase a
    target sparsity level.

    Update rule (applied AFTER each epoch, only once warmup_epochs have passed):
        error        = target_sparsity - current_sparsity   # positive → need more pruning
        lambda_{t+1} = clip(lambda_t + alpha * error, min, max)

    Warm-start
    ----------
    For the first `warmup_epochs` epochs the controller holds lambda fixed at
    lambda_init and does NOT call the update rule.  This lets the network's
    weights and batch-norm statistics settle before any pruning pressure is
    applied, preventing the early-epoch spikes seen in Run 1.

    Intuition
    ---------
    - If sparsity is BELOW target (e.g. 10 % vs 40 %), error is positive
      → lambda increases → more regularisation pressure → gates driven toward zero.
    - If sparsity is ABOVE target (e.g. 70 % vs 40 %), error is negative
      → lambda decreases → pressure relieved → some gates can reopen.
    - Alpha = 0.0005 is deliberately tiny: the loss landscape near the
      phase-transition boundary (~0.047) is extremely steep, so micro-steps
      are needed to avoid overshooting into collapse territory.
    - Hard clamps [lambda_min=0.0, lambda_max=0.055] prevent the controller
      from going negative (which would reward density) or crossing the
      empirically observed collapse threshold (~0.06).
    """

    def __init__(
        self,
        lambda_init: float,
        alpha: float,
        target_sparsity: float,
        lambda_min: float = 0.0,
        lambda_max: float = 0.055,
        warmup_epochs: int = 5,
    ):
        self.lam = lambda_init
        self.alpha = alpha
        self.target = target_sparsity
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.warmup_epochs = warmup_epochs
        self._epoch = 0  # internal epoch counter
        self.lambda_history = [lambda_init]  # track trajectory for later plotting

    def step(self, current_sparsity: float) -> float:
        """
        Call once per epoch with the measured sparsity fraction (0.0–1.0).

        During the warm-start window (epoch <= warmup_epochs) the lambda
        value is held constant and only appended to history for plotting.
        After warm-start, the P-controller update is applied normally.

        Returns the (possibly updated) lambda value.
        """
        self._epoch += 1

        if self._epoch <= self.warmup_epochs:
            # Warm-start: freeze lambda, let weights settle
            self.lambda_history.append(self.lam)
            return self.lam

        # P-controller update
        error = self.target - current_sparsity  # negative if over-pruned
        self.lam = self.lam + self.alpha * error
        self.lam = float(np.clip(self.lam, self.lambda_min, self.lambda_max))
        self.lambda_history.append(self.lam)
        return self.lam


# ─────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────


def train_one_epoch(
    model: PrunableNetwork,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    lambda_val: float,  # current lambda from controller
    device: torch.device,
) -> tuple[float, float]:
    """
    Single training epoch.

    Lambda warmup is handled by the controller itself (it holds lambda fixed
    for the first warmup_epochs epochs). Stacking a separate warmup schedule
    here would mask the controller's true trajectory in the plots.
    """
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        logits = model(images)
        ce_loss = criterion(logits, labels)
        sparsity_loss = model.get_sparsity_loss()
        loss = ce_loss + lambda_val * sparsity_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    sparsity_pct = model.get_sparsity_level(CONFIG["threshold"]) * 100
    return avg_loss, sparsity_pct


@torch.no_grad()
def evaluate(model: PrunableNetwork, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


# ─────────────────────────────────────────────
#  Experiment
# ─────────────────────────────────────────────


def run_experiment(train_loader, test_loader, device, output_dir, checkpoint_dir):
    print(f"\n{'='*60}")
    print(
        f"  ADAPTIVE EXPERIMENT  |  target sparsity = {CONFIG['target_sparsity']*100:.0f}%"
    )
    print(
        f"  λ₀ = {CONFIG['lambda_init']}  |  α = {CONFIG['lambda_alpha']}  "
        f"|  warm-up = {CONFIG['warmup_epochs']} epochs"
    )
    print(f"{'='*60}")

    set_seed(CONFIG["seed"])

    model = PrunableNetwork(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    controller = AdaptiveLambdaController(
        lambda_init=CONFIG["lambda_init"],
        alpha=CONFIG["lambda_alpha"],
        target_sparsity=CONFIG["target_sparsity"],
        lambda_min=CONFIG["lambda_min"],
        lambda_max=CONFIG["lambda_max"],
        warmup_epochs=CONFIG["warmup_epochs"],
    )

    epoch_history = {
        "train_loss": [],
        "sparsity_pct": [],
        "lambda_vals": [],
    }

    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, CONFIG["epochs"] + 1):
        current_lambda = controller.lam  # use BEFORE step (step updates for next epoch)

        train_loss, sparsity_pct = train_one_epoch(
            model, train_loader, optimizer, criterion, current_lambda, device
        )
        scheduler.step()

        # P-controller update: measure sparsity, adjust lambda for next epoch
        # (no-op during warm-start window)
        new_lambda = controller.step(sparsity_pct / 100.0)

        epoch_history["train_loss"].append(train_loss)
        epoch_history["sparsity_pct"].append(sparsity_pct)
        epoch_history["lambda_vals"].append(current_lambda)

        # Warm-start indicator for log readability
        warmup_tag = " [warm-up]" if epoch <= CONFIG["warmup_epochs"] else ""

        # ── Checkpoint every 10 epochs ────────────────────────────────────
        if epoch % 10 == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch:02d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "lambda_val": current_lambda,
                    "sparsity_pct": sparsity_pct,
                    "train_loss": train_loss,
                },
                ckpt_path,
            )
            print(
                f"  Epoch {epoch:>3}/{CONFIG['epochs']}  |"
                f"  Loss: {train_loss:.4f}  |"
                f"  Sparsity: {sparsity_pct:.1f}%  |"
                f"  λ: {current_lambda:.5f} → {new_lambda:.5f}"
                f"  [ckpt saved]{warmup_tag}"
            )
        elif epoch == 1 or epoch % 5 == 0:
            print(
                f"  Epoch {epoch:>3}/{CONFIG['epochs']}  |"
                f"  Loss: {train_loss:.4f}  |"
                f"  Sparsity: {sparsity_pct:.1f}%  |"
                f"  λ: {current_lambda:.5f} → {new_lambda:.5f}"
                f"{warmup_tag}"
            )

    # ── Final evaluation ──────────────────────────────────────────────────
    test_acc = evaluate(model, test_loader, device)
    sparsity_lvl = model.get_sparsity_level(CONFIG["threshold"])

    print(f"\n  Final Test Accuracy : {test_acc*100:.2f}%")
    print(f"  Final Sparsity      : {sparsity_lvl*100:.2f}%")
    print(f"  Final λ             : {controller.lam:.5f}")

    # ── Save final model ──────────────────────────────────────────────────
    final_model_path = os.path.join(checkpoint_dir, "pruned_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"  [saved] Final model → {final_model_path}")

    # ── Gate distribution plot ────────────────────────────────────────────
    plot_path = os.path.join(output_dir, f"gate_dist_adaptive.png")
    plot_gate_distribution(model, round(controller.lam, 5), save_path=plot_path)

    result = {
        "lambda_val": controller.lam,
        "test_accuracy": test_acc,
        "sparsity_level": sparsity_lvl,
    }
    return result, epoch_history, controller


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"\n  Device : {device}")

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(CONFIG["data_dir"], exist_ok=True)
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    verify_gradient_flow(device)

    train_loader, test_loader = get_dataloaders(
        CONFIG["batch_size"], CONFIG["data_dir"], CONFIG["num_workers"]
    )

    result, history, controller = run_experiment(
        train_loader,
        test_loader,
        device,
        CONFIG["output_dir"],
        CONFIG["checkpoint_dir"],
    )

    # Results table (wrapped in a list to reuse existing utility)
    print_results_table([result])

    # Training curves — pass history in the dict shape utils expects
    curves_path = os.path.join(CONFIG["output_dir"], "training_curves.png")
    plot_training_curves({round(controller.lam, 5): history}, save_path=curves_path)

    # Lambda trajectory plot
    _plot_lambda_trajectory(controller.lambda_history, CONFIG["output_dir"])

    print("\n  All outputs saved to ./outputs/")
    print("  Done ✅\n")


def _plot_lambda_trajectory(lambda_history: list, output_dir: str) -> None:
    """
    Plots how lambda evolved over training epochs.
    Shaded warm-start window is highlighted so it's obvious when the
    controller was frozen vs active.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    epochs = range(len(lambda_history))

    ax.plot(epochs, lambda_history, color="#2563EB", linewidth=1.8, label="λ (adaptive)")

    # Shade the warm-start window
    warmup = CONFIG["warmup_epochs"]
    ax.axvspan(
        0, warmup, alpha=0.08, color="#F59E0B", label=f"Warm-start (epochs 1–{warmup})"
    )

    ax.axhline(
        y=CONFIG["lambda_init"],
        color="#9CA3AF",
        linestyle=":",
        linewidth=1.2,
        label=f"λ₀ = {CONFIG['lambda_init']}",
    )
    ax.axhline(
        y=0.047,
        color="#DC2626",
        linestyle="--",
        linewidth=1.2,
        label="Phase-transition boundary (~0.047)",
    )
    ax.axhline(
        y=CONFIG["lambda_max"],
        color="#7C3AED",
        linestyle="--",
        linewidth=1.2,
        label=f"Safety cap λ_max = {CONFIG['lambda_max']}",
    )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Lambda (λ)", fontsize=12)
    ax.set_title("P-Controller Lambda Trajectory", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "lambda_trajectory.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [plot] Lambda trajectory saved → {save_path}")


if __name__ == "__main__":
    main()
