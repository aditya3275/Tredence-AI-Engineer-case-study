"""
tests/test_prunable.py
----------------------
Unit tests for PrunableLinear and PrunableNetwork.

Run with:
    pytest tests/ -v
"""

import pytest
import torch
import torch.nn as nn

from src.model import PrunableLinear, PrunableNetwork

# ─────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────


@pytest.fixture
def layer():
    return PrunableLinear(in_features=16, out_features=8)


@pytest.fixture
def model():
    return PrunableNetwork(num_classes=10)


@pytest.fixture
def dummy_input():
    # Simulates a batch of 4 flattened CIFAR-10 images
    return torch.randn(4, 3 * 32 * 32)


@pytest.fixture
def dummy_batch():
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    return x, y


# ─────────────────────────────────────────────
#  PrunableLinear Tests
# ─────────────────────────────────────────────


class TestPrunableLinear:

    def test_output_shape(self, layer):
        """Forward pass produces correct output shape."""
        x = torch.randn(4, 16)
        out = layer(x)
        assert out.shape == (4, 8), f"Expected (4, 8), got {out.shape}"

    def test_gate_scores_is_parameter(self, layer):
        """gate_scores must be an nn.Parameter so optimizer updates it."""
        assert isinstance(
            layer.gate_scores, nn.Parameter
        ), "gate_scores should be nn.Parameter"

    def test_gate_scores_init_value(self, layer):
        """gate_scores should be initialized to 0.5."""
        assert torch.allclose(
            layer.gate_scores, torch.full_like(layer.gate_scores, 0.5)
        ), "gate_scores should initialize to 0.5"

    def test_gates_in_zero_one(self, layer):
        """sigmoid(gate_scores) must be strictly in (0, 1)."""
        gates = torch.sigmoid(layer.gate_scores)
        assert (gates > 0).all() and (
            gates < 1
        ).all(), "All gate values should be in (0, 1)"

    def test_initial_gates_above_half(self, layer):
        """With init=0.5, gates should start above 0.5."""
        gates = torch.sigmoid(layer.gate_scores)
        assert (gates >= 0.5).all(), "With init=0.5, gates should start above 0.5."

    def test_gradient_flows_to_weight(self, layer):
        """Gradient must reach weight parameter after backward."""
        x = torch.randn(2, 16)
        out = layer(x)
        out.sum().backward()
        assert layer.weight.grad is not None, "weight.grad is None — gradient not flowing"
        assert layer.weight.grad.abs().sum().item() > 0, "weight.grad is all zeros"

    def test_gradient_flows_to_gate_scores(self, layer):
        """Gradient must reach gate_scores parameter after backward."""
        x = torch.randn(2, 16)
        out = layer(x)
        out.sum().backward()
        assert (
            layer.gate_scores.grad is not None
        ), "gate_scores.grad is None — gradient not flowing through sigmoid"
        assert (
            layer.gate_scores.grad.abs().sum().item() > 0
        ), "gate_scores.grad is all zeros"

    def test_pruned_weight_shape(self, layer):
        """pruned_weights should have same shape as weight."""
        gates = torch.sigmoid(layer.gate_scores)
        pruned_weights = layer.weight * gates
        assert pruned_weights.shape == layer.weight.shape

    def test_zero_gate_kills_weight(self):
        """
        If gate_scores → -inf, sigmoid → 0 and the weight is effectively pruned.
        Output should be determined only by bias.
        """
        layer = PrunableLinear(4, 4)
        with torch.no_grad():
            layer.gate_scores.fill_(-1e9)  # sigmoid → 0
            layer.bias.fill_(1.0)

        x = torch.randn(2, 4)
        out = layer(x)

        expected = torch.ones(2, 4)
        assert torch.allclose(
            out, expected, atol=1e-4
        ), "With gates=0, output should equal bias only"


# ─────────────────────────────────────────────
#  PrunableNetwork Tests
# ─────────────────────────────────────────────


class TestPrunableNetwork:

    def test_forward_shape(self, model):
        """Network output should be (batch, 10) logits."""
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        assert out.shape == (4, 10), f"Expected (4, 10), got {out.shape}"

    def test_sparsity_loss_is_scalar(self, model):
        """get_sparsity_loss() should return a scalar tensor."""
        loss = model.get_sparsity_loss()
        assert loss.shape == torch.Size([]), "Sparsity loss should be a scalar"

    def test_sparsity_loss_positive(self, model):
        """Sparsity loss should be positive (sum of sigmoid values)."""
        loss = model.get_sparsity_loss()
        assert loss.item() > 0, "Sparsity loss should be > 0"

    def test_sparsity_loss_differentiable(self, model, dummy_batch):
        """Sparsity loss must flow gradients to gate_scores."""
        x, y = dummy_batch
        logits = model(x)
        ce_loss = nn.CrossEntropyLoss()(logits, y)
        sp_loss = model.get_sparsity_loss()
        total = ce_loss + 0.001 * sp_loss
        total.backward()

        for module in model.modules():
            if isinstance(module, PrunableLinear):
                assert (
                    module.gate_scores.grad is not None
                ), "gate_scores.grad is None after total loss backward"

    def test_sparsity_level_initial(self, model):
        """At init (gates ≈ 0.88), sparsity level should be near 0%."""
        sparsity = model.get_sparsity_level(threshold=1e-2)
        assert sparsity < 0.01, f"Initial sparsity should be ~0%, got {sparsity*100:.2f}%"

    def test_sparsity_level_after_zeroing(self, model):
        """If all gate_scores → -inf, sparsity should be ~100%."""
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, PrunableLinear):
                    module.gate_scores.fill_(-1e9)

        sparsity = model.get_sparsity_level(threshold=1e-2)
        assert sparsity > 0.99, f"Expected ~100% sparsity, got {sparsity*100:.2f}%"

    def test_has_three_prunable_layers(self, model):
        """Network should have exactly 3 PrunableLinear layers."""
        count = sum(1 for m in model.modules() if isinstance(m, PrunableLinear))
        assert count == 3, f"Expected 3 PrunableLinear layers, found {count}"

    def test_output_layer_is_plain_linear(self, model):
        """Output layer should be nn.Linear, not PrunableLinear."""
        assert isinstance(model.out, nn.Linear) and not isinstance(
            model.out, PrunableLinear
        ), "Output layer should be plain nn.Linear"
