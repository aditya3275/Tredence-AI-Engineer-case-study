"""
src/model.py
------------
Contains:
  - PrunableLinear : custom gated linear layer
  - PrunableNetwork: feed-forward classifier built from PrunableLinear layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that learns to prune its own weights.

    Each weight w_ij is multiplied by a scalar gate g_ij = sigmoid(s_ij),
    where s_ij is a learnable parameter (gate_score).

        pruned_weight = weight * sigmoid(gate_scores)
        output        = x @ pruned_weight.T + bias

    Gradient flow
    -------------
    Both `weight` and `gate_scores` are nn.Parameters, so autograd tracks
    gradients through the element-wise product and back into both tensors:

        d_Loss/d_weight      = d_Loss/d_output * x * gates
        d_Loss/d_gate_scores = d_Loss/d_output * x * weight * gates * (1 - gates)
                                                              └─ sigmoid derivative ─┘

    Initialization
    --------------
    gate_scores initialized to 0.5 -> sigmoid(0.5) ~ 0.62
    Gates start open enough to learn features, but close enough to zero
    that the sparsity penalty can push them to zero within training budget.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Init to 0.5: sigmoid(0.5) ~ 0.62, gates start open but prunable
        self.gate_scores = nn.Parameter(
            torch.full((out_features, in_features), fill_value=0.5)
        )

        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


class PrunableNetwork(nn.Module):
    """
    3-hidden-layer feed-forward classifier for CIFAR-10.

    Architecture
    ------------
    Input  : 3 x 32 x 32 = 3072 (flattened)
    Hidden : 512 -> 256 -> 128  (all PrunableLinear + BN + ReLU)
    Output : 10 classes (logits)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.flatten = nn.Flatten()

        self.fc1 = PrunableLinear(3 * 32 * 32, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.fc2 = PrunableLinear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)

        self.fc3 = PrunableLinear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)

        self.out = nn.Linear(128, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        return self.out(x)

    def get_sparsity_loss(self) -> torch.Tensor:
        """
        SparsityLoss = sum of sigmoid(gate_scores) over all PrunableLinear layers
        Called in training loop: TotalLoss = CrossEntropy + lambda * get_sparsity_loss()
        """
        sparsity = torch.tensor(0.0, device=next(self.parameters()).device)
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                sparsity = sparsity + torch.sigmoid(module.gate_scores).sum()
        return sparsity

    def get_sparsity_level(self, threshold: float = 1e-2) -> float:
        """
        Returns fraction of weights whose gate value is below threshold.
        A gate < threshold is considered pruned (effectively zero).
        """
        total = 0
        pruned = 0
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores).detach()
                pruned += (gates < threshold).sum().item()
                total += gates.numel()
        return pruned / total if total > 0 else 0.0
