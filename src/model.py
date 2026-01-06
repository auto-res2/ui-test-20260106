"""Model architectures & optimisation utilities."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

# -----------------------------------------------------------------------------
# Synthetic problem primitives
# -----------------------------------------------------------------------------

def make_problem(d: int, m: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(m, d)).astype(np.float32)
    b = rng.normal(size=(m,)).astype(np.float32)
    return torch.from_numpy(A), torch.from_numpy(b)


def f_smooth(x: torch.Tensor, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # noqa: D401
    z = (A @ x) + b
    return torch.logsumexp(z, dim=0)


def grad_f_smooth(x: torch.Tensor, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # noqa: D401
    z = (A @ x) + b
    w = torch.softmax(z, dim=0)
    return A.t() @ w


def project_ball(v: torch.Tensor, R: float) -> torch.Tensor:  # noqa: D401
    n = torch.norm(v)
    if n <= R:
        return v
    return v * (R / n)

# -----------------------------------------------------------------------------
# α-network
# -----------------------------------------------------------------------------

class AdaptiveAlphaNet(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 8, min_val: float = 0.0, max_val: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.register_buffer("min_val", torch.tensor(float(min_val)))
        self.register_buffer("max_val", torch.tensor(float(max_val)))
        nn.init.uniform_(self.fc1.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc2.weight, -0.1, 0.1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # noqa: D401
        h = F.relu(self.fc1(z))
        a = F.softplus(self.fc2(h))
        a = self.min_val + (self.max_val - self.min_val) * torch.sigmoid(a)
        return a.squeeze()

# -----------------------------------------------------------------------------
# Core optimisation routine (single problem instance)
# -----------------------------------------------------------------------------

def run_moreau_yosida(cfg: DictConfig, net: AdaptiveAlphaNet | None, seed: int, device: torch.device):
    """Run T iterations for the pseudo-random *seed* task.

    Returns
    -------
    final_subopt : torch.Tensor (scalar, differentiable if *net* is trainable)
    trajectory   : List[float] (detached for logging)
    """
    add_p = cfg.training.additional_params
    T = int(add_p.get("T", 400))
    lambda_ = float(add_p.get("lambda", 0.5))
    eta = float(add_p.get("eta_inner", 0.1))
    R = 1.0

    A, b = make_problem(d=2, m=6, seed=seed)
    A, b = A.to(device), b.to(device)
    x = project_ball(torch.randn(2, device=device), R)
    x_prime = x.clone()
    traj: List[float] = []

    for t in range(T):
        g = grad_f_smooth(x_prime, A, b)
        x_prime = x_prime - eta * (g + (1.0 / lambda_) * (x_prime - x))
        x_prime = project_ball(x_prime, R)
        feat = torch.tensor([
            torch.norm(g).detach(),
            float(t) / float(T),
            torch.norm(x).detach() / R,
        ], device=device)
        alpha = net(feat) if net is not None else torch.tensor(0.0, device=device)
        x = x - eta * ((1.0 / lambda_) * (x - x_prime) + alpha * x)
        x = project_ball(x, R)
        traj.append(float(f_smooth(x_prime, A, b).detach().cpu()))

    # Estimate f* via sampling (non-diff. so detach)
    samp = torch.randn(2048, 2, device=device)
    samp = samp / torch.norm(samp, dim=1, keepdim=True)
    samp = samp * (torch.rand(2048, device=device).pow(0.5)).unsqueeze(1) * R
    f_star = torch.min(torch.stack([f_smooth(pt, A, b) for pt in samp])).detach()
    final_subopt = f_smooth(x_prime, A, b) - f_star
    return final_subopt, traj
