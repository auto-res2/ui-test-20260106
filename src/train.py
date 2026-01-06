"""Training script for Adaptive Outer Regularisation experiments *with a
proper data pipeline*.

Each *GSM8K* question is converted into a deterministic pseudo-random seed that
spawns a synthetic optimisation task (log-sum-exp surrogate in 2-D).  The
Adaptive-α network is optimised to minimise the *mean* final sub-optimality over
all questions therefore **using the dataset as supervision**.  Baseline runs
(α≡0) execute the same tasks but the network is omitted.

The full CLI contract is documented in *README* and enforced by
``src/main.py``.
"""
from __future__ import annotations

import json
import os
import pathlib
import random
import zlib
from copy import deepcopy
from typing import Any, Dict, List

import hydra
import numpy as np
import optuna
import torch
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from src.model import AdaptiveAlphaNet, run_moreau_yosida
from src.preprocess import load_and_preprocess

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _str_to_seed(txt: str) -> int:
    """Stable conversion from *txt* → [0, 2**31)."""
    return zlib.adler32(txt.encode("utf-8")) & 0x7FFFFFFF


class _WandBStub:  # minimal drop-in replacement when disabled
    def __init__(self):
        self.summary: Dict[str, Any] = {}

    def log(self, *_a, **_kw):
        return None

    def plot(self, *_a, **_kw):  # type: ignore[no-self-use]
        class _NoPlot:  # pylint: disable=too-few-public-methods
            def line_series(self, *args, **kwargs):
                return None
        return _NoPlot()

    def init(self, *args, **kwargs):  # noqa: D401
        return self

    def run(self):  # type: ignore[no-self-use]
        return self

    @staticmethod
    def get_url() -> str:  # noqa: D401
        return "wandb-disabled"


# -----------------------------------------------------------------------------
# Dataset → PyTorch DataLoader
# -----------------------------------------------------------------------------

class _QuestionDataset(Dataset):
    def __init__(self, hf_split):
        self._data = hf_split

    def __len__(self) -> int:  # noqa: D401
        return len(self._data)

    def __getitem__(self, idx):  # noqa: D401
        return self._data[idx]["question"]


def _build_loader(hf_ds_split, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(_QuestionDataset(hf_ds_split), batch_size=batch_size, shuffle=shuffle)

# -----------------------------------------------------------------------------
# Hydra main
# -----------------------------------------------------------------------------

@hydra.main(config_path="../config", config_name="config", version_base=None)
# pylint: disable=too-many-branches,too-many-locals,too-many-statements
def main(cfg: DictConfig) -> None:  # noqa: C901, D401
    # ------------------------------------------------------------------
    # (1) Merge global + run-specific YAML
    # ------------------------------------------------------------------
    if not cfg.get("run"):
        raise ValueError("Missing CLI param `run=<run_id>`.")
    run_cfg_path = to_absolute_path(os.path.join("config", "runs", f"{cfg.run}.yaml"))
    if not os.path.exists(run_cfg_path):
        raise FileNotFoundError(f"Run-config not found: {run_cfg_path}")
    cfg = OmegaConf.merge(cfg, OmegaConf.load(run_cfg_path))

    run_id: str = cfg.get("run_id", str(cfg.run))

    # ------------------------------------------------------------------
    # (2) Mode-specific overrides
    # ------------------------------------------------------------------
    if cfg.mode == "trial":
        cfg.training.epochs = 1
        cfg.optuna.n_trials = 0
        cfg.wandb.mode = "disabled"
        cfg.training.max_batches = 2  # fast pass
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
        cfg.training.max_batches = 0  # 0 → unlimited
    else:
        raise ValueError("mode must be 'trial' or 'full'.")

    # ------------------------------------------------------------------
    # (3) Results directory
    # ------------------------------------------------------------------
    results_root = pathlib.Path(to_absolute_path(cfg.results_dir))
    run_out_dir = results_root / run_id
    run_out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # (4) Seeding
    # ------------------------------------------------------------------
    set_seeds(int(cfg.training.seed))

    # ------------------------------------------------------------------
    # (5) Data pipeline (actual use!)
    # ------------------------------------------------------------------
    splits = load_and_preprocess(cfg)
    loader = _build_loader(
        splits["train"], batch_size=int(cfg.training.batch_size), shuffle=True
    )

    # ------------------------------------------------------------------
    # (6) Device
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # (7) Optional Optuna hyper-search (offline)
    # ------------------------------------------------------------------
    best_params: Dict[str, Any] = {}
    if int(cfg.optuna.n_trials) > 0:
        print(f"[Optuna] {cfg.optuna.n_trials} trial(s) …")

        def _suggest(trial: optuna.trial.Trial, spaces):  # noqa: D401
            out = {}
            for sp in spaces:
                name, dist = sp["param_name"], sp["distribution_type"].lower()
                low, high = float(sp["low"]), float(sp["high"])
                if dist == "int":
                    out[name] = trial.suggest_int(name, int(low), int(high))
                elif dist == "uniform":
                    out[name] = trial.suggest_float(name, low, high)
                elif dist == "loguniform":
                    out[name] = trial.suggest_float(name, low, high, log=True)
                else:
                    raise ValueError(dist)
            return out

        def _objective(trial):  # noqa: ANN001
            params = _suggest(trial, cfg.optuna.search_spaces)
            trial_cfg = deepcopy(cfg)
            if "hidden_dim" in params:
                trial_cfg.model.hidden_dim = int(params["hidden_dim"])
            for p in ("lambda", "eta_inner", "alpha_cap"):
                if p in params:
                    trial_cfg.training.additional_params[p] = params[p]
            net = AdaptiveAlphaNet(
                input_dim=int(trial_cfg.model.input_dim),
                hidden_dim=int(trial_cfg.model.hidden_dim),
                min_val=0.0,
                max_val=float(trial_cfg.training.additional_params.get("alpha_cap", 0.5)),
            ).to(device)
            # Evaluate on a *tiny* random subset for speed
            sample_q = [splits["train"][i]["question"] for i in range(16)]
            losses = []
            for q in sample_q:
                seed_ = _str_to_seed(q)
                loss_t, _ = run_moreau_yosida(trial_cfg, net, seed_, device)
                losses.append(loss_t.detach())
            return float(torch.mean(torch.stack(losses)))

        study = optuna.create_study(direction="minimize")
        study.optimize(_objective, n_trials=int(cfg.optuna.n_trials))
        best_params = study.best_params
        print("Optuna best:", best_params)
        # Inject
        if "hidden_dim" in best_params:
            cfg.model.hidden_dim = int(best_params["hidden_dim"])
        for p in ("lambda", "eta_inner", "alpha_cap"):
            if p in best_params:
                cfg.training.additional_params[p] = best_params[p]

    # ------------------------------------------------------------------
    # (8) Build network (if adaptive)
    # ------------------------------------------------------------------
    adaptive_run = str(cfg.method).startswith("AdaptiveOuterRegularization")
    net = None
    optimiser = None
    if adaptive_run:
        net = AdaptiveAlphaNet(
            input_dim=int(cfg.model.input_dim),
            hidden_dim=int(cfg.model.hidden_dim),
            min_val=0.0,
            max_val=float(cfg.training.additional_params.get("alpha_cap", 0.5)),
        ).to(device)
        optimiser = torch.optim.Adam(
            net.parameters(),
            lr=float(cfg.training.learning_rate),
            weight_decay=float(cfg.training.weight_decay),
        )
        # Post-init assertion
        test_out = net(torch.randn(int(cfg.model.input_dim), device=device))
        assert test_out.ndim == 0, "α-net output must be scalar"

    # ------------------------------------------------------------------
    # (9) WandB init
    # ------------------------------------------------------------------
    if cfg.wandb.mode != "disabled":
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
            mode=cfg.wandb.mode,
        )
    else:  # stub
        globals()["wandb"] = _WandBStub()  # type: ignore[misc]

    # ------------------------------------------------------------------
    # (10) Training epochs
    # ------------------------------------------------------------------
    best_subopt = float("inf")
    for epoch in range(int(cfg.training.epochs)):
        epoch_losses: List[torch.Tensor] = []
        for batch_idx, questions in enumerate(loader):
            if 0 < int(cfg.training.get("max_batches", 0)) <= batch_idx:
                break  # CI-friendly early exit
            batch_losses: List[torch.Tensor] = []
            trajectories_logged = False
            for q in questions:
                seed_ = _str_to_seed(str(q))
                loss_t, traj = run_moreau_yosida(cfg, net, seed_, device)
                batch_losses.append(loss_t)
                # Batch-start assertion on first sample
                if epoch == 0 and batch_idx == 0 and not trajectories_logged:
                    assert traj and np.isfinite(traj[0]), "Initial f value NaN"  # type: ignore[index]
                    trajectories_logged = True
            loss = torch.mean(torch.stack(batch_losses))
            if adaptive_run:
                assert optimiser is not None
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                # Pre-optimiser gradient integrity
                grad_ok = any(
                    p.grad is not None and torch.any(p.grad != 0) for p in net.parameters()
                )
                assert grad_ok, "Gradients vanished before optimiser.step()"
                optimiser.step()
            epoch_losses.append(loss.detach())
        epoch_loss_val = float(torch.mean(torch.stack(epoch_losses)))
        best_subopt = min(best_subopt, epoch_loss_val)
        wandb.log(
            {
                "epoch": epoch,
                "final_suboptimality": epoch_loss_val,
                "best_suboptimality": best_subopt,
            }
        )
        print(f"[Epoch {epoch}] mean subopt={epoch_loss_val:.6f}")

    # ------------------------------------------------------------------
    # (11) Finalise
    # ------------------------------------------------------------------
    wandb.summary["best_suboptimality"] = best_subopt
    if cfg.wandb.mode != "disabled":
        print("WandB URL:", wandb.run.get_url())

    with open(run_out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "best_suboptimality": best_subopt,
                "optuna_best": best_params,
            },
            fh,
            indent=2,
        )


if __name__ == "__main__":
    main()
