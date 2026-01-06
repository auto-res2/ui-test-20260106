"""Independent evaluation & visualisation (unchanged except robust imports)."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from scipy import stats

PRIMARY_METRIC = "suboptimality (final f̃(x′) − f̃*) after T iterations; lower is better."


def _save_json(obj: Dict, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)


def _draw_learning_curve(history: pd.DataFrame, run_id: str, out_dir: pathlib.Path) -> pathlib.Path:
    plt.figure(figsize=(6, 4))
    if "final_suboptimality" in history.columns:
        plt.plot(history.index, history["final_suboptimality"], label="subopt")
    plt.xlabel("Step")
    plt.ylabel("Suboptimality ↓")
    plt.title(f"Learning curve – {run_id}")
    plt.legend()
    plt.tight_layout()
    fname = f"{run_id}_learning_curve.pdf"
    fpath = out_dir / fname
    plt.savefig(fpath)
    plt.close()
    return fpath


def _export_single_run(run: wandb.apis.public.Run, out_root: pathlib.Path) -> Dict[str, float]:
    history = run.history(keys=["final_suboptimality"], pandas=True)
    summary = {k: v for k, v in run.summary.items() if isinstance(v, (int, float))}
    _save_json({"history": history.to_dict(), "summary": summary, "config": dict(run.config)}, out_root / "metrics.json")
    fig_path = _draw_learning_curve(history, run.id, out_root)
    print("Generated:", out_root / "metrics.json")
    print("Generated:", fig_path)
    return summary


def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str)
    parser.add_argument("run_ids", type=str, help="JSON list e.g. '["run-1", "run-2"]'")
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    run_ids: List[str] = json.loads(args.run_ids)

    api = wandb.Api()
    entity = os.environ.get("WANDB_ENTITY", "toma_tanaka")
    project = os.environ.get("WANDB_PROJECT", "2026-01-06")

    aggregated: Dict[str, Dict[str, float]] = {}
    per_run_primary: Dict[str, float] = {}

    for rid in run_ids:
        run = api.run(f"{entity}/{project}/{rid}")
        out_dir = results_dir / rid
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = _export_single_run(run, out_dir)
        val = summary.get("best_suboptimality")
        if val is not None:
            per_run_primary[rid] = float(val)
        for k, v in summary.items():
            aggregated.setdefault(k, {})[rid] = float(v)

    comp_dir = results_dir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    proposed = {k: v for k, v in per_run_primary.items() if "proposed" in k}
    baseline = {k: v for k, v in per_run_primary.items() if any(t in k for t in ("baseline", "comparative"))}
    best_prop_id = min(proposed, key=proposed.get) if proposed else None
    best_base_id = min(baseline, key=baseline.get) if baseline else None
    best_prop_val = proposed.get(best_prop_id, float("nan")) if best_prop_id else float("nan")
    best_base_val = baseline.get(best_base_id, float("nan")) if best_base_id else float("nan")

    minimise = "lower is better" in PRIMARY_METRIC.lower()
    gap = (best_base_val - best_prop_val) / best_base_val * 100 if minimise else (best_prop_val - best_base_val) / best_base_val * 100

    manifest = {
        "primary_metric": PRIMARY_METRIC,
        "metrics": aggregated,
        "best_proposed": {"run_id": best_prop_id, "value": best_prop_val},
        "best_baseline": {"run_id": best_base_id, "value": best_base_val},
        "gap": gap,
    }
    _save_json(manifest, comp_dir / "aggregated_metrics.json")
    print("Generated:", comp_dir / "aggregated_metrics.json")

    if per_run_primary:
        plt.figure(figsize=(7, 4))
        keys, vals = zip(*per_run_primary.items())
        sns.barplot(x=list(keys), y=list(vals))
        plt.ylabel("Best suboptimality ↓")
        plt.xticks(rotation=45, ha="right")
        for idx, v in enumerate(vals):
            plt.text(idx, v, f"{v:.4f}", ha="center", va="bottom")
        plt.tight_layout()
        bar_path = comp_dir / "comparison_best_subopt_bar_chart.pdf"
        plt.savefig(bar_path)
        plt.close()
        print("Generated:", bar_path)

    if len(proposed) >= 3 and len(baseline) >= 3:
        tstat, pval = stats.ttest_ind(list(proposed.values()), list(baseline.values()), equal_var=False)
        _save_json({"t_statistic": float(tstat), "p_value": float(pval)}, comp_dir / "ttest.json")
        print("Generated:", comp_dir / "ttest.json")


if __name__ == "__main__":
    main()
