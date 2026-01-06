"""Main orchestrator – delegates to *src.train* in a fresh Hydra context."""
import os
import pathlib
import subprocess
import sys
from typing import List

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:  # noqa: D401
    script = pathlib.Path(__file__).resolve().parent / "train.py"
    cmd: List[str] = [
        sys.executable,
        "-u",
        str(script),
        f"run={cfg.run}",
        f"results_dir={to_absolute_path(cfg.results_dir)}",
        f"mode={cfg.mode}",
    ]
    print("Executing:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=os.environ.copy())


if __name__ == "__main__":
    main()
