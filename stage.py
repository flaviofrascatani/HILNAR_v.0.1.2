### Stage.py to apply the defs ###

#ATTENTION! The code doesn't run by clicking the play button on VS Code
# Copy and paste these commands one by one on the powershell if u need
#to use stage.py

#   python -X utf8 stage.py boot 20
#   python -X utf8 stage.py perm 20
#   python -X utf8 stage.py grid
#   python -X utf8 run.py --collect

#   python -X utf8 stage.py boot 1000
#   python -X utf8 stage.py perm 1000
#   python -X utf8 run.py --collect




"""Run one inference stage and checkpoint it to out/<stage>.json.

    python stage.py boot   [B]     document bootstrap
    python stage.py perm   [B]     permutation tests (NSI and SAI)
    python stage.py grid           specification grid

Each stage is independent and idempotent, so a long inference run can be
resumed instead of restarted.  `run.py --collect` assembles the checkpoints.
"""

import json
import os
import sys
import time

import torch

import hilnar_initial_def_for_run as H
from corpus import DOCUMENTS

torch.set_num_threads(min(4, os.cpu_count() or 1))
OUT = "out"
YEARS = (2019, 2025)


def main():
    stage = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    os.makedirs(OUT, exist_ok=True)
    cfg = H.Config()
    t0 = time.time()

    if stage == "boot":
        res = {"n_boot": n,
               "bootstrap": H.bootstrap(DOCUMENTS, cfg, YEARS, n)}
    elif stage == "perm":
        res = {"n_perm": n,
               "permutation_nsi": H.permutation_nsi(DOCUMENTS, cfg, YEARS, n),
               "permutation_sai": {str(y): H.permutation_sai(DOCUMENTS, y, cfg, n)
                                   for y in YEARS}}
    elif stage == "grid":
        rows = H.specification_grid(DOCUMENTS, YEARS)
        res = {"spec_grid": rows}
    else:
        raise SystemExit(f"unknown stage {stage!r}")

    res["elapsed_s"] = round(time.time() - t0, 1)
    with open(f"{OUT}/{stage}.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"{stage}: wrote {OUT}/{stage}.json in {res['elapsed_s']}s")


if __name__ == "__main__":
    main()
