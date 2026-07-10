"""Sweep orchestration for config.yaml's `sweep` section.

Loops through every combination in sweep.grid (+ sweep.extra_runs),
overriding data.organism, data.replica_count, and mlm_model.profile per
combo. Each combo runs `python -m main` as its own subprocess against a
generated override config (via AGENTIC_CONFIG_PATH), with that combo's
sweep.enabled forced to false so it runs exactly once using cfg.run.method.
The checked-in config.yaml is never modified.

Each combo subprocess regenerates its own fragmented dataset automatically
if needed (see preprocessing.preprocessing.ensure_fresh_dataset, called from
main.py) by comparing organism/replica_count/missed_cleavage_ratio against
the sidecar .meta.json saved next to the fragmented file, so preprocessing
naturally re-runs once per distinct combination in the grid without this
module having to track that itself.
"""

from __future__ import annotations

import copy
import itertools
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

from config import CONFIG_PATH
from evaluation.sweep_report import write_sweep_report

ROOT = Path(__file__).resolve().parent.parent
TMP_CONFIG_DIR = ROOT / "results" / "_sweep_configs"
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def _format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _combo_label(combo: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in combo.items())


def _build_combos(sweep_cfg: dict) -> list[dict]:
    grid = sweep_cfg.get("grid", {}) or {}
    axis_names = list(grid.keys())
    axis_values = [grid[name] for name in axis_names]
    combos = [dict(zip(axis_names, values)) for values in itertools.product(*axis_values)]
    combos.extend(sweep_cfg.get("extra_runs") or [])
    if not combos:
        raise ValueError("sweep.enabled is true but sweep.grid and sweep.extra_runs are both empty.")
    return combos


def _apply_overrides(base_cfg: dict, combo: dict, sweep_cfg: dict) -> dict:
    run_cfg = copy.deepcopy(base_cfg)
    run_cfg["sweep"]["enabled"] = False  # each combo subprocess runs a single evaluation
    if "organism" in combo:
        run_cfg["data"]["organism"] = combo["organism"]
    if "replica_count" in combo:
        run_cfg["data"]["replica_count"] = combo["replica_count"]
    if "mlm_profile" in combo:
        run_cfg["mlm_model"]["profile"] = combo["mlm_profile"]
    if "method" in combo:
        run_cfg.setdefault("run", {})["method"] = combo["method"]
    test_samples = combo.get("test_samples", sweep_cfg.get("test_samples"))
    if test_samples is not None:
        run_cfg["data"]["test_samples"] = test_samples
    return run_cfg


def _run_subprocess(module: str, env: dict) -> str | None:
    print(f"    -> python -m {module}")
    # main.py's console output uses box-drawing characters; without forcing
    # UTF-8 the child inherits the OS codepage (cp1252 on Windows) and dies
    # decoding its own stdout the moment it prints one.
    env = dict(env)
    env["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.Popen(
        [sys.executable, "-m", module],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    result_dir = None
    for line in process.stdout:
        print(f"    | {line.rstrip()}")
        if "Saved run artifacts to" in line:
            raw = line.split("Saved run artifacts to", 1)[1]
            result_dir = ANSI_ESCAPE.sub("", raw).strip()
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"{module} exited with code {process.returncode}")
    return result_dir


def run_sweep(sweep_cfg: dict) -> None:
    with open(CONFIG_PATH) as f:
        base_cfg = yaml.safe_load(f)

    combos = _build_combos(sweep_cfg)
    TMP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Sweep: {len(combos)} combination(s) from {CONFIG_PATH}")
    for i, combo in enumerate(combos, 1):
        print(f"  {i}. {_combo_label(combo)}")

    manifest = []
    started = time.time()
    durations: list[float] = []

    for i, combo in enumerate(combos, 1):
        combo_start = time.time()
        completed = i - 1
        percent = completed / len(combos) * 100
        avg = sum(durations) / len(durations) if durations else None
        eta = avg * (len(combos) - completed) if avg is not None else None
        eta_text = f", ETA {_format_duration(eta)} remaining" if eta is not None else ""

        print(f"\n{'=' * 70}")
        print(f"Sweep combo {i}/{len(combos)} ({percent:.0f}% of sweep complete){eta_text}")
        print(f"Sweep elapsed: {_format_duration(time.time() - started)}")
        print(f"Combo settings: {_combo_label(combo)}")
        print(f"{'=' * 70}")

        run_cfg = _apply_overrides(base_cfg, combo, sweep_cfg)
        config_path = TMP_CONFIG_DIR / f"combo_{i:03d}.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(run_cfg, f)

        env = os.environ.copy()
        env["AGENTIC_CONFIG_PATH"] = str(config_path)
        # Hard guard read by main.main(): a combo child must never start its own
        # sweep, even if its config were misread. Prevents the nested-sweep
        # process/memory explosion.
        env["AGENTIC_IN_SWEEP_CHILD"] = "1"

        status, error, result_dir = "ok", None, None
        try:
            result_dir = _run_subprocess("main", env)
        except RuntimeError as exc:
            status, error = "failed", str(exc)
            print(f"  FAILED: {exc}")

        duration = time.time() - combo_start
        durations.append(duration)
        manifest.append(
            {
                "combo": combo,
                "status": status,
                "error": error,
                "result_dir": result_dir,
                "duration_seconds": duration,
            }
        )

    total = time.time() - started
    succeeded = sum(1 for m in manifest if m["status"] == "ok")
    print(f"\n{'=' * 70}")
    print(f"Sweep finished: {succeeded}/{len(manifest)} succeeded in {_format_duration(total)}")
    for m in manifest:
        marker = "OK  " if m["status"] == "ok" else "FAIL"
        print(f"  [{marker}] {_combo_label(m['combo'])} -> {m['result_dir'] or m['error']}")

    report_dir = write_sweep_report(manifest, sweep_cfg, total)
    print(f"\nCombined sweep report saved to {report_dir}")
    for name in sorted(p.name for p in report_dir.iterdir() if p.is_file()):
        print(f"    - {name}")
