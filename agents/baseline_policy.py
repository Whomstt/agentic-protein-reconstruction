"""Non-LLM lever selector for the matched-budget control arm.

The control arm exists to answer the question a plain deterministic-vs-agentic
table cannot: is the agentic gain due to the LLM's *reasoning*, or merely due to
trying several diverse candidates and keeping the best-validity one? This policy
produces the same five levers the agent would, but from a fixed random/grid rule
instead of model reasoning, so an arm driven by it shares the agent's budget,
tool pipeline and selection rule and differs only in where the lever values come
from. Agentic-best minus control-best is then the isolated value of the LLM.
"""

from __future__ import annotations

import itertools
import random

from config import cfg

LEVER_KEYS = (
    "junction_window",
    "search_mode",
    "beam_width",
    "edge_mode",
    "confirmed_bonus",
)

DEFAULT_LEVER_SPACE = {
    "junction_window": [1, 2, 3, 4, 5],
    "search_mode": ["greedy", "beam"],
    "beam_width": [10, 25, 50, 100, 200],
    "edge_mode": ["hard", "soft"],
    "confirmed_bonus": [0.0, 0.5, 1.0],
}


def _lever_space() -> dict:
    control = cfg["run"].get("control_baseline") or {}
    configured = control.get("lever_space") or {}
    return {key: list(configured.get(key, DEFAULT_LEVER_SPACE[key])) for key in LEVER_KEYS}


def _default_levers() -> dict:
    return dict(cfg["search"]["default_levers"])


def _combo_key(levers: dict) -> tuple:
    return tuple(levers.get(key) for key in LEVER_KEYS)


class LeverPolicy:
    """Chooses the five strategy levers without an LLM, for the control arm.

    kind="random": seeded uniform draw over lever_space each iteration, skipping a
    combination already tried this run where possible. kind="grid": a deterministic
    spread across the full lever grid (shuffled once by a fixed seed so a small
    budget still samples diverse combos rather than only varying the last lever).

    When run.iteration1_deterministic is set, iteration 1 returns the fixed
    search.default_levers exactly as the agentic arm does, so both arms share the
    same starting candidate and only iterations 2+ differ by lever source.
    """

    def __init__(self, kind: str | None = None, seed: int | None = None):
        control = cfg["run"].get("control_baseline") or {}
        self.kind = kind or control.get("policy", "random")
        self.space = _lever_space()
        self.iteration1_deterministic = cfg["run"].get("iteration1_deterministic", True)
        base_seed = seed if seed is not None else (cfg["misc"].get("seed") or 0)
        self._rng = random.Random(base_seed)
        self._grid = self._build_grid()

    def _build_grid(self) -> list[dict]:
        product = [
            dict(zip(LEVER_KEYS, values))
            for values in itertools.product(*(self.space[key] for key in LEVER_KEYS))
        ]
        # Fixed-seed shuffle (independent of the per-sample seed) so grid mode is
        # reproducible and covers the space instead of the first N neighbours.
        random.Random((cfg["misc"].get("seed") or 0) + 991).shuffle(product)
        return product

    def _random_levers(self) -> dict:
        return {key: self._rng.choice(self.space[key]) for key in LEVER_KEYS}

    def choose(self, iteration: int, history: list[dict] | None) -> dict:
        if iteration == 1 and self.iteration1_deterministic:
            return _default_levers()

        tried = {
            _combo_key(record["lever_values"])
            for record in (history or [])
            if record.get("lever_values")
        }

        if self.kind == "grid":
            for combo in self._grid:
                if _combo_key(combo) not in tried:
                    return dict(combo)
            return dict(self._grid[(iteration - 1) % len(self._grid)])

        levers = self._random_levers()
        for _ in range(50):
            if _combo_key(levers) not in tried:
                break
            levers = self._random_levers()
        return levers
