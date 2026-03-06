from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Optional

import numpy as np


@dataclass
class EventSpec:
    when: str  # "on_start" | "on_reset" | "on_interval"
    fn: Callable[[np.random.Generator, Dict[str, Any]], Dict[str, Any]]
    interval: Optional[int] = None
    targets: Optional[List[str]] = None


class EventManager:
    def __init__(self, events: List[EventSpec] | None = None, seed: int | None = None):
        self.events = events or []
        self.rng = np.random.default_rng(seed)

    def apply(self, when: str, env_params: List[Dict[str, Any]], step: int | None = None) -> List[Dict[str, Any]]:
        """Apply matching events and return updated param list (always copies dicts).

        Prefer ``apply_inplace`` in hot paths — it avoids dict copying when no
        events fire and returns a ``changed`` flag to gate downstream rebuilds.
        """
        if not self.events:
            return env_params
        out = []
        for i, p in enumerate(env_params):
            updated = dict(p)
            for ev in self.events:
                if ev.when != when:
                    continue
                if ev.when == "on_interval" and ev.interval:
                    if step is None or step % ev.interval != 0:
                        continue
                updated = ev.fn(self.rng, updated)
            out.append(updated)
        return out

    def apply_inplace(
        self, when: str, env_params: List[Dict[str, Any]], step: int | None = None
    ) -> tuple:
        """Like apply() but returns ``(updated_params, changed: bool)``.

        Returns immediately with ``changed=False`` when no events match,
        avoiding unnecessary dict copying and downstream rebuilds.
        """
        if not self.events:
            return env_params, False

        # Filter to events that actually fire right now.
        relevant = []
        for ev in self.events:
            if ev.when != when:
                continue
            if ev.when == "on_interval" and ev.interval:
                if step is None or step % ev.interval != 0:
                    continue
            relevant.append(ev)

        if not relevant:
            return env_params, False

        out = []
        for p in env_params:
            updated = dict(p)
            for ev in relevant:
                updated = ev.fn(self.rng, updated)
            out.append(updated)
        return out, True


def uniform(name: str, low: float, high: float):
    """Return an event function that samples ``name`` uniformly in [low, high]."""
    def _fn(rng: np.random.Generator, params: Dict[str, Any]) -> Dict[str, Any]:
        params[name] = float(rng.uniform(low, high))
        return params
    return _fn


def loguniform(name: str, low: float, high: float):
    """Return an event function that samples ``name`` log-uniformly in [low, high]."""
    def _fn(rng: np.random.Generator, params: Dict[str, Any]) -> Dict[str, Any]:
        params[name] = float(np.exp(rng.uniform(np.log(low), np.log(high))))
        return params
    return _fn


def normal(name: str, mean: float, std: float):
    """Return an event function that samples ``name`` from N(mean, std)."""
    def _fn(rng: np.random.Generator, params: Dict[str, Any]) -> Dict[str, Any]:
        params[name] = float(rng.normal(mean, std))
        return params
    return _fn
