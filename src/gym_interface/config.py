from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from math import exp, log
from pathlib import Path
from typing import TYPE_CHECKING, Any, Tuple

import yaml

from src.randomization import EventSpec

if TYPE_CHECKING:
    from src.gym_interface.room_env import RoomHeatEnv
    from src.gym_interface.vec_env import RoomHeatVecEnv


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML file as a dictionary."""
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, Mapping):
        raise TypeError(f"YAML root must be a mapping/dict in {cfg_path}")
    return dict(data)


def build_randomization_events(
    randomization_cfg: Mapping[str, Any] | None,
) -> tuple[list[EventSpec], int | None]:
    """Build EventSpec list from YAML-friendly domain-randomization config."""
    if not randomization_cfg:
        return [], None

    events_cfg = randomization_cfg.get("events", [])
    if not isinstance(events_cfg, list):
        raise TypeError("domain_randomization.events must be a list")

    seed = randomization_cfg.get("seed", None)
    seed = int(seed) if seed is not None else None

    events: list[EventSpec] = []
    for i, event_cfg in enumerate(events_cfg):
        if not isinstance(event_cfg, Mapping):
            raise TypeError(f"domain_randomization.events[{i}] must be a mapping")

        when = str(event_cfg.get("when", "")).strip()
        if when not in {"on_start", "on_reset", "on_interval"}:
            raise ValueError(
                f"domain_randomization.events[{i}].when must be one of "
                f"on_start/on_reset/on_interval, got: {when!r}"
            )

        interval = event_cfg.get("interval", None)
        if when == "on_interval":
            if interval is None:
                raise ValueError(f"domain_randomization.events[{i}] missing interval")
            interval = int(interval)
            if interval <= 0:
                raise ValueError(
                    f"domain_randomization.events[{i}].interval must be > 0, got: {interval}"
                )

        target = (
            event_cfg.get("target", None)
            or event_cfg.get("name", None)
            or event_cfg.get("param", None)
        )
        if not target:
            raise ValueError(
                f"domain_randomization.events[{i}] must define target/name/param"
            )

        fn = _build_event_fn(i=i, target=str(target), event_cfg=event_cfg)
        events.append(
            EventSpec(
                when=when,
                fn=fn,
                interval=interval,
                targets=[str(target)],
            )
        )
    return events, seed


def make_env_from_config(
    config_or_path: str | Path | Mapping[str, Any],
    env_key: str = "env",
    randomization_key: str = "domain_randomization",
):
    """Build RoomHeatEnv or RoomHeatVecEnv from YAML config."""
    cfg = _load_config(config_or_path)
    env_cfg, randomization_cfg = _split_env_and_randomization(
        cfg,
        env_key=env_key,
        randomization_key=randomization_key,
    )
    env_type = str(env_cfg.get("type", env_cfg.get("kind", ""))).strip().lower()
    if not env_type:
        env_type = "vec" if ("num_envs" in env_cfg or "num_env" in env_cfg) else "room"
    if env_type in {"room", "single", "scalar"}:
        return make_room_heat_env_from_config(
            cfg,
            env_key=env_key,
            randomization_key=randomization_key,
        )
    if env_type in {"vec", "vector", "batched"}:
        return make_room_heat_vec_env_from_config(
            cfg,
            env_key=env_key,
            randomization_key=randomization_key,
        )
    raise ValueError(f"Unknown env type in config: {env_type!r}")


def make_room_heat_env_from_config(
    config_or_path: str | Path | Mapping[str, Any],
    env_key: str = "env",
    randomization_key: str = "domain_randomization",
    overrides: Mapping[str, Any] | None = None,
) -> RoomHeatEnv:
    """Build RoomHeatEnv from YAML config."""
    from src.gym_interface.room_env import RoomHeatEnv

    cfg = _load_config(config_or_path)
    env_cfg, randomization_cfg = _split_env_and_randomization(
        cfg,
        env_key=env_key,
        randomization_key=randomization_key,
    )
    env_kwargs = _normalize_env_kwargs(env_cfg)
    env_kwargs.pop("type", None)
    env_kwargs.pop("kind", None)

    events, seed = build_randomization_events(randomization_cfg)
    if events:
        env_kwargs["randomization_events"] = events
    if seed is not None and "randomization_seed" not in env_kwargs:
        env_kwargs["randomization_seed"] = seed

    if overrides:
        env_kwargs.update(dict(overrides))
    return RoomHeatEnv(**env_kwargs)


def make_room_heat_vec_env_from_config(
    config_or_path: str | Path | Mapping[str, Any],
    env_key: str = "env",
    randomization_key: str = "domain_randomization",
    overrides: Mapping[str, Any] | None = None,
) -> RoomHeatVecEnv:
    """Build RoomHeatVecEnv from YAML config."""
    from src.gym_interface.vec_env import RoomHeatVecEnv

    cfg = _load_config(config_or_path)
    env_cfg, randomization_cfg = _split_env_and_randomization(
        cfg,
        env_key=env_key,
        randomization_key=randomization_key,
    )
    env_kwargs = _normalize_env_kwargs(env_cfg)
    env_kwargs.pop("type", None)
    env_kwargs.pop("kind", None)

    if overrides:
        env_kwargs.update(dict(overrides))

    if "num_envs" not in env_kwargs:
        raise ValueError("Vector env config must define env.num_envs (or env.num_env)")

    events, seed = build_randomization_events(randomization_cfg)
    if events:
        env_kwargs["randomization_events"] = events
    if seed is not None and "randomization_seed" not in env_kwargs:
        env_kwargs["randomization_seed"] = seed

    return RoomHeatVecEnv(**env_kwargs)


def get_config_section(
    config_or_path: str | Path | Mapping[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    """Convenience accessor for top-level config sections."""
    cfg = _load_config(config_or_path)
    return cfg.get(key, default)


def _load_config(config_or_path: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config_or_path, (str, Path)):
        return load_yaml_config(config_or_path)
    if isinstance(config_or_path, Mapping):
        return dict(config_or_path)
    raise TypeError(
        "config_or_path must be a YAML path or a mapping/dict, "
        f"got: {type(config_or_path)}"
    )


def _split_env_and_randomization(
    cfg: Mapping[str, Any],
    env_key: str,
    randomization_key: str,
) -> Tuple[dict[str, Any], Mapping[str, Any] | None]:
    if env_key in cfg:
        env_cfg = cfg.get(env_key)
        if env_cfg is None:
            env_cfg = {}
        if not isinstance(env_cfg, Mapping):
            raise TypeError(f"{env_key} must be a mapping/dict")
        env_cfg = dict(env_cfg)
    else:
        env_cfg = dict(cfg)

    randomization_cfg = cfg.get(randomization_key, None)
    if randomization_cfg is None:
        randomization_cfg = env_cfg.get(randomization_key, None)
    if randomization_cfg is not None and not isinstance(randomization_cfg, Mapping):
        raise TypeError(f"{randomization_key} must be a mapping/dict")

    env_cfg.pop(randomization_key, None)
    return env_cfg, randomization_cfg


def _normalize_env_kwargs(env_cfg: Mapping[str, Any]) -> dict[str, Any]:
    cfg = dict(env_cfg)

    if "mdot_hp" in cfg and "mdot_HP" not in cfg:
        cfg["mdot_HP"] = cfg.pop("mdot_hp")
    if "num_env" in cfg and "num_envs" not in cfg:
        cfg["num_envs"] = int(cfg.pop("num_env"))

    if "weather_forecast_steps" in cfg:
        cfg["weather_forecast_steps"] = _normalize_forecast_steps(cfg["weather_forecast_steps"])
    elif "forecast" in cfg:
        cfg["weather_forecast_steps"] = _normalize_forecast_steps(cfg.pop("forecast"))

    if "goal_temp_range" in cfg and cfg["goal_temp_range"] is not None:
        goal_range = cfg["goal_temp_range"]
        if isinstance(goal_range, (list, tuple)):
            if len(goal_range) != 2:
                raise ValueError("goal_temp_range must have exactly 2 values")
            cfg["goal_temp_range"] = (float(goal_range[0]), float(goal_range[1]))
        else:
            raise TypeError("goal_temp_range must be a 2-element list/tuple")

    return cfg


def _normalize_forecast_steps(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        if value <= 0:
            return []
        return list(range(1, value + 1))
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    raise TypeError(
        "weather_forecast_steps (or forecast) must be int/list/tuple, "
        f"got: {type(value)}"
    )


def _build_event_fn(i: int, target: str, event_cfg: Mapping[str, Any]):
    distribution = str(
        event_cfg.get("distribution", event_cfg.get("type", "uniform"))
    ).lower()
    clip = event_cfg.get("clip", None)
    dtype = event_cfg.get("dtype", None)

    if distribution == "uniform":
        low = float(event_cfg["low"])
        high = float(event_cfg["high"])

        def _sample(rng):
            return float(rng.uniform(low, high))
    elif distribution == "loguniform":
        low = float(event_cfg["low"])
        high = float(event_cfg["high"])
        if low <= 0 or high <= 0:
            raise ValueError(
                f"domain_randomization.events[{i}] loguniform requires low/high > 0"
            )

        def _sample(rng):
            return float(exp(rng.uniform(log(low), log(high))))
    elif distribution == "normal":
        mean = float(event_cfg["mean"])
        std = float(event_cfg["std"])

        def _sample(rng):
            return float(rng.normal(mean, std))
    elif distribution == "constant":
        value = event_cfg["value"]

        def _sample(rng):
            return value
    elif distribution == "choice":
        values = event_cfg["values"]
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"domain_randomization.events[{i}] choice requires non-empty values list"
            )

        def _sample(rng):
            idx = int(rng.integers(0, len(values)))
            return values[idx]
    else:
        raise ValueError(
            f"domain_randomization.events[{i}] unsupported distribution: {distribution!r}"
        )

    def _fn(rng, params: MutableMapping[str, Any]):
        value = _sample(rng)
        if clip is not None:
            if (
                not isinstance(clip, (list, tuple))
                or len(clip) != 2
            ):
                raise ValueError(
                    f"domain_randomization.events[{i}].clip must be [min, max]"
                )
            value = min(max(float(value), float(clip[0])), float(clip[1]))
        if dtype is not None:
            cast = str(dtype).lower()
            if cast == "int":
                value = int(round(float(value)))
            elif cast == "float":
                value = float(value)
            elif cast == "bool":
                value = bool(value)
            else:
                raise ValueError(
                    f"domain_randomization.events[{i}] unsupported dtype: {dtype!r}"
                )
        _set_nested(params, target, value)
        return params

    return _fn


def _set_nested(params: MutableMapping[str, Any], target: str, value: Any) -> None:
    """Set a dot-nested key in a mutable mapping, creating intermediate dicts as needed.

    Example: ``_set_nested(d, "a.b.c", 42)`` sets ``d["a"]["b"]["c"] = 42``.
    """
    keys = target.split(".")
    cur: MutableMapping[str, Any] = params
    for key in keys[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, MutableMapping):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[keys[-1]] = value
