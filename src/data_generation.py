"""Helpers for generating building simulation datasets.

This module keeps notebook code thin: disturbances are generated in memory,
the building is simulated, and the saved artifact is a building-data CSV with
state/sensor, control, cost, and disturbance columns.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd

import data.buildings as building_catalog
from src import disturbances
from src.models import model_hvac
from src.models.model_buildings import Building
from src.simulator import Model_simulator


DEFAULT_LOCATIONS = {
    "building_default": None,
    "freiburg": {
        "lat": 48.0252,
        "long": 7.7184,
        "altitude": 207,
        "timezone": "Europe/Berlin",
    },
    "mannheim": {
        "lat": 49.1423,
        "long": 9.2187,
        "altitude": 100,
        "timezone": "Europe/Berlin",
    },
}


def slugify(value: Any) -> str:
    """Return a filename-safe representation."""
    return (
        str(value)
        .replace("/", "-")
        .replace(" ", "_")
        .replace(".", "p")
        .replace(":", "")
    )


def resolve_location(
    location: str | dict | None,
    locations: dict[str, dict | None] | None = None,
    repo_filepath: str | Path = "",
    default_timezone: str = "Europe/Berlin",
) -> tuple[str, dict | None]:
    """Resolve a named, explicit, default, or random location.

    Parameters
    ----------
    location:
        - ``None`` or ``"building_default"`` keeps the building's stored position.
        - A dict with ``lat`` and ``long`` overrides the building position.
        - A key in ``locations`` selects a predefined location.
        - ``"random"`` or ``"random_DE"`` samples a random city via
          ``disturbances.get_random_location``.

    Notes
    -----
    The random-location helper only returns latitude/longitude. Altitude is set
    to 0 and timezone to ``default_timezone``.
    """
    location_map = {**DEFAULT_LOCATIONS, **(locations or {})}

    if location is None or location == "building_default":
        return "building_default", None

    if isinstance(location, dict):
        if "lat" not in location or "long" not in location:
            raise KeyError("Location dict must contain 'lat' and 'long'.")
        name = location.get("name", "custom")
        return str(name), {
            "lat": float(location["lat"]),
            "long": float(location["long"]),
            "altitude": float(location.get("altitude", 0)),
            "timezone": location.get("timezone", default_timezone),
        }

    if not isinstance(location, str):
        raise TypeError("location must be None, a dict, or a string")

    if location in location_map:
        selected = location_map[location]
        if selected is None:
            return location, None
        if "lat" not in selected or "long" not in selected:
            raise KeyError("Location dict must contain 'lat' and 'long'.")
        return location, {
            "lat": float(selected["lat"]),
            "long": float(selected["long"]),
            "altitude": float(selected.get("altitude", 0)),
            "timezone": selected.get("timezone", default_timezone),
        }

    if location == "random" or location.startswith("random_"):
        country_code = "DE" if location == "random" else location.split("_", 1)[1]
        coords = disturbances.get_random_location(
            country_code=country_code,
            repo_filepath=str(repo_filepath),
        )
        if coords is None:
            raise RuntimeError(f"Could not sample random location for {country_code}.")
        latitude, longitude = coords
        return f"random_{country_code}", {
            "lat": float(latitude),
            "long": float(longitude),
            "altitude": 0,
            "timezone": default_timezone,
        }

    raise KeyError(f"Unknown location '{location}'.")


def select_time_window(df: pd.DataFrame, start_date=None, end_date=None) -> pd.DataFrame:
    """Select a start-inclusive, end-exclusive time window."""
    selected = df.copy()
    if start_date is not None:
        selected = selected.loc[selected.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        selected = selected.loc[selected.index < pd.Timestamp(end_date)]
    if selected.empty:
        raise ValueError(
            f"Selected time window is empty: start={start_date}, end={end_date}"
        )
    return selected


def build_parameter_dataframe(
    building: Building,
    year: int,
    profile_name: str,
    timestep_seconds: int = 900,
    start_date=None,
    end_date=None,
    t_room_set_lower: float = 20.0,
    grid_value: float = 1.0,
    repo_filepath: str | Path = "",
) -> pd.DataFrame:
    """Build disturbances and helper input columns for a simulation."""
    repo_path = Path(repo_filepath)
    building.usage = Path(profile_name).stem

    # Reuse the existing disturbance pipeline instead of rebuilding weather,
    # internal gains, and solar gains locally in this module.
    df = disturbances.generate_disturbances(
        building=building,
        year=year,
        repo_filepath=str(repo_path),
    )
    df["T_room_set_lower"] = float(t_room_set_lower)
    df["grid"] = float(grid_value)

    if timestep_seconds is not None:
        df = df.resample(f"{int(timestep_seconds)}s").ffill()

    return select_time_window(df, start_date=start_date, end_date=end_date)


def add_night_setback(
    df: pd.DataFrame,
    night_temperature: float = 18.0,
    day_temperature: float = 20.0,
) -> pd.DataFrame:
    """Set lower comfort temperature lower during night hours."""
    df = df.copy()
    df["T_room_set_lower"] = float(day_temperature)
    night_mask = (df.index.hour < 7) | (df.index.hour >= 21)
    df.loc[night_mask, "T_room_set_lower"] = float(night_temperature)
    return df


def simulate_building_data(
    building: Building,
    parameters: pd.DataFrame,
    hp_model_name: str = "Heatpump_AW",
    ctrl_method: str = "heatcurve",
    initial_temperature: float = 20.0,
    scenario_mode: bool = False,
    **controller_kwargs,
) -> tuple[pd.DataFrame, dict]:
    """Simulate building data and merge states, control, inputs, and costs."""
    if len(parameters.index) < 2:
        raise ValueError("At least two parameter rows are required for simulation.")

    hp_model = getattr(model_hvac, hp_model_name)(
        mdot_HP=building.params.get("mdot_hp", building.mdot_hp)
    )
    timestep = int((parameters.index[1] - parameters.index[0]).total_seconds())
    simulator = Model_simulator(hp_model=hp_model, bldg_model=building, timestep=timestep)
    x_init = {key: float(initial_temperature) for key in building.state_keys}

    results = simulator.simulate(
        x_init=x_init,
        p=parameters,
        ctrl_method=ctrl_method,
        SCENARIO_MODE=scenario_mode,
        **controller_kwargs,
    )

    # One row per completed simulation step. The states/costs dataframes keep an
    # extra final index entry, while controls/parameters live on the original
    # step grid, so we realign everything to the completed-step timestamps.
    step_index = results["states"].index[1:]
    dataset = results["states"].iloc[1:].copy()

    control = results["control"].copy()
    control.index = step_index
    dataset["T_hp_sup"] = control.values

    params_used = results["parameters"].copy()
    params_used.index = step_index
    for col in params_used.columns:
        dataset[col] = params_used[col].values

    costs = results["costs"].iloc[1:].copy()
    for col in costs.columns:
        dataset[col] = costs[col].values

    return dataset, results


def generate_building_data_file(
    building_name: str,
    location: str | dict | None = "building_default",
    year: int = 2015,
    profile_name: str = "ResidentialDetached.csv",
    method: str = "4R3C",
    timestep_seconds: int = 900,
    start_date=None,
    end_date=None,
    hp_model_name: str = "Heatpump_AW",
    ctrl_method: str = "heatcurve",
    initial_temperature: float = 20.0,
    night_setback: bool = False,
    output_dir: str | Path = "data/generated/building_data",
    locations: dict[str, dict | None] | None = None,
    repo_filepath: str | Path = "",
    **controller_kwargs,
) -> tuple[Path, Path, pd.DataFrame, dict, dict]:
    """Generate and save one building-data CSV plus metadata JSON."""
    repo_path = Path(repo_filepath)
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = repo_path / output_path
    output_path.mkdir(parents=True, exist_ok=True)

    usage = Path(profile_name).stem
    params = copy.deepcopy(getattr(building_catalog, building_name))
    location_name, resolved_location = resolve_location(
        location=location,
        locations=locations,
        repo_filepath=repo_path,
    )
    if resolved_location is not None:
        params["position"] = resolved_location

    building = Building(
        params=params,
        method=method,
        usage=usage,
        mdot_hp=params.get("mdot_hp", 0.25),
    )
    parameters = build_parameter_dataframe(
        building=building,
        year=year,
        profile_name=profile_name,
        timestep_seconds=timestep_seconds,
        start_date=start_date,
        end_date=end_date,
        repo_filepath=repo_path,
    )
    if night_setback:
        parameters = add_night_setback(parameters)

    dataset, results = simulate_building_data(
        building=building,
        parameters=parameters,
        hp_model_name=hp_model_name,
        ctrl_method=ctrl_method,
        initial_temperature=initial_temperature,
        **controller_kwargs,
    )

    scenario = {
        "building": building_name,
        "location": location_name,
        "weather_year": year,
        "internal_gain_profile": profile_name,
        "method": method,
        "hp_model": hp_model_name,
        "controller": ctrl_method,
    }
    for key, value in scenario.items():
        dataset[key] = value

    filename = (
        f"building_data__{slugify(building_name)}"
        f"__{slugify(location_name)}"
        f"__{year}"
        f"__{slugify(Path(profile_name).stem)}"
        f"__{slugify(start_date)}_to_{slugify(end_date)}"
        f"__{timestep_seconds}s.csv"
    )
    csv_path = output_path / filename
    metadata_path = csv_path.with_suffix(".json")

    dataset.to_csv(csv_path, index_label="datetime")

    metadata = {
        **scenario,
        "building_area_floor_m2": building.params["area_floor"],
        "position": building.params["position"],
        "timestep_seconds": timestep_seconds,
        "start_date": start_date,
        "end_date": end_date,
        "time_window_convention": (
            "start inclusive, end exclusive; rows are completed-step sensor timestamps"
        ),
        "initial_temperature": initial_temperature,
        "night_setback": night_setback,
        "columns": list(dataset.columns),
        "rows": int(len(dataset)),
        "first_row": str(dataset.index[0]),
        "last_row": str(dataset.index[-1]),
        "weather_loader_behavior": (
            "src.disturbances.load_weather uses local DWD TRY data for exact "
            "filename matches when year=2015; otherwise PVGIS cache/API fallback."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return csv_path, metadata_path, dataset, metadata, results
