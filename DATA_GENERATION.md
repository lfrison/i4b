# Data Generation

This repo can generate synthetic building datasets by combining:

- building parameters from `data/buildings/`
- weather data loaded by `src/disturbances.py`
- internal gain profiles from `data/profiles/InternalGains/`
- solar gains computed from weather irradiance and window geometry
- controllers and simulation loops from `src/simulator.py`

The companion notebook is `notebooks/DataGeneration.ipynb`.
The reusable Python helpers live in `src/data_generation.py`.

The saved artifact is a building data file. It includes both simulated building "sensor" data and the disturbances used by the simulation. Disturbances are generated internally but are not saved as standalone disturbance-only files.

## What Can Be Varied

### Buildings

Building parameter dictionaries live in `data/buildings/` and are exported through `data.buildings`. Building parameters are taken from Tabula WebTool, which contains a large number of archetypal buildings for different countries: https://webtool.building-typology.eu/

Available bundled buildings:

- `i4c`
- `sfh_1919_1948_0_soc`, `sfh_1919_1948_1_enev`, `sfh_1919_1948_2_kfw`
- `sfh_1949_1957_0_soc`, `sfh_1949_1957_1_enev`, `sfh_1949_1957_2_kfw`
- `sfh_1958_1968_0_soc`, `sfh_1958_1968_1_enev`, `sfh_1958_1968_2_kfw`
- `sfh_1969_1978_0_soc`, `sfh_1969_1978_1_enev`, `sfh_1969_1978_2_kfw`
- `sfh_1979_1983_0_soc`, `sfh_1979_1983_1_enev`, `sfh_1979_1983_2_kfw`
- `sfh_1984_1994_0_soc`, `sfh_1984_1994_1_enev`, `sfh_1984_1994_2_kfw`
- `sfh_1995_2001_0_soc`, `sfh_1995_2001_1_enev`, `sfh_1995_2001_2_kfw`
- `sfh_2002_2009_0_soc`, `sfh_2002_2009_1_enev`, `sfh_2002_2009_2_kfw`
- `sfh_2010_2015_0_soc`, `sfh_2010_2015_1_enev`, `sfh_2010_2015_2_kfw`
- `sfh_2016_now_0_soc`, `sfh_2016_now_1_enev`, `sfh_2016_now_2_kfw`

Each building contains envelope parameters, floor area, room height, window orientation/area, heating-curve offsets, and a default location.

### Different Building Models

The thermal building models are reduced RC networks. The short names indicate the number of thermal resistances (`R`) and thermal capacities (`C`).

- `2R2C`: simplest model, small state space, useful for fast experiments
- `4R3C`: medium model depth, separates room, wall, and return-side dynamics more clearly
- `5R4C`, `6R4C`, `7R5C`: higher-order models with more internal states and more detailed thermal dynamics (only for academia)

In practice: more `R/C` elements mean more states, more parameters, and usually more detailed dynamics, but also a heavier model.

### Locations And Weather Years

Use `src.disturbances.load_weather(latitude, longitude, altitude, year, tz, repo_filepath)`.

Behavior:

- If a matching local DWD TRY file exists under `data/weather/` and `year=2015` is requested, it is used locally.
- Otherwise, the function looks for cached PVGIS JSON files under `data/weather/`.
- If no cache exists, it fetches PVGIS data and stores it under `data/weather/`.

This means arbitrary locations and years are possible as long as PVGIS provides the requested year and the machine has network access for uncached data.

Random locations are possible through the existing `disturbances.get_random_location(country_code=...)` helper. The data-generation module exposes this through location strings such as `random_DE`.

For the bundled Freiburg/i4c-style coordinates, the tracked DWD file is:

```text
data/weather/TRY2015_480252077184_Jahr.dat
```

### Internal Gain Profiles

Disturbance profiles in `data/profiles/InternalGains/` for occupancy and appliance-related internal gains are derived from standardized building-category-specific usage profiles (see DIN EN 16798-1). These profiles contain hourly values for user-related gains and appliance-related gains, including separate scaling factors for weekdays and weekends. The function get_int_gains then multiplies these standardized hourly profiles by the building floor area to obtain absolute internal heat gains in W.

- `ResidentialDetached.csv`
- `ResidentialFlat.csv`
- `Office.csv`
- `SchoolClassroom.csv`

The helper `src.disturbances.get_int_gains(...)` scales the selected profile by building floor area.

### Saved Building Data File

A generated building data file contains simulated building states/sensors. The exact state columns depend on the RC model, for example:

- `T_room`
- `T_wall`
- `T_hp_ret`
- `T_surf`, `T_mass`, or other higher-order model states
- `T_hp_sup`: controller output / supply temperature

It also includes the disturbances used in each simulation step:

- `T_amb`: ambient temperature in degC
- `Qdot_gains`: total gains in W
- `Qdot_sol`: solar gains in W
- `Qdot_int`: internal gains in W

And it includes heat-pump and comfort metrics:

- `P_el`, `E_el`, `COP`, `Qdot_th`
- `dev_neg_sum`, `dev_neg_max`, `dev_pos_sum`, `dev_pos_max`

MPC-style helper columns can also be included:

- `T_room_set_lower`: lower comfort bound in degC
- `grid`: price or grid signal

### Time Window

The notebook lets you choose `START_DATE` and `END_DATE`. The convention is:

```text
[START_DATE, END_DATE)
```

That means the start is included and the end is excluded. For example, `2015-01-01` to `2015-02-01` selects January.

Rows in the saved file are completed-step sensor timestamps: a row at `01:00` contains the building state after simulating the interval that started at `00:00`.

## Minimal Python Example

```python
from pathlib import Path

from src import data_generation as dg

repo_root = Path(".").resolve()
output_dir = repo_root / "data/generated/building_data"
output_dir.mkdir(parents=True, exist_ok=True)

locations = {
    **dg.DEFAULT_LOCATIONS,
    "custom_example": {
        "lat": 48.0252,
        "long": 7.7184,
        "altitude": 207,
        "timezone": "Europe/Berlin",
    },
}

csv_path, metadata_path, building_data, metadata, results = dg.generate_building_data_file(
    building_name="sfh_1984_1994_1_enev",
    location="custom_example",       # or "building_default" or "random_DE"
    year=2015,
    profile_name="ResidentialDetached.csv",
    method="4R3C",
    timestep_seconds=900,
    start_date="2015-01-01",
    end_date="2015-01-08",
    hp_model_name="Heatpump_AW",
    ctrl_method="heatcurve",
    initial_temperature=20.0,
    night_setback=True,
    output_dir=output_dir,
    locations=locations,
    repo_filepath=repo_root,
)
```

For explicit custom locations, pass a dictionary directly:

```python
location = {
    "lat": 48.0252,
    "long": 7.7184,
    "altitude": 207,
    "timezone": "Europe/Berlin",
}
```

## Recommended Workflow

1. Pick one or more building names from `data.buildings.__all__`.
2. Decide whether to keep the building's default location or override `params["position"]`.
3. Pick one or more weather years.
4. Pick one or more internal gain profiles.
5. Choose `START_DATE` and `END_DATE`.
6. Generate exogenous inputs in memory, run the building simulation, and save one building-data CSV per scenario.
7. Save metadata that records building, profile, location, year, timestep, date range, controller, and source assumptions.

## Things To Extend

- Add new buildings by adding a dictionary file under `data/buildings/` and exporting it in `data/buildings/__init__.py`.
- Add new internal gain profiles as semicolon-separated CSVs with the same columns as the existing profiles.
- Add new local weather readers for non-PVGIS CSV formats if you want to use measured or third-party weather files.
- Add stochastic disturbances such as randomized occupancy, window opening, shading/blind control, sensor noise, or heat-pump outages.
- Add a dedicated CLI around the notebook helper once the scenario matrix is stable.
