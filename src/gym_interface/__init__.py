from gymnasium.envs.registration import make, pprint_registry, register, registry, spec


POSITION_FREIBURG = {'lat': 48.0252,
                    'long': 7.7184,
                    'altitude': 207,
                    'timezone': 'Europe/Berlin'}
POSITION_RUEGEN = {'lat': 54.3611,
                    'long': 13.2909,
                    'altitude': 10,
                    'timezone': 'Europe/Berlin'}
POSITION_MUNICH = {'lat': 48.1408,
                    'long': 11.5736,
                    'altitude': 520,
                    'timezone': 'Europe/Berlin'}
POSITION_BERLIN = {'lat': 52.5200,
                    'long': 13.4050,
                    'altitude': 34,
                    'timezone': 'Europe/Berlin'}

LOCATION_DICT = {
    'freiburg': POSITION_FREIBURG,
    'ruegen': POSITION_RUEGEN,
    'munich': POSITION_MUNICH,
    'berlin': POSITION_BERLIN
}

BUILDING_NAMES = [
            # "sfh_58_68_geg",
            # "sfh_84_94_soc",
            "sfh_1919_1948", # For testing only
            "sfh_1949_1957",
            "sfh_1958_1968",
            "sfh_1969_1978",
            "sfh_1979_1983",
            "sfh_1984_1994",
            "sfh_1995_2001",
            "sfh_2002_2009",
            "sfh_2010_2015",
            "sfh_2016_now",
            "i4c",
        ]

TRAIN_BUILDINGS_NAMES = [
            # "sfh_1919_1948", # For testing only
            "sfh_1919_1948_0_soc",
            "sfh_1949_1957_0_soc",
            "sfh_1958_1968_0_soc",
            "sfh_1969_1978_0_soc",
            "sfh_1984_1994_0_soc",
            "sfh_1995_2001_0_soc",
            "sfh_2010_2015_0_soc",
            "sfh_2016_now_0_soc",
            "sfh_1919_1948_1_enev",
            "sfh_1949_1957_1_enev",
            "sfh_1958_1968_1_enev",
            "sfh_1969_1978_1_enev",
            "sfh_1984_1994_1_enev",
            "sfh_1995_2001_1_enev",
            "sfh_2010_2015_1_enev",
            "sfh_2016_now_1_enev",
            "sfh_1919_1948_2_kfw",
            "sfh_1949_1957_2_kfw",
            "sfh_1958_1968_2_kfw",
            "sfh_1969_1978_2_kfw",
            "sfh_1984_1994_2_kfw",
            "sfh_1995_2001_2_kfw",
            "sfh_2010_2015_2_kfw",
            "sfh_2016_now_2_kfw",
            "i4c",
        ]
TEST_BUILDING_NAMES = ["sfh_1979_1983_0_soc", "sfh_1979_1983_1_enev", "sfh_1979_1983_2_kfw",
                       "sfh_2002_2009_0_soc", "sfh_2002_2009_1_enev", "sfh_2002_2009_2_kfw"]

INTERNAL_GAIN_PROFILES = [  # type: ignore
            "ResidentialDetached",
            "ResidentialFlat",
            "SchoolClassroom",
            "Office",
            ]

OBSERVATION_SPACE_LIMIT = {
    'T_room': (5, 60),
    'T_wall': (5, 60),
    'T_hp_ret': (5, 65),
    'T_hp_sup': (5, 65),
    'T_amb': (-25, 45),
    'T_forecast' : (-25, 45),
    'Qdot_gains': (0, 8000),
    'goal_constraint' : (0, 3),
}

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data.buildings.i4c_building import i4c
from data.buildings.sfh_1919_1948 import sfh_1919_1948_1_enev, sfh_1919_1948_2_kfw, sfh_1919_1948_0_soc
from data.buildings.sfh_1949_1957 import sfh_1949_1957_1_enev, sfh_1949_1957_2_kfw, sfh_1949_1957_0_soc
from data.buildings.sfh_1958_1968 import sfh_1958_1968_1_enev, sfh_1958_1968_2_kfw, sfh_1958_1968_0_soc
from data.buildings.sfh_1969_1978 import sfh_1969_1978_1_enev, sfh_1969_1978_2_kfw, sfh_1969_1978_0_soc
from data.buildings.sfh_1979_1983 import sfh_1979_1983_1_enev, sfh_1979_1983_2_kfw, sfh_1979_1983_0_soc
from data.buildings.sfh_1984_1994 import sfh_1984_1994_1_enev, sfh_1984_1994_2_kfw, sfh_1984_1994_0_soc
from data.buildings.sfh_1995_2001 import sfh_1995_2001_1_enev, sfh_1995_2001_2_kfw, sfh_1995_2001_0_soc
from data.buildings.sfh_2002_2009 import sfh_2002_2009_1_enev, sfh_2002_2009_2_kfw, sfh_2002_2009_0_soc
from data.buildings.sfh_2010_2015 import sfh_2010_2015_1_enev, sfh_2010_2015_2_kfw, sfh_2010_2015_0_soc
from data.buildings.sfh_2016_now import sfh_2016_now_1_enev, sfh_2016_now_2_kfw, sfh_2016_now_0_soc

BUILDING_NAMES2CLASS = {
    "sfh_1919_1948_0_soc": sfh_1919_1948_0_soc,
    "sfh_1919_1948_1_enev": sfh_1919_1948_1_enev,
    "sfh_1919_1948_2_kfw": sfh_1919_1948_2_kfw,
    "sfh_1949_1957_0_soc": sfh_1949_1957_0_soc,
    "sfh_1949_1957_1_enev": sfh_1949_1957_1_enev,
    "sfh_1949_1957_2_kfw": sfh_1949_1957_2_kfw,
    "sfh_1958_1968_0_soc": sfh_1958_1968_0_soc,
    "sfh_1958_1968_1_enev": sfh_1958_1968_1_enev,
    "sfh_1958_1968_2_kfw": sfh_1958_1968_2_kfw,
    "sfh_1969_1978_0_soc": sfh_1969_1978_0_soc,
    "sfh_1969_1978_1_enev": sfh_1969_1978_1_enev,
    "sfh_1969_1978_2_kfw": sfh_1969_1978_2_kfw,
    "sfh_1979_1983_0_soc": sfh_1979_1983_0_soc,
    "sfh_1979_1983_1_enev": sfh_1979_1983_1_enev,
    "sfh_1979_1983_2_kfw": sfh_1979_1983_2_kfw,
    "sfh_1984_1994_0_soc": sfh_1984_1994_0_soc,
    "sfh_1984_1994_1_enev": sfh_1984_1994_1_enev,
    "sfh_1984_1994_2_kfw": sfh_1984_1994_2_kfw,
    "sfh_1995_2001_0_soc": sfh_1995_2001_0_soc,
    "sfh_1995_2001_1_enev": sfh_1995_2001_1_enev,
    "sfh_1995_2001_2_kfw": sfh_1995_2001_2_kfw,
    "sfh_2002_2009_0_soc": sfh_2002_2009_0_soc,
    "sfh_2002_2009_1_enev": sfh_2002_2009_1_enev,
    "sfh_2002_2009_2_kfw": sfh_2002_2009_2_kfw,
    "sfh_2010_2015_0_soc": sfh_2010_2015_0_soc,
    "sfh_2010_2015_1_enev": sfh_2010_2015_1_enev,
    "sfh_2010_2015_2_kfw": sfh_2010_2015_2_kfw,
    "sfh_2016_now_0_soc": sfh_2016_now_0_soc,
    "sfh_2016_now_1_enev": sfh_2016_now_1_enev,
    "sfh_2016_now_2_kfw": sfh_2016_now_2_kfw,
    "i4c": i4c,
}


# --- Gymnasium registration helpers ---
def make_env_id(
    building: str,
    hp_model: str,
    method: str,
    forecast: str = "nofc",
    version: int = 0,
):
    """Compose a deterministic Gymnasium env-id string.

    Example: i4b/RoomHeatEnv-sfh_2016_now_HPBasic_4R3C_nofc-v0
    """
    name = f"i4b/RoomHeatEnv-{building}_{hp_model}_{method}_{forecast}-v{version}"
    return name


def register_room_heat_env(
    env_id: str,
    entry_point: str = "gym_interface.room_env:RoomHeatEnv",
    **kwargs,
):
    """Register RoomHeatEnv with Gymnasium if not already present.

    Parameters are passed as default kwargs to the environment constructor.
    """
    if env_id in registry:  # type: ignore
        return
    register(
        id=env_id,
        entry_point=entry_point,
        kwargs=kwargs,
    )


def make_room_heat_env(
    building: str,
    hp_model: str,
    method: str,
    mdot_HP: float,
    internal_gain_profile: str,
    weather_forecast_steps=None,
    timestep: int = 3600,
    days: int = None,
    random_init: bool = False,
    noise_level: float = 0.0,
    version: int = 0,
):
    """Factory to create a RoomHeatEnv instance via Gymnasium make.

    This registers an env-id on the fly using the provided configuration and
    then returns a constructed environment.
    """
    if weather_forecast_steps is None:
        weather_forecast_steps = []
        forecast = "nofc"
    else:
        forecast = f"fc{len(weather_forecast_steps)}"

    env_id = make_env_id(building, hp_model, method, forecast, version)
    register_room_heat_env(
        env_id,
        building=building,
        hp_model=hp_model,
        method=method,
        mdot_HP=mdot_HP,
        internal_gain_profile=internal_gain_profile,
        weather_forecast_steps=weather_forecast_steps,
        timestep=timestep,
        days=days,
        random_init=random_init,
        noise_level=noise_level,
    )
    return make(env_id)

