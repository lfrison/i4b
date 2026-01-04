from gymnasium.envs.registration import make, pprint_registry, register, registry, spec
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

    Example: RoomHeatEnv-sfh_2016_now_Heatpump_AW_4R3C_nofc-v0
    """
    name = f"RoomHeatEnv-{building}_{hp_model}_{method}_{forecast}-v{version}"
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
    delta_t: int = 900,
    days: int = None,
    random_init: bool = False,
    goal_based: bool = False,
    goal_temp_range: tuple = (19.0, 28.0),
    temp_deviation_weight: float = 0.0,
    noise_level: float = 0.0,
    version: int = 0,
):
    """Factory to create a RoomHeatEnv instance via Gymnasium make.

    This registers an env-id on the fly using the provided configuration and
    then returns a constructed environment.
    
    Parameters
    ----------
    building : str
        Building name/key.
    hp_model : str
        Heat pump model name.
    method : str
        Building model method (e.g., '4R3C').
    mdot_HP : float
        Mass flow rate in kg/s.
    internal_gain_profile : str
        Path to internal gains profile.
    weather_forecast_steps : list, optional
        List of forecast steps.
    delta_t : int
        Simulation timestep in seconds (default: 900).
    days : int, optional
        Episode length in days.
    random_init : bool
        Whether to randomize initial conditions.
    goal_based : bool
        Enable goal-based learning mode.
    goal_temp_range : tuple
        Range for goal temperature (min, max) in °C.
    temp_deviation_weight : float
        Weight for temperature deviation in reward.
    noise_level : float
        Observation noise standard deviation.
    version : int
        Environment version number.
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
        delta_t=delta_t,
        days=days,
        random_init=random_init,
        goal_based=goal_based,
        goal_temp_range=goal_temp_range,
        temp_deviation_weight=temp_deviation_weight,
        noise_level=noise_level,
    )
    return make(env_id)

