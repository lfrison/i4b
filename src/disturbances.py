# -*- coding: utf-8 -*-
# Functions to generate disturbance profiles.

from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
import requests
import json
import random


def generate_disturbances_all(building_model, year=2015, timestep=900, offset_days=0, GRID_ON=True):
   '''
   Compose all disturbances:
   - T_amb (ambient temperature)
   - Q_gains (internal + solar gains)
   - T_set_low (lower setpoint temperature)
   - grid (grid signal, e.g., price)

   Parameters
   ----------
   building_model: Building

   timestep: int
      Sampling time in seconds, used for resampling
      
      
      
   GRID_ON: bool
      If True uses a grid-supportive signal; otherwise uses a constant unit signal.
      
   Returns
   -------
   p = [T_amb, Q_gains, T_set_low, grid]
   '''
   
   # 1. Load weather data as pandas df
   weather_df = load_weather(latitude = building_model.params['position']['lat'], 
                                 longitude = building_model.params['position']['long'],
                                 altitude  = building_model.params['position']['altitude'],
                                 year=year)[0:8760]

   # 2. Generate absolute heat gain profiles based on datetime, building usage and floor area.
   # Profiles: Profil_HIL, ResidentialDetached
   int_gains_df = get_int_gains(time = weather_df.index,
                                   profile_path = 'data/profiles/InternalGains/ResidentialDetached.csv',
                                   bldg_area = building_model.params['area_floor'] )

   # 3. Generate profiles of solar heat gains, based on datetime and irradiance data in the weather df,
   # and the window properties defined in the building parameters.
   Qdot_sol = get_solar_gains(weather = weather_df, bldg_params = building_model.params)

   # Combine all disturbances (solar gain, internal gains) into one disturbance dataframe:
   Qdot_gains = pd.DataFrame(Qdot_sol + int_gains_df['Qdot_tot'], columns = ['Qdot_gains']) # calculate total gains

   # Concatenate parameter vector
   p_hourly = pd.concat([weather_df['T_amb'], Qdot_gains], axis = 1) # create new df
   
   # modify T_amb
   #p_hourly['T_amb'][17:36]-=10

   # 4. Set a lower set point temperature (e.g. night setback)
   T_lower = np.ones(24)*building_model.T_room_set_lower
   #T_lower[:7] = 16
   #T_lower[21:] = 16
   #p_hourly['T_amb'] = np.ones(Qdot_sol.shape[0])
   p_hourly['T_room_set_lower'] = np.reshape([T_lower]*(int(np.ceil(Qdot_sol.shape[0]/24))),int(np.ceil(Qdot_sol.shape[0]/24)*24))

   # 5. Define the grid signal
   data_grid,data_grid_ = np.ones(Qdot_sol.shape[0]),np.ones(Qdot_sol.shape[0])
   # read grid data
   if GRID_ON: 
      data_grid = pd.read_csv(Path('data/grid/grid_signals.csv'),sep=',',header='infer')['EEX2015'].values*0.001*100 # Cent/kWh
      data_grid_ = data_grid.copy()
      data_grid_[:data_grid[::2].shape[0]] = data_grid[::2]
      data_grid_[data_grid[::2].shape[0]:] = data_grid[1::2]

      # check for NaNs: np.argwhere(np.isnan(data_grid))
      #bins = np.linspace(np.nanmin(data_grid),np.nanmax(data_grid),10)
      #grid_bins=(np.digitize(data_grid,bins))#*.1
      #data_grid=grid_bins
   p_hourly['grid'] = data_grid_[:Qdot_sol.shape[0]]
   

   # resample parameter vector to new frequency
   p = p_hourly.resample(f'{timestep}s').ffill() # resample disturbance (freq)

   # set offset days in order to shift some days (default 0 offset)
   p_hourly = p_hourly[int(offset_days*24):]
   
   return p


def generate_disturbances(building, year=2015, repo_filepath=''):
    '''Generate a dataframe of ambient temperature and internal/solar heat gain disturbances.

    Parameters
    ----------
    building : Building
        Object containing the building model and parameters

    Returns
    -------
    pandas df
        - T_amb : ambient temperature [degC]
        - Qdot_tot : total heat gains (solar and internal gains) [W]
        - Qdot_sol : solar heat gains  [W]
        - Qdot_int : internal heat gains (occupancy and appliances) [W]
    '''

    # Extract building location from building
    pos = building.params['position']
    # Load weather data as pandas df
    weather_df = load_weather(latitude = pos['lat'], longitude = pos['long'], 
                              altitude  = pos['altitude'], year=year, repo_filepath=repo_filepath) # Load, and then select the first week of the data

    # Generate absolute heat gain profiles based on datetime, building usage and floor area.
    profile_path = Path(repo_filepath, 'data', 'profiles', 'InternalGains', f'{building.usage}.csv')
    int_gains_df = get_int_gains(time = weather_df.index, 
                                 profile_path = str(profile_path),
                                 bldg_area = building.params['area_floor'] )

    # Generate profiles of solar heat gains, based on datetime and irradiance data in the weather df, 
    # and the window properties defined in the building parameters.
    Qdot_sol = get_solar_gains(weather = weather_df, bldg_params = building.params)

    # Combine all disturbances in one disturbance dataframe:
    Qdot_gains = pd.DataFrame(Qdot_sol + int_gains_df['Qdot_tot'] , columns = ['Qdot_gains']) # calculate total gains
    Qdot_gains['Qdot_sol'] = Qdot_sol
    Qdot_gains['Qdot_int'] = int_gains_df['Qdot_tot']
    p_hourly = pd.concat([weather_df['T_amb'], Qdot_gains], axis = 1) # create new df

    return p_hourly

def load_weather(latitude, longitude, altitude=0, year=2015, tz='Europe/Berlin', repo_filepath=''):
    '''
    Loads weather data, extracts ambient temperature & irradiance data and returns them in
    a pandas dataframe.
    
    If a DWD weather file exists for the given location, this file is used. Otherwise, PVGIS online data is used.
    
    An example DWD weather file for a location close to Freiburg can be found in the data/weather/ directory.
    Custom DWD weather files can be downloaded at: https://kunden.dwd.de/obt/.

    Parameters
    ----------
    latitude: float
        Latitude in deg.
        south pole...equator...north pole = -90...0...90.
    
    longitude: float
        Longitude in deg.
        west...greenwich...east = negative...0...positive.

    altitude: float, optional
        Meters over sea level. Default is 0.

    year: int, optional
        Year of weather data. Default is 2015.
        Note: Not valid for leap years.

    tz: str, optional
        Timezone. Default is 'Europe/Berlin'.

    repo_filepath: str, optional
        Base filepath of the i4c repository in case the function
        is called from outside the repository.

    Returns
    -------

    Weather data  : pandas.DataFrame
        - index   : pandas.DatetimeIndex
        - T_amb   : Ambient temperature             [°C]
        - dhi     : Diffuse horizontal irradiation  [W/m^2]
        - ghi     : Global horizontal irradiation   [W/m^2]
        - dni     : Direct normal irradiation       [W/m^2]
    '''

    # Generate filename in the same format as it would have been downloaded from dwd
    lat_str = f'{latitude:07.4f}'.replace('.', '')
    lon_str = f'{longitude:07.4f}'.replace('.', '')
    dwd_filename = f'TRY2015_{lat_str}{lon_str}_Jahr.dat'
    dwd_filepath = Path(repo_filepath, 'data', 'weather', dwd_filename)

    # If a dwd file for the given location exists, process it
    if dwd_filepath.exists():

        raw = pd.read_table(dwd_filepath, header=27, na_values='***', sep='\s+')
        raw.dropna(inplace=True)
        raw.index = pd.date_range(start=f'{year}-01-01 00:00:00', periods=len(raw), freq='h')
        
        df = pd.DataFrame(data=None, index=raw.index)
        df['T_amb'] = raw.t.values              # Ambient temperature
        df['dhi'] = raw.D.values                # Diffuse horizontal irradiation
        df['ghi'] = raw.B.values + raw.D.values # Global horizontal irradiation

        # Unfortunately the DWD files do not provide the direct normal irradiation (dni)
        # which is needed to calculate the solar heat gains. Pvlib can approximate this.
        location = pvlib.location.Location(latitude=latitude, longitude=longitude,
                                           tz=tz, altitude=altitude)
        solarposition = location.get_solarposition(df.index)
        aoi_projection = pvlib.irradiance.aoi_projection(surface_tilt=0, surface_azimuth=0,
                                                         solar_zenith=solarposition.zenith,
                                                         solar_azimuth=solarposition.azimuth)
        df['dni'] = raw.B.values / np.maximum(0.05, aoi_projection)

    # Otherwise load the data from PVGIS online API tool
    else:
        df = load_weather_pvgis(latitude, longitude, start_year=year, end_year=year, tz=tz, repo_filepath=repo_filepath)

    return df



def load_weather_pvgis(latitude, longitude, start_year=2015, end_year=2015, 
                       tz='Europe/Berlin', repo_filepath=''):
    '''
    Load weather data (ambient temperature and irradiance) via PVGIS online tool.

    Parameters
    ----------
    latitude: float
        Latitude in deg.
        south pole...equator...north pole = -90...0...90.
    
    longitude: float
        Longitude in deg.
        west...greenwich...east = negative...0...positive.

    start_year: int, optional
        Year with which weather data should start.
        Default is 2015.
        Currently ranges from 2005 to 2020.
        (2022-04-21)

    start_year: int, optional
        Year with which weather data should end.
        Default is 2015.
        Currently ranges from 2005 to 2020.
        (2022-04-21)

    tz: str, optional
        Timezone. Default is 'Europe/Berlin'.

    repo_filepath: str, optional
        Base filepath of the i4c repository in case the function
        is called from outside the repository.

    Returns
    -------

    Weather data  : pandas.DataFrame
        - index   : pandas.DatetimeIndex
        - T_amb   : Ambient temperature             [°C]
        - dhi     : Diffuse horizontal irradiation  [W/m^2]
        - ghi     : Global horizontal irradiation   [W/m^2]
        - dni     : Direct normal irradiation       [W/m^2]
    
    OR

    None (in case of error)
    '''

    # Fetch base horizontal irradiation data from PVGIS seriescalc tool
    df = _fetch_pvgis_data(latitude, longitude, start_year=start_year, end_year=end_year,
                           repo_filepath=repo_filepath)

    # Create time index (timezone-aware with UTC)
    df['time'] = pd.to_datetime(df.pop('time'), format='%Y%m%d:%H%M')
    df.set_index('time', inplace=True)

    # Calculate global horizontal irradiation
    # Global horizontal irradiation = Direct horizontal irradiation +
    #                                 Diffuse horizontal irradiation +
    #                                 Reflected horizontal irradiation
    df['ghi'] = df['Gb(i)'] + df['Gd(i)'] + df['Gr(i)']

    # Filter and rename columns
    col_dict = {
                'T2m':'T_amb',      # Ambient temperature
                'Gd(i)':'dhi',      # Diffuse horizontal irradiation
                'ghi':'ghi',        # Global horizontal irradiation
               }
    col_names = [name for name in col_dict.keys()]
    df = df[col_names]
    df.rename(columns=col_dict, inplace=True)

    # Fetch normal irradiation data from PVGIS seriescalc tool
    # (for direct normal irradiation)
    df_normal = _fetch_pvgis_data(latitude, longitude, two_axis_tracking=True,
                                  start_year=start_year, end_year=end_year,
                                  repo_filepath=repo_filepath)
    if df_normal is None:
        return None
    df['dni'] = df_normal['Gb(i)'].values

    # Convert UTC time index to given timezone
    df.index = df.index.tz_localize('utc')
    df.index = df.index.tz_convert(tz)
    df.index = df.index.tz_localize(None)

    # Remove duplicated indices which can occur during timezone conversion
    df = df[~df.index.duplicated()]

    # Resample to make sure that index is in full-hour steps
    # (e.g. not 00:10:00, 01:10:00, ...)
    df = df.resample('1h').ffill().bfill()

    return df



def _fetch_pvgis_data(latitude, longitude, two_axis_tracking=False, 
                      start_year=None, end_year=None, repo_filepath=''):
    '''
    Fetches data from PVGIS online tool.

    Parameters
    ----------
    latitude: float
        Latitude in deg.
        south pole...equator...north pole = -90...0...90.
    
    longitude: float
        Longitude in deg.
        west...greenwich...east = negative...0...positive.

    two_axis_tracking: boolean
        If irradiation data for two-axis-tracking should be fetched
        (for normal irradiation).

    start_year: int, optional
        Year with which weather data should start.
        Default is 2015.
        Currently ranges from 2005 to 2020.
        (2022-04-21)

    start_year: int, optional
        Year with which weather data should end.
        Default is 2015.
        Currently ranges from 2005 to 2020.
        (2022-04-21)

    repo_filepath: str, optional
        Base filepath of the i4c repository in case the function
        is called from outside the repository.

    Returns
    -------

    Weather data  : pandas.DataFrame
        - index   : pandas.DatetimeIndex            [UTC]
        - T_amb   : Ambient temperature             [°C]
        - dhi     : Diffuse horizontal irradiation  [W/m^2]
        - ghi     : Global horizontal irradiation   [W/m^2]
        - dni     : Direct normal irradiation       [W/m^2]

    OR

    None (in case of error)
    '''
    
    print(latitude, longitude, repo_filepath)

    # Check for already downloaded file
    str_latitude = f'{latitude:07.4f}'.replace('.', '')
    str_longitude = f'{longitude:07.4f}'.replace('.', '')
    pvgis_filename = f'pvgis_{str_latitude}_{str_longitude}'
    if two_axis_tracking:
        pvgis_filename += '_tat'
    if (start_year is not None) and (end_year is not None):
        pvgis_filename += f'_{start_year}_{end_year}'
    pvgis_filename += '.json'
    pvgis_filepath = Path(repo_filepath, 'data', 'weather', pvgis_filename)
    if pvgis_filepath.exists():
        print('Found PVGIS data file ' + pvgis_filepath.as_posix())
        with pvgis_filepath.open('r') as file:
            data = json.load(file)
        return pd.json_normalize(data['outputs']['hourly'])

    # Build query URL
    url = 'https://re.jrc.ec.europa.eu/api/v5_2/seriescalc?'
    url += f'lat={latitude}&lon={longitude}&components=1&'
    url += 'outputformat=json'
    if two_axis_tracking:
        url += '&trackingtype=2'
    if (start_year is not None) and (end_year is not None):
        url += f'&startyear={start_year}&endyear={end_year}'

    # Fetch data
    try:
        response = requests.get(url)
    except:
        print('Fetching data from PVGIS failed with exception')
        return None
    if response.status_code != 200:
        print(f'Fetching data from PVGIS failed with status code {response.status_code}')
        print(f'-> {response.content}')
        return None
    data = json.loads(response.content)

    # Save JSON data to file
    # Ensure the directory exists
    pvgis_filepath.parent.mkdir(parents=True, exist_ok=True)
    print('Save PVGIS data to file ' + pvgis_filepath.as_posix())
    with pvgis_filepath.open('w') as file:
        json.dump(data, file)

    return pd.json_normalize(data['outputs']['hourly'])



def get_random_location(country_code='DE', repo_filepath=''):
    '''
    Samples a random city location from the country specified by country_code.

    Parameters
    ----------
    country_code: string, optional
        ISO3166-1 code of country to return random location from.
        Default is 'DE' (Germany).

    repo_filepath: str, optional
        Base filepath of the i4c repository in case the function
        is called from outside the repository.

    Returns
    -------

    Tuple (latitude, longitude)

    OR

    None (in case of error)
    '''

    overpass_filename = f'overpass_{country_code}_cities.json'
    overpass_filepath = Path(repo_filepath, 'data', 'buildings', overpass_filename)
    if overpass_filepath.exists():
        # Load saved data file
        print('Found Overpass data file ' + overpass_filepath.as_posix())
        with overpass_filepath.open('r') as file:
            data = json.load(file)
    else:
        # Build query URL & data
        url = 'http://overpass-api.de/api/interpreter'
        query = f'[out:json];area["ISO3166-1"="{country_code}"];(node[place="city"](area););out;'

        # Fetch data
        try:
            response = requests.get(url, params={'data': query})
        except:
            print('Fetching data from Overpass failed with exception')
            return None
        if response.status_code != 200:
            print(f'Fetching data from Overpass failed with status code {response.status_code}')
            print(f'-> {response.content}')
            return None
        data = response.json()

        # Save JSON data to file
        print('Save Overpass data to file ' + overpass_filepath.as_posix())
        with overpass_filepath.open('w') as file:
            json.dump(data, file)

    # Get coordinates of random location
    cities = data.get('elements', [])
    city = random.choice(cities)

    return city['lat'], city['lon']



def get_solar_gains(weather, bldg_params, albedo = 0.2):
    ''' Iterate over windows and calculate total heatflow due to solar irradiation

    Parameters
    ----------
    weather: pandas.Dataframe
        - index  : pandas.DatetimeIndex
        - dhi    : Diffuse horizontal irradiance [W/m^2]
        - ghi    : Global horizontal irradiance  [W/m^2]
        - dni    : Direct normal irradiance      [W/m^2]
    
    building_params: dict of dicts
        - windows  (list of dicts) : geometry and material properties of the windows
        - position (dict)          : latitude, longitude and elevation and timezone of the building

        The structure for three example buildings is shown in data/buildings/.
    
    albedo: float - optional
        Measure of diffuse reflection of solar radiation, grass = 0.2, asphalt = 0.04, snow = 0.8
        
        
    Returns
    -------

    pandas timeseries
        Solar heat gains [W]


    Notes
    -----

    !ToDo:

    - return individual aswell as sum of all solar heatflows.
    
    '''
    
    Qdot_sol = 0

    windows = bldg_params['windows']
    pos = bldg_params['position']

    for window in windows:

        tilt = int(min(180, max(0, window['tilt'])))
        azimuth = int(min(360, max(0, window['azimuth'])))
        albedo = min(1.0, max(0.0, albedo))

        location = pvlib.location.Location(latitude = pos['lat'],
                                           longitude = pos['long'],
                                           tz = pos['timezone'],
                                           altitude = pos['altitude'])
        solarposition = location.get_solarposition(weather.index)
        airmass   = location.get_airmass(times = weather.index, solar_position = solarposition)
        dni_extra = pvlib.irradiance.get_extra_radiation(weather.index)
        
        # Calculate plane of array irradiance
        poa = pvlib.irradiance.get_total_irradiance(surface_tilt = tilt,
                                                    surface_azimuth = azimuth,
                                                    solar_zenith = solarposition.zenith,
                                                    solar_azimuth = solarposition.azimuth,
                                                    dni = weather['dni'],
                                                    ghi = weather['ghi'],
                                                    dhi = weather['dhi'],
                                                    dni_extra = dni_extra,
                                                    airmass = airmass.airmass_absolute,
                                                    albedo = albedo,
                                                    model = 'perez',
                                                    )
        
        # Extract global poa irradiance from poa df 
        poa_global = poa.poa_global.fillna(0)

        # Caculate solar gains for window and store sum of all windows
        Qdot_sol += window['area'] * window['g_value'] * poa_global * (1 - window['c_frame'] * window['c_shade']) # [W]

    return Qdot_sol



def get_int_gains(time, profile_path, bldg_area = None):
    ''' Function to calculate the internal gains of a building from the profiles given
    in DIN EN 16798-1.
    
    Parameters
    ----------
    time: pandas.DatetimeIndex
        Timestamps as datetime index
        
    profile_path: string
        Path to csv-file with internal gains profile 
        e. g.:  data/profiles/InternalGains/ResidentialDetached.csv
                                            
    bldg_area: float - optional
        Heated living space [m^2]
        
    Returns
    -------
    Internal heat gains : pandas.DataFrame
        - index            : pandas.DatetimeIndex
        - qdot_oc          : Specific internal gains through occupancy [W/m^2]
        - qdot_app         : Specific internal gains through appliances [W/m^2]
        - qdot_tot         : Total specific internal gains [W/m^2]
        - Qdot_oc          : Absolute internal gains through occupancy [W]
        - Qdot_app         : Absolute internal gains through appliances [W]
        - Qdot_tot         : Total absolute internal gains [W]
    '''
    if isinstance(profile_path, str):
        profile = pd.read_csv(profile_path,
                            delimiter = ";",
                            index_col = "hour")
    else:
        profile = profile_path
        
    # initialization
    df = pd.DataFrame(index = time)
    df["qdot_oc"] = 0.
    df["qdot_app"] = 0.
    
    mask = time.dayofweek < 5
    workday_idx = time[mask]
    weekend_idx = time[~mask]
    df.loc[workday_idx, "qdot_oc"] = (profile["user [W/m^2]"][workday_idx.hour] * profile["workday_user"][workday_idx.hour]).values
    df.loc[workday_idx, "qdot_app"] = (profile["appliances [W/m^2]"][workday_idx.hour] * profile["workday_appliances"][workday_idx.hour]).values
    
    df.loc[weekend_idx, "qdot_oc"] = (profile["user [W/m^2]"][weekend_idx.hour] * profile["weekend_user"][weekend_idx.hour]).values
    df.loc[weekend_idx, "qdot_app"] = (profile["appliances [W/m^2]"][weekend_idx.hour] * profile["weekend_appliances"][weekend_idx.hour]).values

    # total specific internal gains in W/m^2
    df["qdot_tot"] = df["qdot_oc"] + df["qdot_app"]

    if bldg_area != None:
        # absolute internal gains in W
        df["Qdot_oc"]  = df["qdot_oc"]  * bldg_area
        df["Qdot_app"] = df["qdot_app"] * bldg_area
        df["Qdot_tot"] = df["qdot_tot"] * bldg_area

    return df


# CALCULATION OF BOREHOLE TEMPERATURE
def t_borehole_from_amb(t_amb):
    '''
    Estimate the source temperature from the ambient temperature.
    
    Parameters
    ----------
    t_amb : float
        Ambient temperature [degC]
    
    Returns
    -------
    float
        Source temperature [degC]
    
    Notes
    -----
    Function originally written by Felix Ohr. The parameters were estimated
    using least squares with a softl1 loss function. The data was gathered 
    during a monitoring project performed by Fraunhofer ISE.
    '''
   
    t_src = 4.918 * np.tanh(0.095 * (t_amb - 10.545)) + 8.261
    return t_src

def t_borehole_from_hoy(hour_of_year):
    '''
    Estimate the source temperature from the hour of the year
    
    Parameters
    ---------
    hour_of_year: int or list of ints
        Hour of the year
    
    Returns
    -------
    float or list of floats
        Source temperature [degC]
        
    Notes
    -----
    credits: GetSolPV simulation-framework by Hans-Martin Henning
    '''
    
    temp_soil = 8 # constant mean ground temperature °C
    t_src = temp_soil + 8 * np.sin(2*np.pi * (hour_of_year - 1700) / 8760)
    return t_src

# CALCULATION OF GROUND TEMPERATURE
def t_ground_from_hoy(hour_of_year):
    '''Approximation of return fluid temperatur from a ground collector from hour of the year
    
    Parameters
    ---------
    hour_of_year: int or list of ints
        Hour of the year
    
    Returns
    -------
    float or list of floats
        Ground temperature [degC]
        
    Notes
    -----
    credits: GetSolPV simulation-framework by Hans-Martin Henning
    original function and parameters:
    :math:`T_{ground} = T_{soil} + 8 \cdot sin(2 \cdot \pi \cdot (h_{ofyear} - 1700) / 8760)`
    '''
    t_ground_const = 8 # constant mean ground temperature °C
    t_ground = t_ground_const + 8 * np.sin(2*np.pi * (hour_of_year - 3200) / 8760)
    return t_ground

def t_ground_from_hoy_depth(hour_of_year, depth):
    '''Approximation of return fluid temperatur from a ground collector from depth
    
    Parameters
    ----------
    hour_of_year: int or list of ints
        Hour of the year

    depth: float or list of floats
        Depth [m] 
 
    Returns
    -------
    float or list of floats
        Ground temperature [degC]

    Notes
    -----
    credits: 
        https://nrc-publications.canada.ca/eng/view/ft/?id=386ddf88-fe8d-45dd-aabb-0a55be826f3f

    '''
    hoy = hour_of_year # hour of the year [hour]
    t_avg = -2 # Average surface temperature [°C]
    t_amp = 18 # Difference between max and min temperature for the period [°C]
    diff_th = 0.01 # Thermal diffusivity, wet sand = 0.01 [cm^2/sec]
    t_tot = 8760 # time for one complete cycle [hours]
    t_ground = t_avg + t_amp * np.exp(-depth*np.sqrt(np.pi/(diff_th * t_tot))
                                      *np.cos(2*np.pi*hoy/t_tot)-depth*
                                      np.sqrt(np.pi/(diff_th * t_tot)))
    return t_ground

def t_ground_from_amb(t_ambient):
    '''Approximation of return fluid temperatur from a ground collector from abient temperature
    
    parameters
    ----------
    t_ambient: float or list of floats
        Ambient temperature [degC]
        
    returns
    -------
    float or list of floats
        Ground temperature [degC]
        
    notes
    -----
    credits: @fohr
    original function and parameters fitted for borehole:
    :math:`T_{source} = 4.918 \cdot tanh(0.095 \cdot (T_{amb} - 10.545)) + 8.261`
    '''
    t_ground = 6.645 * np.tanh(0.188 * (t_ambient - 9.177)) + 7.872
    return t_ground


def t_ground_from_amb_seasonal(t_ambient, hour_of_year):
    '''Approximation of return fluid temperatur from a ground collector from ambient temperature and hour of the year
    
    Parameters
    ----------
    t_ambient: float or list of floats
        Ambient temperature [degC]
    hour_of_year: int or list of ints
        Hour of the year
        
    Returns
    -------
    float or list of floats
        Ground temperature [degC]
    '''
    if (hour_of_year > 744) and (hour_of_year < 5136):
        t_ground = 7.163 * np.tanh(0.155 * (t_ambient - 12.479)) + 7.517
    else:
        t_ground = 7.252 * np.tanh(0.183 * (t_ambient - 6.374)) + 8.202
    return t_ground