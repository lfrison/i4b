# i4b - Intelligence for Buildings 

## Advanced Building Heat Pump Control Testing Framework

This project features a light-weight Python-based thermal simulation framework for heat pump operation in buildings. It is particularly useful for the following tasks:
- Serves as evaluation framework for testing different building heat pump control strategies
- Features different detailed reduced-order models for implementing building heat pump controllers (e.g., MPC)
- Serves as synthetic data generation framework, e.g. for ML-based controllers or anomaly detection

This Python project facilitates the quick generation of reduced order building models. It features a simulator class providing a high-level interface for one-step and multi-step simulations. These simulations return the next state(s) of the building (temperatures), indicators for comfort levels, and energy demand. This interface can be used to evaluate and test different control strategies. Interface to RL, MPC, and reference heat curve controller is provided. Implementation for MPC and reference heat curve is given. The project includes simple heat pump models based on performance curves for heating systems, and disturbance profiles for ambient temperature, internal heat gains by occupancy, and solar heat gains.

![I4C_Grafik](https://github.com/lfrison/i4b/assets/104891971/65cce2cf-8801-45ba-811d-a965a0115c08)

## Table of Contents

1. [Install Dependencies](#install-dependencies)
2. [Building Model](#building-models)
3. [Disturbances](#disturbances)
4. [Controller](#controller)
5. [Usage](#usage)
6. [License](#license)
7. [Acknowledgements](#acknowledgements)

## Install Dependencies

Ensure you have the following dependencies installed:

- `numpy`
- `pandas`
- `pvlib`
- `casadi` (for MPC)

You can install these dependencies using pip:

`pip install numpy pandas pvlib casadi`

## Building Models

Geometrical and physical parameters that specify different buildings are available in the `/data/buildings` directory, including:

- `i4c_building`: an energy-efficient KFW 40+ house. Source: EnEV-Nachweis.
- `sfh_58_68_geg`: a single-family home constructed between 1958-68, with an envelope refurbished according to GEG regulations. Source: Tabula.
- `sfh_84_94_soc`: a single-family home constructed between 1984-94, with an envelope in the state of construction. Source: Tabula.

Building models are reduced order models based on RC networks. Different degrees of modelling depths can be selected, e.g., 2R2C, 4R3C.

Building specifications can be added by creating separate `.py` files containing a dictionary with the following parameters, which are accessible via the [TABULA web tool](https://webtool.building-typology.eu):

- `H_ve`: Heat transfer coefficient for ventilation (indoors --> ambient) [W/K]
- `H_tr`: Heat transfer coefficient for transmission (indoors --> ambient) [W/K]
- `H_tr_light`: Heat transfer coefficient for light building components (indoors --> ambient) [W/K]
- `c_bldg`: Specific heat capacity of the building [Wh/m²/K]
- `area_floor`: Conditioned floor area [m²]
- `height_room`: Average height of the heated zone [m³]
- `T_offset`: Optional parameter, used for heating curve control [°C]
- `windows`: List of dictionaries containing:
  - `area`: Absolute window area [m²]
  - `tilt`: Tilt angle [degree]
  - `azimuth`: Azimuth angle, 0 = North, 180 = South [degree]
  - `g-value`: Total solar heat gain factor [-]
  - `c_frame`: Fraction of window that is opaque due to the frame [-]
  - `c_shade`: Shading factor due to external influences, e.g., trees [-]
- `position`: Dictionary containing:
  - `lat`: Latitude [degree]
  - `long`: Longitude [degree]
  - `altitude`: Altitude above sea level [m]
  - `timezone`: Timezone as defined [here](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones)

## Disturbances

Functions are provided to generate disturbance profiles for:

- Ambient Temperature [°C]
- Internal heat gains by occupancy and appliances [W]
- Solar heat gains through transparent building elements [W]

These functions generate `pandas` dataframes, where the columns correspond to individual disturbances, and the index is a `pandas.DatetimeIndex`.

To manually generate disturbance profiles, start with the weather data and give each entry a `pandas.DatetimeIndex`. The datetime index is required to generate the internal and solar heat gain profiles. For the solar heat gain profiles, the weather dataframe should also contain information about solar irradiation.

## Controller

Heating curves are a widely used approach to control heating temperatures based on outdoor temperature to maintain constant room temperature. In this framework, the heating curve controller can be used as a baseline.
MPC controller using [CasADi](https://web.casadi.org/is available).
Interface to RL controller.

![i4c-cntroller](https://github.com/lfrison/i4b/assets/104891971/87e45eff-9fea-4771-a8bc-ead693e322ee)


## License

Licensed under the terms of the BSD 3-Clause License.
