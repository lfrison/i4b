
Disturbances
------------

Functions to generate disturbance profiles of:

- Ambient Temperature [degC]
- Internal heat gains by occupancy and appliances [W] 
- Solar heat gains through transparent building elements [W]

The functions generate pandas dataframes, where the columns correspond to the individual disturbances and the index is a pandas.DatetimeIndex.  

To manually generate distrubance profiles we suggest, that you start with the weather data and give each entry a pandas.DatetimeIndex.
The datetime index is needed to generate the internal- and solar heat gain profiles.
For the solar heat gains profiles the weather df additionally has to contain information about the solar irradiation.


.. automodule:: disturbances 
   :members:
   :undoc-members:
   :show-inheritance:

