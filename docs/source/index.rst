
Welcome to I4C's documentation!
=================================

This python project can be used to quickly generate reduced order building models.
A simulator class provides a high level interface to perform one and multistep simulations that return the next state(s) of the building (temperatures)
and indicators for comfort levels and the energy demand. This interface can be used to evaluate and test different control strategies. 
Simple heat pump models, based on performance curves can be used as heating systems.
Disturbance profiles for the ambient temperature, internal heat gains by occupancy, and appliances as well as solar heat gains can be generated as pandas.Timeseries.
The parameters for buildings of different refurbishment states are provided in the /data/buildings directory. 
The disturbances profiles for different building types can be found in the /data/profiles directory.
A simple rule based heating curve controller can be used as the reference case at the lower end. A MPC controller provides a reference case at the upper end.


.. toctree::
   :maxdepth: 2
   :caption: Overview

   installing
   quickstart


.. toctree::
   :maxdepth: 2
   :caption: Modules

   building
   hvac
   disturbances
   controller
   simulator


Roadmap
-------
- The dynamics of the heat distribution system shall be defined as a separate xRxC model.
- Custom buildings shall be defined in a .yml file.

Limitations
-----------
   - no pipes, no losses
   - convective and conductive hvac heatflow is combined :math:`\dot{Q}_{rad,conv}`
   - no thermostat model
   - no solar gains on facade.
   - no shading devices
   - solar gains through windows are not split in connective and radiative. 
   - interior surfaces temperatures are not available
   - single zone building model
   - infiltration and ventilation losses are combined and constant for each timestep
   - no domestic hot water included
   - massflow of hvac system is constant










Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
