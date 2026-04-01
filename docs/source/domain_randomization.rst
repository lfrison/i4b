Domain Randomization
====================

Domain randomization varies simulation parameters across episodes or within
episodes so that RL policies generalise to a range of building configurations.
Events are defined as ``EventSpec`` objects and managed by an ``EventManager``.

Event Triggers
--------------

Each event fires at one of three points in the simulation lifecycle:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - ``when``
     - Fires at
     - Use case
   * - ``on_start``
     - Once, during environment construction
     - Set up a fixed distribution of buildings across parallel envs
   * - ``on_reset``
     - Every call to ``env.reset()`` (or ``auto_reset`` for done envs)
     - Per-episode variation (insulation, capacity, setpoints)
   * - ``on_interval``
     - Every *N* simulation steps (requires ``interval`` parameter)
     - Slow drifts during an episode (COP degradation, occupancy changes)

Supported Distributions
-----------------------

Distributions can be specified either programmatically or via YAML config.

.. list-table::
   :header-rows: 1
   :widths: 18 35 47

   * - Distribution
     - Parameters
     - Description
   * - ``uniform``
     - ``low``, ``high``
     - Uniform sampling in [low, high]. **Default** if omitted.
   * - ``loguniform``
     - ``low``, ``high`` (both > 0)
     - Log-uniform: equal probability per decade. Good for multiplicative
       factors (e.g. COP scale, thermal mass).
   * - ``normal``
     - ``mean``, ``std``
     - Gaussian sampling (unbounded). Use ``clip`` to bound the output.
   * - ``constant``
     - ``value``
     - Always returns the same value. Useful for ablations or overrides.
   * - ``choice``
     - ``values`` (non-empty list)
     - Uniformly samples one element from the list.

Post-Sampling Options
~~~~~~~~~~~~~~~~~~~~~

These options can be combined with any distribution:

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Option
     - Format
     - Description
   * - ``clip``
     - ``[min, max]``
     - Clamps the sampled value to the given range. Applied before ``dtype``.
   * - ``dtype``
     - ``int``, ``float``, or ``bool``
     - Casts the sampled value. ``int`` rounds to nearest integer.

Targetable Parameters
---------------------

Any key in the building parameter dict can be targeted. Dot notation
(e.g. ``building.params.area_floor``) is supported for nested access;
intermediate dicts are created automatically if missing.

Building Envelope (raw parameters)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are the primary physical parameters defined in each building file
(``data/buildings/*.py``):

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Key
     - Unit
     - Description
   * - ``H_ve``
     - W/K
     - Ventilation heat transfer coefficient
   * - ``H_tr``
     - W/K
     - Transmission heat transfer coefficient (total)
   * - ``H_tr_light``
     - W/K
     - Transmission through lightweight components (windows, doors)
   * - ``c_bldg``
     - Wh/(m²K)
     - Specific thermal capacity of the building
   * - ``area_floor``
     - m²
     - Net floor area
   * - ``height_room``
     - m
     - Room height

Heating System
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Key
     - Unit
     - Description
   * - ``mdot_hp``
     - kg/s
     - Heat pump mass flow rate
   * - ``T_offset``
     - K
     - Heating curve offset (supply temp = f(T_amb) + T_offset)
   * - ``T_amb_lim``
     - °C
     - Ambient temperature above which heating is disabled

Control Setpoints
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Key
     - Unit
     - Description
   * - ``T_room_set_lower``
     - °C
     - Lower comfort band boundary (default: 20.0)
   * - ``T_room_set_upper``
     - °C
     - Upper comfort band boundary (default: 22.0)
   * - ``cop_scale``
     - —
     - Multiplier on COP curve (default: 1.0)

Window Properties
~~~~~~~~~~~~~~~~~

Each building has a ``windows`` list. Individual window dicts contain:

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Key
     - Unit
     - Description
   * - ``area``
     - m²
     - Window area
   * - ``tilt``
     - °
     - Surface tilt (90 = vertical)
   * - ``azimuth``
     - °
     - Surface azimuth (180 = south)
   * - ``g_value``
     - —
     - Total solar energy transmittance
   * - ``c_frame``
     - —
     - Frame fraction (0–1)
   * - ``c_shade``
     - —
     - Shading reduction factor (0–1)

Non-Dynamics Keys (Performance Optimisation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The vectorised environment (``RoomHeatVecEnv``) tracks which parameters affect
the RC state-space matrices. When domain randomization only changes
**non-dynamics keys**, the expensive matrix recomputation (GPU ``expm``) is
skipped:

- ``T_amb_lim``
- ``T_offset``
- ``T_room_set_lower``
- ``T_room_set_upper``
- ``cop_scale``

All other parameter changes trigger a rebuild of the discretised matrices
``(Ad, Bd, Cd)``.

Python API
----------

.. code-block:: python

   from src.randomization import EventSpec, uniform, loguniform, normal

   events = [
       # Vary insulation at the start of each episode
       EventSpec(
           when="on_reset",
           fn=uniform("H_tr", 50, 300),
       ),
       # Log-uniform thermal mass variation per episode
       EventSpec(
           when="on_reset",
           fn=loguniform("c_bldg", 30, 80),
       ),
       # Slowly drift COP every 96 steps (~1 day at 15-min timestep)
       EventSpec(
           when="on_interval",
           fn=normal("cop_scale", 1.0, 0.05),
           interval=96,
       ),
   ]

   from src.gym_interface.vec_env import RoomHeatVecEnv

   env = RoomHeatVecEnv(
       num_envs=1024,
       building="sfh_2016_now_0_soc",
       hp_model="Heatpump_AW",
       method="4R3C",
       mdot_HP=0.25,
       internal_gain_profile="data/profiles/InternalGains/ResidentialDetached.csv",
       device="gpu",
       randomization_events=events,
       randomization_seed=42,
   )

YAML Configuration
------------------

Events can also be specified entirely in YAML:

.. code-block:: yaml

   domain_randomization:
     seed: 42
     events:
       # Vary insulation per episode
       - when: on_reset
         target: H_tr
         distribution: uniform
         low: 50
         high: 300

       # Log-uniform thermal mass
       - when: on_reset
         target: c_bldg
         distribution: loguniform
         low: 30
         high: 80

       # COP drift every day
       - when: on_interval
         interval: 96
         target: cop_scale
         distribution: normal
         mean: 1.0
         std: 0.05
         clip: [0.8, 1.2]

       # Fixed comfort setpoint override
       - when: on_start
         target: T_room_set_lower
         distribution: constant
         value: 21.0

       # Randomly pick a building mass flow
       - when: on_reset
         target: mdot_hp
         distribution: choice
         values: [0.15, 0.20, 0.25, 0.30]
         dtype: float

.. code-block:: python

   from src.gym_interface.config import make_env_from_config

   env = make_env_from_config("path/to/config.yaml")
