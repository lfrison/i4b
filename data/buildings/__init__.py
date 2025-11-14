"""Building parameter definitions for various single family houses and i4c building.

This module provides building configurations from different construction periods
with various renovation standards:
- 0_soc: State of construction (original)
- 1_enev: EnEV renovation standard
- 2_kfw: KfW renovation standard
"""

# i4c building - Two family house, KfW 40+
from .i4c_building import i4c

# Single family houses (1919-1948)
from .sfh_1919_1948 import (
    sfh_1919_1948_0_soc,
    sfh_1919_1948_1_enev,
    sfh_1919_1948_2_kfw,
)

# Single family houses (1949-1957)
from .sfh_1949_1957 import (
    sfh_1949_1957_0_soc,
    sfh_1949_1957_1_enev,
    sfh_1949_1957_2_kfw,
)

# Single family houses (1958-1968)
from .sfh_1958_1968 import (
    sfh_1958_1968_0_soc,
    sfh_1958_1968_1_enev,
    sfh_1958_1968_2_kfw,
)

# Single family houses (1969-1978)
from .sfh_1969_1978 import (
    sfh_1969_1978_0_soc,
    sfh_1969_1978_1_enev,
    sfh_1969_1978_2_kfw,
)

# Single family houses (1979-1983)
from .sfh_1979_1983 import (
    sfh_1979_1983_0_soc,
    sfh_1979_1983_1_enev,
    sfh_1979_1983_2_kfw,
)

# Single family houses (1984-1994)
from .sfh_1984_1994 import (
    sfh_1984_1994_0_soc,
    sfh_1984_1994_1_enev,
    sfh_1984_1994_2_kfw,
)

# Single family houses (1995-2001)
from .sfh_1995_2001 import (
    sfh_1995_2001_0_soc,
    sfh_1995_2001_1_enev,
    sfh_1995_2001_2_kfw,
)

# Single family houses (2002-2009)
from .sfh_2002_2009 import (
    sfh_2002_2009_0_soc,
    sfh_2002_2009_1_enev,
    sfh_2002_2009_2_kfw,
)

# Single family houses (2010-2015)
from .sfh_2010_2015 import (
    sfh_2010_2015_0_soc,
    sfh_2010_2015_1_enev,
    sfh_2010_2015_2_kfw,
)

# Single family houses (2016-now)
from .sfh_2016_now import (
    sfh_2016_now_0_soc,
    sfh_2016_now_1_enev,
    sfh_2016_now_2_kfw,
)

# Export all buildings
__all__ = [
    # i4c building
    'i4c',
    # SFH 1919-1948
    'sfh_1919_1948_0_soc',
    'sfh_1919_1948_1_enev',
    'sfh_1919_1948_2_kfw',
    # SFH 1949-1957
    'sfh_1949_1957_0_soc',
    'sfh_1949_1957_1_enev',
    'sfh_1949_1957_2_kfw',
    # SFH 1958-1968
    'sfh_1958_1968_0_soc',
    'sfh_1958_1968_1_enev',
    'sfh_1958_1968_2_kfw',
    # SFH 1969-1978
    'sfh_1969_1978_0_soc',
    'sfh_1969_1978_1_enev',
    'sfh_1969_1978_2_kfw',
    # SFH 1979-1983
    'sfh_1979_1983_0_soc',
    'sfh_1979_1983_1_enev',
    'sfh_1979_1983_2_kfw',
    # SFH 1984-1994
    'sfh_1984_1994_0_soc',
    'sfh_1984_1994_1_enev',
    'sfh_1984_1994_2_kfw',
    # SFH 1995-2001
    'sfh_1995_2001_0_soc',
    'sfh_1995_2001_1_enev',
    'sfh_1995_2001_2_kfw',
    # SFH 2002-2009
    'sfh_2002_2009_0_soc',
    'sfh_2002_2009_1_enev',
    'sfh_2002_2009_2_kfw',
    # SFH 2010-2015
    'sfh_2010_2015_0_soc',
    'sfh_2010_2015_1_enev',
    'sfh_2010_2015_2_kfw',
    # SFH 2016-now
    'sfh_2016_now_0_soc',
    'sfh_2016_now_1_enev',
    'sfh_2016_now_2_kfw',
]

