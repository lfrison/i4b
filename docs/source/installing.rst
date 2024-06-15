Installation
============


Download repository
-------------------

Please clone or download this repository.


Install dependencies
--------------------

- numpy
- pandas
- pvlib
- casadi (for MPC)


Build the documentation (optional)
----------------------------------
The documentation is build with sphinx automatically with every commit. To render your local changes before commiting you can install sphinx:

.. code:: console

    $ pip install sphinx sphinx-rtd-theme

Then navigate to the docs directory and generate the static html pages.

.. code:: console

    $ cd docs  
    $ make html  

The documentation can then be found in docs/build/html/index.html


