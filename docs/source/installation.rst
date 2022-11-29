.. _installation:

==========================
Installation Guide
==========================

Librep can be installed in multiple ways, including via docker and pip

Pip Installation
---------------------

The package can be installed directly from pip using:

.. code-block:: bash

  pip install git+https://github.com/otavioon/wisardlib/

Development
--------------------

The package can used as development also using pip:

.. code-block:: bash

  git clone https://github.com/otavioon/wisardlib.git
  cd wisardlib
  pip install -e .[dev]


Running tests
^^^^^^^^^^^^^^

All tests are located in the tests folder. The tests run with `pytest` framework.
In development mode, tests can run using:

.. code-block:: bash

  python -m pytest tests


Building Documentation
^^^^^^^^^^^^^^^^^^^^^^^

In development mode, documentation can be built using the following commands:


.. code-block:: bash

  cd wisardlib/docs
  make clean
  make html
