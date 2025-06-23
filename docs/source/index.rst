.. feedbackGRAPE documentation master file, created by
   sphinx-quickstart on Fri Mar 28 18:13:40 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

feedbackGRAPE documentation
============================

Installation
============

We recommend using a fresh conda environment with python >=3.11 to avoid conflicts with other packages.

Then, to install feedbackGRAPE, run the following command:

.. code-block:: bash

   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ feedbackGRAPE

then simply import as follows:

.. code-block:: python

   from feedback_grape.grape import (
    optimize_pulse,
    plot_control_amplitudes,
    sesolve,
    fidelity
   )
   
   from feedback_grape.utils.operators import identity, destroy, sigmap, sigmaz
   from feedback_grape.utils.tensor import tensor
   from feedback_grape.utils.states import basis


Refer to the documentation to see the full list of available functions in :doc:`grape`, :doc:`fgrape` and in :doc:`utils`.

Click here to get started with some example tutorials for GRAPE: :doc:`tutorials/GRAPE-tutorials/index`.

Click here to get started with some example tutorials for feedbackGRAPE: :doc:`tutorials/feedbackGRAPE-tutorials/index`.

.. toctree::
   :caption: Contents:
   :maxdepth: 2
   
   fgrape.rst
   grape.rst
   utils.rst
   tutorials/index.rst

