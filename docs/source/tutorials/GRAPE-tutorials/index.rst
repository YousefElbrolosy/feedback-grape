GRAPE Tutorials
===============

These tutorials introduce you to GRAPE (Gradient Ascent Pulse Engineering) for quantum optimal control. Work through them sequentially for the best learning experience.

Getting Started
---------------

.. toctree::
   :maxdepth: 1

   time_indep_tutorial
   hadamard_example_tutorial

Advanced Topics
----------------

.. toctree::
   :maxdepth: 1

   time_dep_tutorial
   density_matrix_tutorial

Preparing a cat state
----------------------

.. toctree::
   :maxdepth: 1

   cat_state_l_bfgs_tutorial
   cat_state_adam_tutorial


.. note::
   Here we note that sometimes different optimizers may navigate the cost landscape better, as you notice from the ability of the l-bfgs optimization algorithm to converge at a global minima while using adam led to being stuck at a local minima.

Open Systems
-------------

.. toctree::
   :maxdepth: 1

   dissipative_tutorial
   dissipative_tutorial_2

.. note::
   The first dissipative_tutorial shows you how to prepare a gate in the presence of dissipation using Liouvillian superoperators, while the second example shows how to optimize control pulses for dissipative state preparation by passing hamiltonians and jump operators.