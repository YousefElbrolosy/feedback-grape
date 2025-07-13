Tutorials
=========

Welcome to the feedbackGRAPE tutorials! These interactive notebooks will guide you through the key concepts and practical applications of GRAPE (Gradient Ascent Pulse Engineering) optimization.

.. grid:: 4
   :gutter: 2

   .. grid-item-card:: GRAPE Tutorials
      :link: GRAPE-tutorials/index
      :link-type: doc
      :class-header: bg-primary text-white

      Learn the fundamentals of GRAPE optimization for quantum control. Covers basic concepts, time-dependent Hamiltonians, and various optimization techniques.


   .. grid-item-card:: FeedbackGRAPE Tutorials  
      :link: feedbackGRAPE-tutorials/index
      :link-type: doc
      :class-header: bg-success text-white

      Explore advanced feedback-based GRAPE techniques for enhanced quantum control with real-time optimization.

   .. grid-item-card:: QubitCavity API Tutorials  
      :link: QubitCavity_api_tutorials/index
      :link-type: doc
      :class-header: bg-success text-white

      Use an easier more intuitive way to model your qubit in a cavity systems.

   .. grid-item-card:: Examples Using Qutip's operators  
      :link: qutip_operators_tutorials/index
      :link-type: doc
      :class-header: bg-success text-white

      Use qutip's extensive library of operators and states to construct your system dynamics with the jaxify function from feedback_grape.utils.operators

.. toctree::
   :maxdepth: 2
   :hidden:   

   GRAPE-tutorials/index
   feedbackGRAPE-tutorials/index
   QubitCavity_api_tutorials/index
   qutip_operators_tutorials/index

Quick Start Guide
-----------------

1. **New to quantum control?** Start with :doc:`GRAPE-tutorials/time_indep_tutorial`
2. **Familiar with GRAPE?** Jump to :doc:`feedbackGRAPE-tutorials/example_A`  
3. **Looking for specific examples?** Browse the sections above

.. seealso::
   
   :doc:`../fgrape`
      API documentation for feedbackGRAPE functions
   
   :doc:`../grape`  
      API documentation for standard GRAPE functions