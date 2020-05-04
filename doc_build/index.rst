Cellfinder
================================================

Introduction:
--------------
Cellfinder is a collection of tools from the
`Margrie Lab <https://www.sainsburywellcome.org/web/groups/margrie-lab/>`_
and others at the
`Sainsbury Wellcome Centre <https://www.sainsburywellcome.org/web/>`_
for the analysis of whole-brain imaging data such as
`serial-section imaging <https://sainsburywellcomecentre.github.io/OpenSerialSection/>`_
and lightsheet imaging in cleared tissue.


The aim is to provide a single solution for:
   * Cell detection (initial cell candidate detection and refinement using deep learning implemented in `Keras <https://keras.io/>`_)
   * Atlas registration (using `aMAP <https://github.com/SainsburyWellcomeCentre/amap-python/>`_)
   * Analysis of cell positions in a common space


User Guide
--------------

.. toctree::
   :maxdepth: 1

   main/user_guide/install.md
   main/user_guide/run.md
   main/user_guide/training.md
   main/user_guide/troubleshooting.md


Additional tools
------------------
.. toctree::
   :maxdepth: 1
   :glob:

   main/user_guide/tools/*


About
------------------
.. toctree::
   :maxdepth: 1
   :glob:

   main/about/*

Developers
------------------
.. toctree::
   :maxdepth: 1
   :glob:

   main/dev/*

Misc
------------------
.. toctree::
   :maxdepth: 1
   :glob:

   main/user_guide/misc/*