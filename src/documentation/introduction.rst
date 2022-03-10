.. _introduction:


************
Introduction
************

This project replicates part of the main results from `DellaVigna and Pope, 2018 <https://doi.org/10.1093/restud/rdx033>`_,
specifically focusing on structural estimates using non linear least squares (NLS). The
starting point from which we build upon our codes is the replication of the mentioned
paper by `Nunnari and Pozzi, 2021 <https://github.com/MassimilianoPozzi/python_julia_structural_behavioral_economics>`_
from Bocconi University. Our goal is to improve on that by putting emphasis on programming best-practices and applying concepts learned
in the course "Effective Programming Practices for Economists", such as Pytask, Pytest,
`Estimagic (Gabler, 2021) <https://github.com/OpenSourceEconomics/estimagic>`_, Sphinx, functional programming and docstrings.

This project has been build based on `a template by von Gaudecker, 2019 <https://econ-project-templates.readthedocs.io/en/stable/index.html>`_.

.. _getting_started:

Getting started
===============

For a local machine to run this project, it needs to have a Python and LaTex distribution.
The project was tested on Windows 10 operating system.

The project environment includes all the dependencies needed to run the project.

To run this project in a local machine:
 - after cloning the repo, open a terminal in the root directory of the project
and create and activate the environment typing `$ conda env create -f environment.yml` and `$ conda activate structural_behavioral_economics_dellavigna_pope_2018_using_estimagic` commands, respectively.
 - For imports to work, the following command should be typed to the terminal in the root directory of the terminal: `$ conda develop .`
 - To generate the output files that will be stored in `bld` folder, type `$ pytask` in the root directory of your terminal.
