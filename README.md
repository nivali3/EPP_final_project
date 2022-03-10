# Structural Behavioral Economics (Della Vigna, Pope 2018) using Estimagic by Edoardo Falchi and Nigar Valiyeva
## EPP for Economists Final Project, University of Bonn WS 21/22

This project replicates part of the main results from [DellaVigna and Pope, 2018](https://github.com/MassimilianoPozzi/python_julia_structural_behavioral_economics),
specifically focusing on structural estimates using non linear least squares (NLS). The
starting point from which we build upon our codes is the replication of the mentioned
paper by Nunnari and Pozzi, 2021 from Bocconi University. Our goal is to improve on
that by putting emphasis on programming best-practices and applying concepts learned
in the course "Effective Programming Practices for Economists", such as Pytask, Pytest,
Estimagic (Gabler, 2021), Sphinx, functional programming and docstrings.

### Environment

For a local machine to run this project, it needs to have a Python and LaTex distribution.
The project was tested on Windows 10 operating system with python. To run this project in a local machine

`conda ..`

jhjkh


### Structure of the project

This project has been build based on [a template by von Gaudecker, 2019](https://econ-project-templates.readthedocs.io/en/stable/index.html)


jkh


# IDEA STORAGE

IDEA 1: Add an extra cost function and do the analysis again ...

IDEA 2: In the NLS estimation instead of using 100 point bins use continuous point score using maximum likelihood.

IDEA 3 (MAIN): Using estimagic/Replace scipy with estimagic. In other words, do the optimization using estimagic instead of scipy!

JUPYTER NOTEBOOK - table_5_6_NLS, Cell 6: Maybe the difference between replicated numbers and original numbers are because in the "power cost function" case it is LOG of effort ...

QUESTION: In table_5_6_NLS, how does the initial values for k, g, and s guessed? 

First Task: `table_5_6_nls.ipynb`- TRY to re-do the code using estimagic.

## REMEMBER: Fix the PATH in the files at the end!



QUESTION: Would it matter if we store all the generated outpus in .csv files or should we store them in .yaml and only final ones in .csv?

In table_5_6_NLS, how does the initial values for k, g, and s guessed? 

## REMEMBER: Fix the key "variances" to "standard errors"!

