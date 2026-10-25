# POPSRegression.jl

[![docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://POPS-UQ.github.io/POPSRegression.jl)

A library of probabilistic surrogate models targeting the low-noise, misspecified regime, written in Julia.
The name comes from the POPS regression algorithm from [Perez & Swinburne (2025)](https://doi.org/10.1088/2632-2153/ad9fce).
                                                                                                                                                                                        
## Core features   
  - Univariate and multivariate POPS hypercube regression for linear models                                                                                                                                           
  - Leverage-based filtering of training points for efficient fitting                                                                              
  - Predictive uncertainty quantification: min–max bounds, standard deviations, and differential entropy estimates                                                
  - Compliance with the StatsAPI.jl interface                                                                                                                                            
  - Linear equality and inequality constraints on the model coefficients (constrained POPS, solved with [OSQP](https://osqp.org))
  - Unit testing, including checks of the constrained fit against closed-form solutions
  - Simple examples, including uncertainty quantification for a linear MLIP predictions, and structural quantities from subsequent molecular dynamics simulations.