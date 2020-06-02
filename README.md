[![Build Status](https://travis-ci.org/helske/bssm.png?branch=master)](https://travis-ci.org/helske/bssm)
[![cran version](http://www.r-pkg.org/badges/version/bssm)](http://cran.r-project.org/package=bssm)
[![downloads](http://cranlogs.r-pkg.org/badges/bssm)](http://cranlogs.r-pkg.org/badges/bssm)
[![DOI](https://zenodo.org/badge/53692028.svg)](https://zenodo.org/badge/latestdoi/53692028)



bssm: an R package for Bayesian inference of state space models
==========================================================================

Efficient methods for Bayesian inference of state space models via particle Markov 
chain Monte Carlo and importance sampling type weighted Markov chain Monte Carlo. 
Currently Gaussian, Poisson, binomial, or negative binomial observation densities 
and linear-Gaussian state dynamics, as well as general non-linear Gaussian models are supported.

For details, see [package vignette](https://github.com/helske/bssm/blob/master/bssm.pdf) and paper [Importance sampling type correction of Markov chain Monte Carlo and exact approximations](http://arxiv.org/abs/1609.02541). There is also a separate [vignette for non-linear Gaussian models](https://github.com/helske/bssm/blob/master/growth_model.pdf), and couple posters related to IS-correction methodology:



Current Status
==========================================================================
Now on CRAN. Still under development, pull requests very welcome especially related to post-processing, visualization, and C++ modularization.

You can install the latest development version by using the devtools package:

```R
install.packages("devtools")
devtools::install_github("helske/bssm")
```
