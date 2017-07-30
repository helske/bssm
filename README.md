[![Build Status](https://travis-ci.org/helske/bssm.png?branch=master)](https://travis-ci.org/helske/bssm)
[![cran version](http://www.r-pkg.org/badges/version/bssm)](http://cran.r-project.org/package=bssm)
[![downloads](http://cranlogs.r-pkg.org/badges/bssm)](http://cranlogs.r-pkg.org/badges/bssm)
[![codecov.io](http://codecov.io/github/helske/bssm/coverage.svg?branch=master)](http://codecov.io/github/helske/bssm?branch=master)
[![DOI](https://zenodo.org/badge/53692028.svg)](https://zenodo.org/badge/latestdoi/53692028)



bssm: an R package for Bayesian inference of state space models
==========================================================================

Efficient methods for Bayesian inference of state space models where the observation density is Gaussian, Poisson, binomial or negative binomial, and where the state dynamics are Gaussian.

For details, see [package vignette](https://github.com/helske/bssm/blob/master/bssm.pdf) and paper [Importance sampling type correction of Markov chain Monte Carlo and exact approximations](http://arxiv.org/abs/1609.02541).

Current Status
==========================================================================
Now on CRAN. Still under development, pull requests very welcome especially related to post-processing and visualization. 
You can install the latest development version by using the devtools package:

```R
install.packages("devtools")
devtools::install_github("helske/bssm")
```
