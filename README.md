[![Build Status](https://travis-ci.org/helske/bssm.png?branch=master)](https://travis-ci.org/helske/bssm)
[![codecov.io](http://codecov.io/github/helske/bssm/coverage.svg?branch=master)](http://codecov.io/github/helske/bssm?branch=master)

bssm: an R package for Bayesian inference of exponential family and stochastic volatility state space models
==========================================================================

Efficient methods for Bayesian inference of state space models where the observation density is Gaussian, Poisson, binomial or negative binomial, and where the state dynamics are Gaussian.

For details, see [package vignette](https://github.com/helske/bssm/blob/master/bssm.pdf) and paper [Importance sampling type correction of Markov chain Monte Carlo and exact approximations](http://arxiv.org/abs/1609.02541).

Current Status
==========================================================================
Under development. You can install the latest development version by using the devtools package:

```R
install.packages("devtools")
devtools::install_github("helske/bssm")
```
