 [![R-CMD-check](https://github.com/helske/bssm/workflows/R-CMD-check/badge.svg)](https://github.com/helske/bssm/actions)
[![cran version](http://www.r-pkg.org/badges/version/bssm)](http://cran.r-project.org/package=bssm)
[![downloads](http://cranlogs.r-pkg.org/badges/bssm)](http://cranlogs.r-pkg.org/badges/bssm)



bssm: an R package for Bayesian inference of state space models
==========================================================================

Efficient methods for Bayesian inference of state space models via particle Markov 
chain Monte Carlo and importance sampling type weighted Markov chain Monte Carlo. 
Currently Gaussian, Poisson, binomial, negative binomial, and Gamma observation densities 
and linear-Gaussian state dynamics, as well as general non-linear Gaussian models are supported.

For details, see [package vignettess at CRAN](https://cran.r-project.org/web/packages/bssm/index.html) and paper [Importance sampling type estimators based on approximate marginal Markov chain Monte Carlo](https://onlinelibrary.wiley.com/doi/abs/10.1111/sjos.12492). There are also couple posters related to IS-correction methodology: [SMC 2017 workshop: Accelerating MCMC with an approximation ](http://users.jyu.fi/~jovetale/posters/SMC2017) and [UseR!2017: Bayesian non-Gaussian state space models in R](http://users.jyu.fi/~jovetale/posters/user2017.pdf).


You can install the latest development version by using the devtools package:

```R
install.packages("devtools")
devtools::install_github("helske/bssm")
```

Recent changes (For all changes, see NEWS file.)
==========================================================================

bssm 1.0.2 (Release date: )
==============

   * Added function `suggest_N` which can be used to choose 
     suitable number of particles for IS-MCMC.
   * Added function `post_correct` which can be used to update 
     previous approximate MCMC with IS-weights.
   * Gamma priors are now supported in easy-to-use models such as `bsm_lg`. 
   * The adaptation of the proposal distribution now continues also after the burn-in by default. 
   * Changed default MCMC type to typically most efficient and robust IS2.
   * Renamed `nsim` argument to `particles` in most of the R functions (`nsim` also works with a warning).
   * Fixed a bug with bsm models with covariates, where all standard deviation parameters were fixed. 
     This resulted error within MCMC algorithms.
   * Fixed a dimension drop bug in the predict method which caused error for univariate models.
   * Fixed few typos in vignette (thanks Kyle Hussman) and added more examples.
   
bssm 1.0.1-1 (Release date: 2020-11-12)
==============

  * Added an argument `future` for predict method which allows 
    predictions for current time points by supplying the original model 
    (e.g., for posterior predictive checks). 
    At the same time the argument name `future_model` was changed to `model`.
  * Fixed a bug in summary.mcmc_run which resulted error when 
    trying to obtain summary for states only.
  * Added a check for Kalman filter for a degenerate case where all 
    observational level and state level variances are zero.
  * Renamed argument `n_threads` to `threads` for consistency 
    with `iter` and `burnin` arguments.
  * Improved documentation, added examples.
  * Added a vignette regarding psi-APF for non-linear models.
  
bssm 1.0.0 (Release date: 2020-06-09)
==============
Major update

  * Major changes for model definitions, now model updating and priors 
    can be defined via R functions (non-linear and SDE models still rely on C++ snippets).
  * Added support for multivariate non-Gaussian models.
  * Added support for gamma distributions.
  * Added the function as.data.frame for mcmc output which converts the MCMC samples 
    to data.frame format for easier post-processing.
  * Added truncated normal prior.
  * Many argument names and model building functions have been changed for clarity and consistency.
  * Major overhaul of C++ internals which can bring minor efficiency gains and smaller installation size.
  * Allow zero as initial value for positive-constrained parameters of bsm models.
  * Small changes to summary method which can now return also only summaries of the states.
  * Fixed a bug in initializing run_mcmc for negative binomial model. 
  * Fixed a bug in phi-APF for non-linear models.
  * Reimplemented predict method which now always produces data frame of samples.
  
bssm 0.1.11 (Release date: 2020-02-25)
==============
  * Switched (back) to approximate posterior in RAM for PM-SPDK and PM-PSI, 
    as it seems to work better with noisy likelihood estimates.
  * Print and summary methods for MCMC output are now coherent in their output.
  
bssm 0.1.10 (Release date: 2020-02-04)
==============
  * Fixed missing weight update for IS-SPDK without OPENMP flag.
  * Removed unused usage argument ... from expand_sample.
  
bssm 0.1.9 (Release date: 2020-01-27)
==============
  * Fixed state sampling for PM-MCMC with SPDK.
  * Added ts attribute for svm model.
  * Corrected asymptotic variance for summary methods.
  
For older versions, see NEWS file.
