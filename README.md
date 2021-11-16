
<!-- README.md is generated from README.Rmd. Please edit that file -->

# bssm

<!-- badges: start -->

[![Project Status: Active - The project has reached a stable, usable
state and is being actively
developed](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![R-CMD-check](https://github.com/helske/bssm/workflows/R-CMD-check/badge.svg)](https://github.com/helske/bssm/actions)
[![Codecov test
coverage](https://codecov.io/gh/helske/bssm/branch/master/graph/badge.svg)](https://codecov.io/gh/helske/bssm?branch=master)
[![cran
version](http://www.r-pkg.org/badges/version/bssm)](http://cran.r-project.org/package=bssm)
[![downloads](http://cranlogs.r-pkg.org/badges/bssm)](http://cranlogs.r-pkg.org/badges/bssm)

<!-- badges: end -->

The `bssm` R package provides efficient methods for Bayesian inference
of state space models via particle Markov chain Monte Carlo and
importance sampling type weighted MCMC. Currently Gaussian, Poisson,
binomial, negative binomial, and Gamma observation densities with
linear-Gaussian state dynamics, as well as general non-linear Gaussian
models and discretely observed latent diffusion processes are supported.

For details, see

-   [The bssm paper on ArXiv](https://arxiv.org/abs/2101.08492) (to
    appear in R Journal),
-   [Package vignettes at
    CRAN](https://cran.r-project.org/web/packages/bssm/index.html)
-   Paper on [Importance sampling type estimators based on approximate
    marginal Markov chain Monte
    Carlo](https://onlinelibrary.wiley.com/doi/abs/10.1111/sjos.12492)

There are also couple posters and a talk related to IS-correction
methodology and bssm package:

-   [UseR!2021 talk
    slides](https://jounihelske.netlify.app/talk/user2021/)  
-   [SMC 2017 workshop: Accelerating MCMC with an
    approximation](http://users.jyu.fi/~jovetale/posters/SMC2017)
-   [UseR!2017: Bayesian non-Gaussian state space models in
    R](http://users.jyu.fi/~jovetale/posters/user2017.pdf)

The `bssm` package was originally developed with the support of Academy
of Finland grants 284513, 312605, and 311877. Current development is
focused on increased usability and stability.

## Installation

You can install the released version of bssm from
[CRAN](https://CRAN.R-project.org) with:

``` r
install.packages("bssm")
```

And the development version from [GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("helske/bssm")
```

Or from R-universe with

``` r
install.packages("bssm", repos = "https://helske.r-universe.dev")
```

## Example

Consider the daily air quality measurements in New Your from May to
September 1973, available in the `datasets` package. Let’s try to
predict the missing ozone levels by simple linear-Gaussian local linear
trend model with temperature and wind as explanatory variables:

``` r
library("bssm")
#> 
#> Attaching package: 'bssm'
#> The following object is masked from 'package:base':
#> 
#>     gamma
library("dplyr")
#> 
#> Attaching package: 'dplyr'
#> The following objects are masked from 'package:stats':
#> 
#>     filter, lag
#> The following objects are masked from 'package:base':
#> 
#>     intersect, setdiff, setequal, union
library("ggplot2")

data("airquality", package = "datasets")

# Covariates as matrix. For complex cases, check out as_bssm function
xreg <- airquality %>% select(Wind, Temp) %>% as.matrix()

model <- bsm_lg(airquality$Ozone,
  xreg = xreg,  
  beta = normal(rep(0, ncol(xreg)), 0, 1),
  sd_y = gamma_prior(1, 2, 0.01),
  sd_level = gamma_prior(1, 2, 0.01), 
  sd_slope = gamma_prior(1, 2, 0.01))
  
fit <- run_mcmc(model, iter = 20000, burnin = 5000)
fit
#> 
#> Call:
#> run_mcmc.gaussian(model = model, iter = 20000, burnin = 5000)
#> 
#> Iterations = 5001:20000
#> Thinning interval = 1
#> Length of the final jump chain = 3601
#> 
#> Acceptance rate after the burn-in period:  0.24
#> 
#> Summary for theta:
#> 
#>                Mean        SD          SE
#> sd_y     20.9745103 1.9548950 0.065711076
#> sd_level  6.1826612 2.6898276 0.110026275
#> sd_slope  0.3227774 0.2496854 0.008370595
#> Wind     -2.5224090 0.5692881 0.017491284
#> Temp      1.0289642 0.1991063 0.006160738
#> 
#> Effective sample sizes for theta:
#> 
#>                ESS
#> sd_y      885.0538
#> sd_level  597.6626
#> sd_slope  889.7614
#> Wind     1059.3047
#> Temp     1044.4906
#> 
#> Summary for alpha_154:
#> 
#>              Mean        SD         SE
#> level -28.3237394 19.921774 0.55159040
#> slope  -0.3782085  1.723463 0.04120739
#> 
#> Effective sample sizes for alpha_154:
#> 
#>            ESS
#> level 1304.436
#> slope 1749.257
#> 
#> Run time:
#>    user  system elapsed 
#>    0.92    0.00    0.90
obs <- data.frame(Time = 1:nrow(airquality),
        Ozone = airquality$Ozone) %>% filter(!is.na(Ozone))

pred <- fitted(fit, model)
pred %>%
  ggplot(aes(x = Time, y = Mean)) + 
  geom_ribbon(aes(ymin = `2.5%`, ymax = `97.5%`), 
    alpha = 0.5, fill = "steelblue") + 
  geom_line() + 
  geom_point(data = obs, 
    aes(x = Time, y = Ozone), colour = "Tomato") +
  theme_bw()
```

<img src="man/figures/README-example-1.png" width="100%" />

Same model but now assuming observations are from Gamma distribution:

``` r
model2 <- bsm_ng(airquality$Ozone,
    xreg = xreg,  
    beta = normal(rep(0, ncol(xreg)), 0, 1),
    distribution = "gamma",
    phi = gamma_prior(1, 2, 0.01),
    sd_level = gamma_prior(1, 2, 0.1), 
    sd_slope = gamma_prior(1, 2, 0.1))

fit2 <- run_mcmc(model2, iter = 20000, burnin = 5000, particles = 10)
fit2
#> 
#> Call:
#> run_mcmc.nongaussian(model = model2, iter = 20000, particles = 10, 
#>     burnin = 5000)
#> 
#> Iterations = 5001:20000
#> Thinning interval = 1
#> Length of the final jump chain = 3902
#> 
#> Acceptance rate after the burn-in period:  0.26
#> 
#> Summary for theta:
#> 
#>                  Mean          SD           SE        SE-IS
#> sd_level  0.060119156 0.036304969 0.0021676154 7.340294e-04
#> sd_slope  0.004282931 0.003742675 0.0001661556 7.508864e-05
#> phi       3.974702619 0.514745897 0.0174002858 1.019891e-02
#> Wind     -0.057474480 0.015071259 0.0004125317 2.982147e-04
#> Temp      0.052108192 0.008977271 0.0002718235 1.784298e-04
#> 
#> Effective sample sizes for theta:
#> 
#>                ESS
#> sd_level  280.5225
#> sd_slope  507.3815
#> phi       875.1309
#> Wind     1334.7023
#> Temp     1090.7239
#> 
#> Summary for alpha_154:
#> 
#>               Mean         SD           SE        SE-IS
#> level -0.154108984 0.74092804 0.0206049048 0.0148100409
#> slope -0.003410796 0.02343375 0.0005549973 0.0004921424
#> 
#> Effective sample sizes for alpha_154:
#> 
#>            ESS
#> level 1293.037
#> slope 1782.797
#> 
#> Run time:
#>    user  system elapsed 
#>   10.74    0.07   10.75
```

Comparison:

``` r
pred2 <- fitted(fit2, model2)

bind_rows(list(Gaussian = pred, Gamma = pred2), .id = "Model") %>%
  ggplot(aes(x = Time, y = Mean)) + 
  geom_ribbon(aes(ymin = `2.5%`, ymax = `97.5%`, fill = Model), 
    alpha = 0.25) + 
  geom_line(aes(colour = Model)) + 
  geom_point(data = obs, 
    aes(x = Time, y = Ozone)) +
  theme_bw()
```

<img src="man/figures/README-compare-1.png" width="100%" />

## Recent changes (For all changes, see NEWS file.)

#### bssm 1.1.6 (Release date: 2021-09-06)

-   Cleaned codes and added more comprehensive tests in line with
    pkgcheck tests. This resulted in finding and fixing multiple bugs:
-   Fixed a bug in EKF-based particle filter which returned filtered
    estimates also in place of one-step ahead predictions.
-   Fixed a bug which caused an error in suggest\_N for nlg\_ssm.
-   Fixed a bug which caused incorrect sampling of smoothing
    distribution for ar1\_lg model when predicting past or when using
    simulation smoother.
-   Fixed a bug which caused an error when predicting past values in
    multivariate time series case.
-   Fixed sampling of negative binomial distribution in predict method,
    which used std::negative\_binomial which converts non-integer phi to
    integer. Sampling now uses Gamma-Poisson mixture for simulation.

#### bssm 1.1.4 (Release date: 2021-04-13)

-   Better documentation for SV model, and changed ordering of arguments
    to emphasise the recommended parameterization.
-   Fixed predict method for SV model.

#### bssm 1.1.3-2 (Release date: 2021-02-24)

-   Fixed missing parenthesis causing compilation fail in case of no
    OpenMP support.
-   Added pandoc version &gt;= 1.12.3 to system requirements.

#### bssm 1.1.3-1 (Release date: 2021-02-22)

-   Fixed PM-MCMC and DA-MCMC for SDE models and added an example to
    `ssm_sde`.
-   Added vignette for SDE models.
-   Updated citation information and streamlined the main vignette.

#### bssm 1.1.2 (Release date: 2021-02-08)

-   Some bug fixes, see NEWS for details.

#### bssm 1.1.0 (Release date: 2021-01-19)

-   Added function `suggest_N` which can be used to choose suitable
    number of particles for IS-MCMC.
-   Added function `post_correct` which can be used to update previous
    approximate MCMC with IS-weights.
-   Gamma priors are now supported in easy-to-use models such as
    `bsm_lg`.
-   The adaptation of the proposal distribution now continues also after
    the burn-in by default.
-   Changed default MCMC type to typically most efficient and robust
    IS2.
-   Renamed `nsim` argument to `particles` in most of the R functions
    (`nsim` also works with a warning).
-   Fixed a bug with bsm models with covariates, where all standard
    deviation parameters were fixed. This resulted error within MCMC
    algorithms.
-   Fixed a dimension drop bug in the predict method which caused error
    for univariate models.
-   Fixed few typos in vignette (thanks Kyle Hussman) and added more
    examples.

#### bssm 1.0.1-1 (Release date: 2020-11-12)

-   Added an argument `future` for predict method which allows
    predictions for current time points by supplying the original model
    (e.g., for posterior predictive checks). At the same time the
    argument name `future_model` was changed to `model`.
-   Fixed a bug in summary.mcmc\_run which resulted error when trying to
    obtain summary for states only.
-   Added a check for Kalman filter for a degenerate case where all
    observational level and state level variances are zero.
-   Renamed argument `n_threads` to `threads` for consistency with
    `iter` and `burnin` arguments.
-   Improved documentation, added examples.
-   Added a vignette regarding psi-APF for non-linear models.

#### bssm 1.0.0 (Release date: 2020-06-09)

Major update

-   Major changes for model definitions, now model updating and priors
    can be defined via R functions (non-linear and SDE models still rely
    on C++ snippets).
-   Added support for multivariate non-Gaussian models.
-   Added support for gamma distributions.
-   Added the function as.data.frame for mcmc output which converts the
    MCMC samples to data.frame format for easier post-processing.
-   Added truncated normal prior.
-   Many argument names and model building functions have been changed
    for clarity and consistency.
-   Major overhaul of C++ internals which can bring minor efficiency
    gains and smaller installation size.
-   Allow zero as initial value for positive-constrained parameters of
    bsm models.
-   Small changes to summary method which can now return also only
    summaries of the states.
-   Fixed a bug in initializing run\_mcmc for negative binomial model.
-   Fixed a bug in phi-APF for non-linear models.
-   Reimplemented predict method which now always produces data frame of
    samples.

#### bssm 0.1.11 (Release date: 2020-02-25)

-   Switched (back) to approximate posterior in RAM for PM-SPDK and
    PM-PSI, as it seems to work better with noisy likelihood estimates.
-   Print and summary methods for MCMC output are now coherent in their
    output.

#### bssm 0.1.10 (Release date: 2020-02-04)

-   Fixed missing weight update for IS-SPDK without OPENMP flag.
-   Removed unused usage argument … from expand\_sample.

#### bssm 0.1.9 (Release date: 2020-01-27)

-   Fixed state sampling for PM-MCMC with SPDK.
-   Added ts attribute for svm model.
-   Corrected asymptotic variance for summary methods.

For older versions, see NEWS file.
