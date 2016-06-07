[![Build Status](https://travis-ci.org/helske/bssm.png?branch=master)](https://travis-ci.org/helske/bssm)

bssm: an R Package for Bayesian Inference of Exponential Family State Space Models
==========================================================================

The R package `bssm` is designed for Bayesian inference of exponential family state space models of form

$$
\begin{aligned}
p(y_t | Z_t \alpha_t, x'_t\beta) \qquad (\textrm{observation equation})\\
\alpha_{t+1} = T_t \alpha_t + R_t \eta_t, \qquad (\textrm{transition equation})
\end{aligned}
$$

where $\eta_t \sim N(0, I_k)$ and $\alpha_1 \sim N(a_1, P_1)$ independently of each other, $x_t$ contains the exogenous covariate values at time time, with $\beta$ corresponding to the regression coefficients. Currently, following observational level distributions are supported:

* Gaussian distribution: $p(y_t | Z_t \alpha_t, x_t\beta) = x'_t \beta + Z_t \alpha_t + H_t \epsilon_t$ with $\epsilon_t \sim N(0, 1)$.

* Poisson distribution: $p(y_t | Z_t \alpha_t, x_t \beta) = \textrm{Poisson}(\phi_t \textrm{exp}(x'_t \beta + Z_t \alpha_t))$, where $\phi_t$ is the exposure at time $t$.

* Binomial distribution: $p(y_t | Z_t \alpha_t, x_t \beta) = \textrm{binomial}(\phi_t, \pi_t)$, where $\phi_t$ is the size and $\pi_t = \textrm{exp}(x_t \beta + Z_t \alpha_t) / (1 + \textrm{exp}(x'_t \beta + Z_t \alpha_t))$ is the probability of the success.

* Negative binomial and Gamma distributions are added in near future.


Current Status
==========================================================================
Under heavy active development.

You can install the latest development version from Github using devtools package:

```R
install.packages("devtools")
library(devtools)
install_github("helske/bssm")
```
Here is a short example:

```R
library("bssm")
set.seed(123)

init_sd <- 0.1 * sd(log10(UKgas))
model <- bsm(log10(UKgas), sd_y = init_sd, sd_level = init_sd,
  sd_slope = init_sd, sd_seasonal = init_sd)
mcmc_out <- run_mcmc(model, n_iter = 5000)
mcmc_out$acceptance_rate
summary(mcmc_out$theta)
# posterior mode estimates:
mcmc_out$theta[which.max(mcmc_out$logLik), ]
# posterior covariance matrix:
cov(mcmc_out$theta)
# compare to shape of the proposal distribution:
cor(mcmc_out$theta)
cov2cor(mcmc_out$S %*% t(mcmc_out$S))

# smoothed trend 
ts.plot(model$y, rowMeans(mcmc_out$alpha[, "level", ]), col = 1:2)

# predictions
pred <- predict(model, n_iter = 10000, n_ahead = 40, 
  probs = c(0.025, 0.1, 0.9, 0.975), S = mcmc_out$S)
ts.plot(log10(UKgas), pred$mean, pred$intervals, 
  col = c(1, 2, c(3, 4, 4, 3)), lty = c(1, 1, rep(2, 4)))
# with ggplot2
require("ggplot2")
autoplot(pred, interval_colour = "red", alpha_fill = 0.2)
```
