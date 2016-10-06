## ----setup, include=FALSE------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)

## ----UKgas---------------------------------------------------------------
library("bssm")
set.seed(123)

init_sd <- 0.1 * sd(log10(UKgas))
prior <- halfnormal(init_sd, 1)
model <- bsm(log10(UKgas), sd_y = prior, sd_level = prior,
  sd_slope = prior, sd_seasonal = prior)
mcmc_out <- run_mcmc(model, n_iter = 6e4)
mcmc_out

plot(mcmc_out$theta)

# posterior mode estimates
mcmc_out$theta[which.max(mcmc_out$posterior), ]

# posterior covariance matrix:
cov(mcmc_out$theta)

# compare to shape of the proposal distribution:
cor(mcmc_out$theta)
cov2cor(mcmc_out$S %*% t(mcmc_out$S))

## ----trend, dev.args=list(pointsize = 10), fig.cap="Smoothed trend component."----
ts.plot(model$y, rowMeans(mcmc_out$alpha[, "level", ]), col = 1:2)

## ----predict, dev.args=list(pointsize = 10), fig.cap="Mean predictions and prediction intervals."----
pred <- predict(model, n_iter = 1e4, n_ahead = 40,
  probs = c(0.025, 0.1, 0.9, 0.975), S = mcmc_out$S)
ts.plot(log10(UKgas), pred$mean, pred$intervals[,-3],
  col = c(1, 2, c(3, 4, 4, 3)), lty = c(1, 1, rep(2, 4)))

## ----predict2, dev.args=list(pointsize = 10), fig.cap="Prediction plots with ggplot2."----
require("ggplot2")
autoplot(pred, interval_color = "red", alpha_fill = 0.2)

