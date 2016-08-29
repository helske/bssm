## Testing different sizes of importance samples m with delayed acceptance
## Measure the efficiency using the asymptotic standard error estimate from coda package
## compare also with the non-asymptotic SE

## Simulatation smoother in bssm uses one antithetic variable,
## so we get almost two samples in price of one with increased accuracy.
## Therefore set m as multiples of 2

library("bssm")
library("coda")

nsim <- 100
n_iter <- 5000
m <- seq(10, 50, by = 10)

model <- ng_bsm(Seatbelts[, "VanKilled"], distribution = "poisson",
  sd_level = 0.01, sd_seasonal = 0.01, slope = FALSE,
  xreg = Seatbelts[, "law"])
## Better initial values for theta and S so we dont need burn-in (this is it)
out<- run_mcmc(model, n_iter = n_iter, nsim_states = 10, type="param", seed = 1, end_adaptive_phase = FALSE)
out$acceptance_rate

model$beta[] <- mean(out$theta[, 3])
diag(model$R[,,1]) <- colMeans(out$theta[, 1:2])
S <- out$S

## Without the adaptive approximation
theta_se_naa <- theta_mean_naa <- array(NA, c(3, length(m), nsim))
alpha_se_naa <- alpha_mean_naa <- array(NA, c(length(m), nsim))
mu_se_naa <-mu_mean_naa <- array(NA, c(length(m), nsim))

for (j in 1:nsim) {
  for (k in seq_along(m)) {
    out <- run_mcmc(model, n_iter = n_iter, nsim_states = m[k], n_burnin = 0, S = S,
      adaptive_approx = FALSE)

    res <- summary(mcmc(out$theta))$stat
    theta_se_naa[, k, j] <- res[, 4]
    theta_mean_naa[, k, j] <- res[, 1]

    res <- summary(mcmc(out$alpha[192,1,]))$stat
    alpha_se_naa[k, j] <- res[4]
    alpha_mean_naa[k, j] <- res[1]

    res <- summary(mcmc(exp(out$theta[,3] + colSums(out$alpha[192, 1:2, ]))))$stat
    mu_se_naa[k, j] <- res[4]
    mu_mean_naa[k, j] <- res[1]
  }
  print(j)
  print(apply(mu_mean_naa,1,sd,  na.rm = TRUE))
}



## summary Without the adaptive approximation
summary_alpha_naa <- summary_alphavar_naa <- array(NA, c(length(m), nsim))
summary_mu_naa <- summary_muvar_naa <- array(NA, c(length(m), nsim))

for (j in 1:nsim) {
  for (k in seq_along(m)) {
    out <- run_mcmc(model, n_iter = n_iter, nsim_states = m[k], n_burnin = 0, S = S,
      adaptive_approx = FALSE, type = "summary")

    summary_alpha_naa[k, j] <- out$alphahat[192, 1]
    summary_alphavar_naa[k, j] <- out$Vt[1, 1, 192]

    summary_mu_naa[k, j] <- out$muhat[192]
    summary_muvar_naa[k, j] <- out$Vmu[192]

  }
  print(j)
  print(apply(summary_mu_naa,1,sd, na.rm = TRUE))
}

## with AA

theta_se <- theta_mean <- array(NA, c(3, length(m), nsim))
alpha_se <- alpha_mean <- array(NA, c(length(m), nsim))
mu_se <-mu_mean <- array(NA, c(length(m), nsim))

for (j in 1:nsim) {
  for (k in seq_along(m)) {
    out <- run_mcmc(model, n_iter = n_iter, nsim_states = m[k], n_burnin = 0, S = S,
      adaptive_approx = TRUE)

    res <- summary(mcmc(out$theta))$stat
    theta_se[, k, j] <- res[, 4]
    theta_mean[, k, j] <- res[, 1]

    res <- summary(mcmc(out$alpha[192,1,]))$stat
    alpha_se[k, j] <- res[4]
    alpha_mean[k, j] <- res[1]

    res <- summary(mcmc(exp(out$theta[,3] + colSums(out$alpha[192, 1:2, ]))))$stat
    mu_se[k, j] <- res[4]
    mu_mean[k, j] <- res[1]
  }
  print(j)
  print(apply(mu_mean,1, sd,  na.rm = TRUE))
}



## summary Without the adaptive approximation
summary_alpha <- summary_alphavar <- array(NA, c(length(m), nsim))
summary_mu <- summary_muvar <- array(NA, c(length(m), nsim))

for (j in 1:nsim) {
  for (k in seq_along(m)) {
    out <- run_mcmc(model, n_iter = n_iter, nsim_states = m[k], n_burnin = 0, S = S,
      adaptive_approx = TRUE, type = "summary")

    summary_alpha[k, j] <- out$alphahat[192, 1]
    summary_alphavar[k, j] <- out$Vt[1, 1, 192]

    summary_mu[k, j] <- out$muhat[192]
    summary_muvar[k, j] <- out$Vmu[192]

  }
  print(j)
  print(apply(summary_mu,1,sd, na.rm = TRUE))
}
