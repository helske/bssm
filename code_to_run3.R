library("bssm")


set.seed(42)
slope <- cumsum(c(0.01, rnorm(499, sd = 0.001)))
y <- rpois(500, exp(cumsum(slope + c(1,rnorm(499, sd = 0.1)))))
model <- ng_bsm(y, sd_level = 0.1, sd_slope = 0.001, distribution = "poisson")

is_da_comparison <- function(nsim = 1000, n_iter = 1e4, n_burnin = 0, nsim_states = 10) {

  results <- array(NA, c(10, 10, nsim), dimnames =
      list(c("approximate", "MCMC", "DA-MCMC", "DA-MCMC2",
        "IS-MCMC1 (1 thread)", "IS-MCMC1 (16 threads)",
        "IS-MCMC2 (1 thread)", "IS-MCMC2 (16 threads)",
        "IS-MCMC3 (1 thread)", "IS-MCMC3 (16 threads)"),
        c("time (seconds)", "acceptance rate",
          "E(mu_500)", "E(nu_500)", "Var(E(mu_500)))", "Var(E(nu_500))",
          "E(theta_1)", "E(theta_2)", "Var(E(theta_1))", "Var(E(theta_2))"),
        NULL))

  for (i in 1:nsim) {

    s <- system.time(res <- run_mcmc(model, n_iter = n_iter, n_burnin = n_burnin,
      nsim_states = 1, method = "standard"))[3]
    results[1, , i] <-  c(s, res$acceptance_rate,
      rowMeans(res$alpha[500,,]),
      summary(coda::mcmc(t(res$alpha[500,,])))$stat[, 4]^2,
      colMeans(res$theta),
      summary(res$theta)$stat[, 4]^2)

    s <- system.time(res <- run_mcmc(model, n_iter = n_iter, n_burnin = n_burnin,
      nsim_states = nsim_states, method = "standard"))[3]
    results[2, , i] <-   c(s, res$acceptance_rate,
      rowMeans(res$alpha[500,,]),
      summary(coda::mcmc(t(res$alpha[500,,])))$stat[, 4]^2,
      colMeans(res$theta),
      summary(res$theta)$stat[, 4]^2)


    s <- system.time(res <- run_mcmc(model, n_iter = n_iter, n_burnin = n_burnin,
      nsim_states = nsim_states, method = "delayed acceptance"))[3]
    results[3, , i] <-   c(s, res$acceptance_rate,
      rowMeans(res$alpha[500,,]),
      summary(coda::mcmc(t(res$alpha[500,,])))$stat[, 4]^2,
      colMeans(res$theta),
      summary(res$theta)$stat[, 4]^2)

    s <- system.time(res <- run_mcmc(model, n_iter = n_iter, n_burnin = n_burnin,
      nsim_states = nsim_states, target = res$acceptance_rate / 0.234, method = "delayed acceptance"))[3]
    results[4, , i] <-   c(s, res$acceptance_rate,
      rowMeans(res$alpha[500,,]),
      summary(coda::mcmc(t(res$alpha[500,,])))$stat[, 4]^2,
      colMeans(res$theta),
      summary(res$theta)$stat[, 4]^2)

    s <- system.time(res <- run_mcmc(model, n_iter = n_iter, n_burnin = n_burnin,
      nsim_states = nsim_states, method = "IS correction", n_threads = 1))[3]
    w <- res$weights / sum(res$weights)
    results[5, , i] <-   c(s, res$acceptance_rate,
      sum(w * res$alpha[500,1,]), sum(w * res$alpha[500,2,]),
      sum(w^2 * (res$alpha[500,1,] - sum(w * res$alpha[500,1,]))^2),
      sum(w^2 * (res$alpha[500,2,] - sum(w * res$alpha[500,2,]))^2),
      sum(w * res$theta[,1]), sum(w * res$theta[,2]),
      sum(w^2 * (res$theta[, 1] - sum(w * res$theta[, 1]))^2),
      sum(w^2 * (res$theta[, 2] - sum(w * res$theta[, 2]))^2))

    s <- system.time(res <- run_mcmc(model, n_iter = n_iter, n_burnin = n_burnin,
      nsim_states = nsim_states, method = "IS correction", n_threads = 16))[3]
    w <- res$weights / sum(res$weights)
    results[6, , i] <-   c(s, res$acceptance_rate,
      sum(w * res$alpha[500,1,]), sum(w * res$alpha[500,2,]),
      sum(w^2 * (res$alpha[500,1,] - sum(w * res$alpha[500,1,]))^2),
      sum(w^2 * (res$alpha[500,2,] - sum(w * res$alpha[500,2,]))^2),
      sum(w * res$theta[,1]), sum(w * res$theta[,2]),
      sum(w^2 * (res$theta[, 1] - sum(w * res$theta[, 1]))^2),
      sum(w^2 * (res$theta[, 2] - sum(w * res$theta[, 2]))^2))

    s <- system.time(res <- run_mcmc(model, n_iter = n_iter, n_burnin = n_burnin,
      nsim_states = nsim_states, method = "block IS correction", n_threads = 1))[3]
    w <- res$counts * res$weights / sum(res$counts * res$weights)
    results[7, , i] <-   c(s, res$acceptance_rate,
      sum(w * res$alpha[500,1,]), sum(w * res$alpha[500,2,]),
      sum(w^2 * (res$alpha[500,1,] - sum(w * res$alpha[500,1,]))^2),
      sum(w^2 * (res$alpha[500,2,] - sum(w * res$alpha[500,2,]))^2),
      sum(w * res$theta[,1]), sum(w * res$theta[,2]),
      sum(w^2 * (res$theta[, 1] - sum(w * res$theta[, 1]))^2),
      sum(w^2 * (res$theta[, 2] - sum(w * res$theta[, 2]))^2))

    s <- system.time(res <- run_mcmc(model, n_iter = n_iter, n_burnin = n_burnin,
      nsim_states = nsim_states, method = "block IS correction", n_threads = 16))[3]
    w <- res$counts * res$weights / sum(res$counts * res$weights)
    results[8, , i] <-   c(s, res$acceptance_rate,
      sum(w * res$alpha[500,1,]), sum(w * res$alpha[500,2,]),
      sum(w^2 * (res$alpha[500,1,] - sum(w * res$alpha[500,1,]))^2),
      sum(w^2 * (res$alpha[500,2,] - sum(w * res$alpha[500,2,]))^2),
      sum(w * res$theta[,1]), sum(w * res$theta[,2]),
      sum(w^2 * (res$theta[, 1] - sum(w * res$theta[, 1]))^2),
      sum(w^2 * (res$theta[, 2] - sum(w * res$theta[, 2]))^2))

    s <- system.time(res <- run_mcmc(model, n_iter = n_iter, n_burnin = n_burnin,
      nsim_states = nsim_states, method = "IS2", n_threads = 1))[3]
    w <- res$counts * res$weights / sum(res$counts * res$weights)
    results[9, , i] <-   c(s, res$acceptance_rate,
      sum(w * res$alpha[500,1,]), sum(w * res$alpha[500,2,]),
      sum(w^2 * (res$alpha[500,1,] - sum(w * res$alpha[500,1,]))^2),
      sum(w^2 * (res$alpha[500,2,] - sum(w * res$alpha[500,2,]))^2),
      sum(w * res$theta[,1]), sum(w * res$theta[,2]),
      sum(w^2 * (res$theta[, 1] - sum(w * res$theta[, 1]))^2),
      sum(w^2 * (res$theta[, 2] - sum(w * res$theta[, 2]))^2))

    s <- system.time(res <- run_mcmc(model, n_iter = n_iter, n_burnin = n_burnin,
      nsim_states = nsim_states, method = "IS2", n_threads = 16))[3]
    w <- res$counts * res$weights / sum(res$counts * res$weights)
    results[10, , i] <-   c(s, res$acceptance_rate,
      sum(w * res$alpha[500,1,]), sum(w * res$alpha[500,2,]),
      sum(w^2 * (res$alpha[500,1,] - sum(w * res$alpha[500,1,]))^2),
      sum(w^2 * (res$alpha[500,2,] - sum(w * res$alpha[500,2,]))^2),
      sum(w * res$theta[,1]), sum(w * res$theta[,2]),
      sum(w^2 * (res$theta[, 1] - sum(w * res$theta[, 1]))^2),
      sum(w^2 * (res$theta[, 2] - sum(w * res$theta[, 2]))^2))

    print(i)
  }

  results

}
