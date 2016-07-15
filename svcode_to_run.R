library("bssm")

y <- scan("sv.dat")
model <- svm(y, ar = 0.9731, sd_ar = 0.1726, sigma = 0.6338)

is_da_comparison <- function(nsim = 100, n_iter = 1e4, n_burnin = 0, nsim_states = 10) {

  results <- array(NA, c(10, 10, nsim), dimnames =
      list(c("approximate", "MCMC", "DA-MCMC", "DA-MCMC2",
        "IS-MCMC1 (1 thread)", "IS-MCMC1 (16 threads)",
        "IS-MCMC2 (1 thread)", "IS-MCMC2 (16 threads)",
        "IS-MCMC3 (1 thread)", "IS-MCMC3 (16 threads)"),
        c("time (seconds)", "acceptance rate",
          "E(alpha_945)", "Var(E(alpha_945)))",
          "E(phi)", "E(sd_phi)","E(sigma)",
          "Var(E(phi))", "Var(E(sd_phi))", "Var(E(sigma))"),
        NULL))

  for (i in 1:nsim) {

    s <- system.time(res <- run_mcmc(model, n_iter = n_iter, n_burnin = n_burnin,
      nsim_states = 1, method = "standard"))[3]
    results[1, , i] <-  c(s, res$acceptance_rate,
      mean(res$alpha[945,1,]),
      summary(coda::mcmc(res$alpha[945,1,]))$stat[4]^2,
      colMeans(res$theta),
      summary(res$theta)$stat[, 4]^2)

    s <- system.time(res <- run_mcmc(model, n_iter = n_iter, n_burnin = n_burnin,
      nsim_states = nsim_states, method = "standard"))[3]
    results[2, , i] <-   c(s, res$acceptance_rate,
      mean(res$alpha[945,1,]),
      summary(coda::mcmc(res$alpha[945,1,]))$stat[4]^2,
      colMeans(res$theta),
      summary(res$theta)$stat[, 4]^2)


    s <- system.time(res <- run_mcmc(model, n_iter = n_iter, n_burnin = n_burnin,
      nsim_states = nsim_states, method = "delayed acceptance"))[3]
    results[3, , i] <-   c(s, res$acceptance_rate,
      mean(res$alpha[945,1,]),
      summary(coda::mcmc(res$alpha[945,1,]))$stat[4]^2,
      colMeans(res$theta),
      summary(res$theta)$stat[, 4]^2)

    s <- system.time(res <- run_mcmc(model, n_iter = n_iter, n_burnin = n_burnin,
      nsim_states = nsim_states, target = 0.234^2 / res$acceptance_rate, method = "delayed acceptance"))[3]
    results[4, , i] <-   c(s, res$acceptance_rate,
      mean(res$alpha[945,1,]),
      summary(coda::mcmc(res$alpha[945,1,]))$stat[4]^2,
      colMeans(res$theta),
      summary(res$theta)$stat[, 4]^2)

    s <- system.time(res <- run_mcmc(model, n_iter = n_iter, n_burnin = n_burnin,
      nsim_states = nsim_states, method = "IS correction", n_threads = 1))[3]
    w <- res$weights / sum(res$weights)
    results[5, , i] <-   c(s, res$acceptance_rate,
      sum(w * res$alpha[945,1,]),
      sum(w^2 * (res$alpha[945,1,] - sum(w * res$alpha[945,1,]))^2),
      sum(w * res$theta[,1]), sum(w * res$theta[,2]), sum(w * res$theta[,3]),
      sum(w^2 * (res$theta[, 1] - sum(w * res$theta[, 1]))^2),
      sum(w^2 * (res$theta[, 2] - sum(w * res$theta[, 2]))^2),
      sum(w^2 * (res$theta[, 3] - sum(w * res$theta[, 3]))^2))

    s <- system.time(res <- run_mcmc(model, n_iter = n_iter, n_burnin = n_burnin,
      nsim_states = nsim_states, method = "IS correction", n_threads = 16))[3]
    w <- res$weights / sum(res$weights)
    results[6, , i] <-   c(s, res$acceptance_rate,
      sum(w * res$alpha[945,1,]),
      sum(w^2 * (res$alpha[945,1,] - sum(w * res$alpha[945,1,]))^2),
      sum(w * res$theta[,1]), sum(w * res$theta[,2]), sum(w * res$theta[,3]),
      sum(w^2 * (res$theta[, 1] - sum(w * res$theta[, 1]))^2),
      sum(w^2 * (res$theta[, 2] - sum(w * res$theta[, 2]))^2),
      sum(w^2 * (res$theta[, 3] - sum(w * res$theta[, 3]))^2))

    s <- system.time(res <- run_mcmc(model, n_iter = n_iter, n_burnin = n_burnin,
      nsim_states = nsim_states, method = "block IS correction", n_threads = 1))[3]
    w <- res$counts * res$weights / sum(res$counts * res$weights)
    results[7, , i] <-   c(s, res$acceptance_rate,
      sum(w * res$alpha[945,1,]),
      sum(w^2 * (res$alpha[945,1,] - sum(w * res$alpha[945,1,]))^2),
      sum(w * res$theta[,1]), sum(w * res$theta[,2]), sum(w * res$theta[,3]),
      sum(w^2 * (res$theta[, 1] - sum(w * res$theta[, 1]))^2),
      sum(w^2 * (res$theta[, 2] - sum(w * res$theta[, 2]))^2),
      sum(w^2 * (res$theta[, 3] - sum(w * res$theta[, 3]))^2))

    s <- system.time(res <- run_mcmc(model, n_iter = n_iter, n_burnin = n_burnin,
      nsim_states = nsim_states, method = "block IS correction", n_threads = 16))[3]
    w <- res$counts * res$weights / sum(res$counts * res$weights)
    results[8, , i] <-   c(s, res$acceptance_rate,
      sum(w * res$alpha[945,1,]),
      sum(w^2 * (res$alpha[945,1,] - sum(w * res$alpha[945,1,]))^2),
      sum(w * res$theta[,1]), sum(w * res$theta[,2]), sum(w * res$theta[,3]),
      sum(w^2 * (res$theta[, 1] - sum(w * res$theta[, 1]))^2),
      sum(w^2 * (res$theta[, 2] - sum(w * res$theta[, 2]))^2),
      sum(w^2 * (res$theta[, 3] - sum(w * res$theta[, 3]))^2))

    s <- system.time(res <- run_mcmc(model, n_iter = n_iter, n_burnin = n_burnin,
      nsim_states = nsim_states, method = "IS2", n_threads = 1))[3]
    w <- res$counts * res$weights / sum(res$counts * res$weights)
    results[9, , i] <-   c(s, res$acceptance_rate,
      sum(w * res$alpha[945,1,]),
      sum(w^2 * (res$alpha[945,1,] - sum(w * res$alpha[945,1,]))^2),
      sum(w * res$theta[,1]), sum(w * res$theta[,2]), sum(w * res$theta[,3]),
      sum(w^2 * (res$theta[, 1] - sum(w * res$theta[, 1]))^2),
      sum(w^2 * (res$theta[, 2] - sum(w * res$theta[, 2]))^2),
      sum(w^2 * (res$theta[, 3] - sum(w * res$theta[, 3]))^2))

    s <- system.time(res <- run_mcmc(model, n_iter = n_iter, n_burnin = n_burnin,
      nsim_states = nsim_states, method = "IS2", n_threads = 16))[3]
    w <- res$counts * res$weights / sum(res$counts * res$weights)
    results[10, , i] <-   c(s, res$acceptance_rate,
      sum(w * res$alpha[945,1,]),
      sum(w^2 * (res$alpha[945,1,] - sum(w * res$alpha[945,1,]))^2),
      sum(w * res$theta[,1]), sum(w * res$theta[,2]), sum(w * res$theta[,3]),
      sum(w^2 * (res$theta[, 1] - sum(w * res$theta[, 1]))^2),
      sum(w^2 * (res$theta[, 2] - sum(w * res$theta[, 2]))^2),
      sum(w^2 * (res$theta[, 3] - sum(w * res$theta[, 3]))^2))

    print(i)
  }

  results

}
