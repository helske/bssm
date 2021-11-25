context("Test predictions")


test_that("Gaussian predictions work", {
  
  set.seed(1)
  y <- rnorm(10, cumsum(rnorm(10, 0, 0.1)), 0.1)
  model <- ar1_lg(y, 
    rho = uniform(0.9, 0, 1), mu = 0, 
    sigma = halfnormal(0.1, 1),
    sd_y = halfnormal(0.1, 1))
  
  set.seed(123)
  mcmc_results <- run_mcmc(model, iter = 1000)
  future_model <- model
  future_model$y <- rep(NA, 3)
  set.seed(1)
  pred <- predict(mcmc_results, future_model, type = "mean", 
    nsim = 100)
  
  expect_gt(mean(pred$value[pred$time == 3]), -0.5)
  expect_lt(mean(pred$value[pred$time == 3]), 0.5)
  
  # Posterior predictions for past observations:
  set.seed(1)
  expect_error(yrep <- predict(mcmc_results, model, type = "response", 
    future = FALSE, nsim = 100), NA)
  set.seed(1)
  expect_error(meanrep <- predict(mcmc_results, model, type = "mean", 
    future = FALSE, nsim = 100), NA)
  
  expect_equal(mean(yrep$value - meanrep$value), 0, tol = 0.1)
  
  ufun <- function(x) {
    T <- array(x[1])
    R <- array(exp(x[2]))
    H <- array(exp(x[3]))
    dim(T) <- dim(R) <- dim(H) <- c(1, 1, 1)
    P1 <- matrix(exp(x[2])^2) / (1 - x[1]^2)
    list(T = T, R = R, P1 = P1, H = H)
  }
  pfun <- function(x) {
    ifelse(x[1] > 1 | x[1] < 0, -Inf, sum(-0.5 * exp(x[2:3])^2 + x[2:3]))
  }
  
  expect_error(model2 <- ssm_mlg(matrix(model$y, length(model$y), 1), 
    Z = 1, H = model$H, T = model$T, R = model$R,
    a1 = model$a1, P1 = model$P1, 
    init_theta = c(rho = 0.9, sigma = log(0.1), sd_y = log(0.1)),
    update_fn = ufun, prior_fn = pfun, state_names = "signal"), NA)
  
  set.seed(123)
  expect_error(mcmc_results2 <- run_mcmc(model2, iter = 1000), 
    NA)
  # transform manually
  mcmc_results2$theta[, 2:3] <- exp(mcmc_results2$theta[, 2:3])
  expect_equal(mcmc_results$theta, mcmc_results2$theta)
  expect_equal(mcmc_results$alpha, mcmc_results2$alpha)
  expect_equal(mcmc_results$posterior, mcmc_results2$posterior)
  # transform back to predict...
  mcmc_results2$theta[, 2:3] <- log(mcmc_results2$theta[, 2:3])
  future_model2 <- model2
  future_model2$y <- matrix(NA, 3, 1)
  set.seed(1)
  expect_error(pred2 <- predict(mcmc_results2, future_model2, type = "mean", 
    nsim = 100), NA)
  expect_equal(pred, pred2)
  # Posterior predictions for past observations:
  set.seed(1)
  expect_error(yrep2 <- predict(mcmc_results2, model2, type = "response", 
    future = FALSE, nsim = 100), NA)
  set.seed(1)
  expect_error(meanrep2 <- predict(mcmc_results2, model2, type = "mean", 
    future = FALSE, nsim = 100), NA)
  expect_equal(yrep, yrep2)
  expect_equal(meanrep, meanrep2)
  
  expect_error(predict(mcmc_results2, model, type = "response", 
    future = FALSE, nsim = 100))

  expect_error(predict(mcmc_results2, model2, type = "response", 
    future = FALSE, nsim = 0))
  expect_error(predict(mcmc_results2, model2, type = "response", 
    future = 5, nsim = 100)) 
  expect_error(predict(mcmc_results2, model = 465, type = "response", 
      future = FALSE, nsim = 100))
  mcmc_results3 <- run_mcmc(model2, iter = 1000, output_type = "theta")
  expect_error(predict(mcmc_results3, model2, type = "response", 
    future = FALSE, nsim = 100))
  class(model) <- "aa"
  expect_error(predict(mcmc_results3, model2, type = "response", 
    future = FALSE, nsim = 100))
  
  
  set.seed(1)
  y <- rnorm(10, cumsum(rnorm(10, 0, 0.1)), 0.1)
  model <- bsm_lg(y, 
    sd_level = halfnormal(1, 1),
    sd_slope = halfnormal(0.1, 0.1),
    sd_y = halfnormal(0.1, 1))
  
  mcmc_results <- run_mcmc(model, iter = 1000)
  future_model <- model
  future_model$y <- rep(NA, 3)
  
  expect_error(predict(mcmc_results, future_model, type = "mean", 
    nsim = 1000), paste0("The number of samples should be smaller than or ",
    "equal to the number of posterior samples 500."))
  expect_error(predict(mcmc_results, future_model, type = "state", 
    nsim = 50), NA)
  
  set.seed(1)
  expect_error(pred <- predict(mcmc_results, future_model, type = "mean", 
    nsim = 500), NA)
  
  expect_gt(mean(pred$value[pred$time == 3]), 0)
  expect_lt(mean(pred$value[pred$time == 3]), 0.5)
  
  # Posterior predictions for past observations:
  set.seed(1)
  expect_error(yrep <- predict(mcmc_results, model, type = "response", 
    future = FALSE, nsim = 100), NA)
  set.seed(1)
  expect_error(meanrep <- predict(mcmc_results, model, type = "mean", 
    future = FALSE, nsim = 100), NA)
  
  expect_equal(mean(yrep$value - meanrep$value), 0, tol = 0.1)

  
})

test_that("Non-gaussian predictions work", {
  
  set.seed(1)
  y <- rpois(10, exp(cumsum(rnorm(10, 0, 0.1))))
  model <- ar1_ng(y, 
    rho = uniform(0.9, 0, 1), mu = 0, 
    sigma = halfnormal(0.1, 1), distribution = "poisson")
  
  set.seed(123)
  expect_error(mcmc_results <- run_mcmc(model, iter = 1000, particles = 5), NA)
  future_model <- model
  future_model$y <- rep(NA, 3)
  set.seed(1)
  expect_error(pred <- predict(mcmc_results, future_model, type = "mean", 
    nsim = 100), NA)
  
  expect_gt(mean(pred$value[pred$time == 3]), 1)
  expect_lt(mean(pred$value[pred$time == 3]), 1.5)
  
  # Posterior predictions for past observations:
  set.seed(1)
  expect_error(yrep <- predict(mcmc_results, model, type = "response", 
    future = FALSE, nsim = 100), NA)
  set.seed(1)
  expect_error(meanrep <- predict(mcmc_results, model, type = "mean", 
    future = FALSE, nsim = 100), NA)
  
  expect_equal(mean(yrep$value - meanrep$value), 0, tol = 0.5)
  
  update_fn <- function(x) {
    T <- array(x[1])
    R <- array(exp(x[2]))
    dim(T) <- dim(R) <- c(1, 1, 1)
    P1 <- matrix(exp(x[2])^2) / (1 - x[1]^2)
    list(T = T, R = R, P1 = P1)
  }
  prior_fn <- function(x) {
    ifelse(x[1] < 0 | x[1] > 1, -Inf, - 0.5 * exp(x[2])^2 + x[2])
  }
  model2 <- ssm_ung(model$y, Z = 1, T = model$T, R = model$R, a1 = model$a1,
    P1 = model$P1, distribution = "poisson", update_fn = update_fn, 
    prior_fn = prior_fn, init_theta = c(rho = 0.9, log(model$theta[2])), 
    state_names = "signal")
  
  set.seed(123)
  expect_error(mcmc_results2 <- run_mcmc(model2, iter = 1000, particles = 5), 
    NA)
  # transform manually
  mcmc_results2$theta[, 2] <- exp(mcmc_results2$theta[, 2])
  expect_equal(mcmc_results$theta, mcmc_results2$theta)
  expect_equal(mcmc_results$alpha, mcmc_results2$alpha)
  expect_equal(mcmc_results$posterior, mcmc_results2$posterior)
  
  # transform back for predict
  mcmc_results2$theta[, 2] <- log(mcmc_results2$theta[, 2])
  
  future_model2 <- model2
  future_model2$y <- rep(NA, 3)
  set.seed(1)
  expect_error(pred2 <- predict(mcmc_results2, future_model2, type = "mean", 
    nsim = 100), NA)
  expect_equal(pred, pred2)
  # Posterior predictions for past observations:
  set.seed(1)
  expect_error(yrep2 <- predict(mcmc_results2, model2, type = "response", 
    future = FALSE, nsim = 100), NA)
  set.seed(1)
  expect_error(meanrep2 <- predict(mcmc_results2, model2, type = "mean", 
    future = FALSE, nsim = 100), NA)
  expect_equal(yrep, yrep2)
  expect_equal(meanrep, meanrep2)
  expect_error(predict(mcmc_results2, model, type = "response", 
    future = FALSE, nsim = 100))
})

test_that("Predictions for nlg_ssm work", {
  skip_on_cran()
  set.seed(1)
  n <- 10
  x <- y <- numeric(n)
  y[1] <- rnorm(1, exp(x[1]), 0.1)
  for(i in 1:(n-1)) {
    x[i+1] <- rnorm(1, 0.9 * x[i], 0.1)
    y[i+1] <- rnorm(1, exp(x[i+1]), 0.1)
  }
  
  pntrs <- cpp_example_model("nlg_ar_exp")
  
  expect_error(model <- ssm_nlg(y = y, a1 = pntrs$a1, P1 = pntrs$P1, 
    Z = pntrs$Z_fn, H = pntrs$H_fn, T = pntrs$T_fn, R = pntrs$R_fn, 
    Z_gn = pntrs$Z_gn, T_gn = pntrs$T_gn,
    theta = c(mu = 0, rho = 0.9, log_R = log(0.1), log_H = log(0.1)), 
    log_prior_pdf = pntrs$log_prior_pdf,
    n_states = 1, n_etas = 1, state_names = "state"), NA)
  
  expect_error(mcmc_results <- run_mcmc(model, iter = 5000, particles = 10), 
    NA)
  future_model <- model
  future_model$y <- rep(NA, 3)
  expect_error(pred <- predict(mcmc_results, particles = 10, 
    future_model, type = "mean", nsim = 1000), NA)
  
  expect_gt(mean(pred$value[pred$time == 3]), 0.5)
  expect_lt(mean(pred$value[pred$time == 3]), 1.5)
  
  # Posterior predictions for past observations:
  set.seed(1)
  expect_error(yrep <- predict(mcmc_results, model, type = "response", 
    future = FALSE, nsim = 100), NA)
  set.seed(1)
  expect_error(meanrep <- predict(mcmc_results, model, type = "mean", 
    future = FALSE, nsim = 100), NA)
  
  expect_equal(mean(yrep$value - meanrep$value), 0, tol = 0.5)
})


test_that("Predictions for mng_ssm work", {
  set.seed(1)
  n <- 20
  x <- cumsum(rnorm(n, sd = 0.5))
  phi <- 2
  y <- cbind(rnbinom(n, size = phi, mu = exp(x)),
    rpois(n, exp(x)))
  
  Z <- matrix(1, 2, 1)
  T <- 1
  R <- 0.5
  a1 <- 0
  P1 <- 1
  
  update_fn <- function(theta) {
    list(R = array(theta[1], c(1, 1, 1)), phi = c(theta[2], 1))
  }
  
  prior_fn <- function(theta) {
    ifelse(all(theta > 0), sum(dnorm(theta, 0, 1, log = TRUE)), -Inf)
  }
  
  expect_error(model <- ssm_mng(y, Z, T, R, a1, P1, phi = c(2, 1), 
    init_theta = c(0.5, 2), 
    distribution = c("negative binomial", "poisson"),
    update_fn = update_fn, prior_fn = prior_fn), NA)
  
  
  expect_error(mcmc_results <- run_mcmc(model, iter = 5000, particles = 10), 
    NA)
  future_model <- model
  future_model$y <- matrix(NA, 3, 2)
  expect_error(pred <- predict(mcmc_results, particles = 10, 
    future_model, type = "mean", nsim = 1000), NA)
  
  expect_gte(min(pred$value), 0)
  expect_lt(max(pred$value), 1000)
  
  # Posterior predictions for past observations:
  set.seed(1)
  expect_error(yrep <- predict(mcmc_results, model, type = "response", 
    future = FALSE, nsim = 100), NA)
  set.seed(1)
  expect_error(meanrep <- predict(mcmc_results, model, type = "mean", 
    future = FALSE, nsim = 100), NA)
  
  expect_equal(mean(yrep$value - meanrep$value), 0, tol = 0.5)
})

