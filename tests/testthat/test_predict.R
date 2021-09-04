context("Test predictions")


test_that("Gaussian predictions work", {
  
  set.seed(1)
  y <- rnorm(10, cumsum(rnorm(10, 0, 0.1)), 0.1)
  model <- ar1_lg(y, 
    rho = uniform(0.9, 0, 0.99), mu = 0, 
    sigma = halfnormal(0.1, 1),
    sd_y = halfnormal(0.1, 1))
  
  mcmc_results <- run_mcmc(model, iter = 1000)
  future_model <- model
  future_model$y <- rep(NA, 3)
  pred <- predict(mcmc_results, future_model, type = "mean", 
    nsim = 100)
  
  expect_gt(mean(pred$value[pred$time == 3]), -0.5)
  expect_lt(mean(pred$value[pred$time == 3]), 0.5)
  
  # Posterior predictions for past observations:
  yrep <- predict(mcmc_results, model, type = "response", 
    future = FALSE, nsim = 100)
  meanrep <- predict(mcmc_results, model, type = "mean", 
    future = FALSE, nsim = 100)
  
  expect_equal(mean(yrep$value-meanrep$value), 0, tol = 0.1)

})


test_that("Predictions for nlg_ssm work", {
  skip_on_cran()
  set.seed(1)
  n <- 10
  x <- y <- numeric(n)
  y[1] <- rnorm(1, exp(x[1]), 0.1)
  for(i in 1:(n-1)) {
    x[i+1] <- rnorm(1, sin(x[i]), 0.1)
    y[i+1] <- rnorm(1, exp(x[i+1]), 0.1)
  }
  
  pntrs <- nlg_example_models("sin_exp")
  
  expect_error(model <- ssm_nlg(y = y, a1 = pntrs$a1, P1 = pntrs$P1, 
    Z = pntrs$Z_fn, H = pntrs$H_fn, T = pntrs$T_fn, R = pntrs$R_fn, 
    Z_gn = pntrs$Z_gn, T_gn = pntrs$T_gn,
    theta = c(log_H = log(0.1), log_R = log(0.1)), 
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
  yrep <- predict(mcmc_results, model, type = "response", 
    future = FALSE, nsim = 100)
  meanrep <- predict(mcmc_results, model, type = "mean", 
    future = FALSE, nsim = 100)
  
  expect_equal(mean(yrep$value-meanrep$value), 0, tol = 0.1)
})
