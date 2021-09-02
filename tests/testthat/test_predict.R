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
