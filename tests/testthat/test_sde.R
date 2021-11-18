context("Test SDE")
test_that("MCMC for SDE works", {
  skip_on_cran()
  
  pntrs <- cpp_example_model("sde_gbm")
  set.seed(1)
  n <- 50
  dt <- 1
  mu <- 0.05
  sigma_x <- 0.3
  sigma_y <- 1
  x <- numeric(n)
  x[1] <- 1
  for (k in 2:n) {
    x[k] <- x[k-1] * exp((mu - 0.5 * sigma_x^2) * dt + 
        sqrt(dt) * rnorm(1, sd = sigma_x))
  }
  y <- rnorm(n, log(x), sigma_y)
  
  model <- ssm_sde(y, pntrs$drift, pntrs$diffusion, 
    pntrs$ddiffusion, pntrs$obs_density,
    pntrs$prior, c(mu = 0.08, log_sigma_x = 0.4, sigma_y = 1.5), 
    x0 = 1, positive = TRUE)
  
  expect_error(out <- run_mcmc(model, iter = 2e4, 
    particles = 50, mcmc_type = "is2", 
    L_c = 4, L_f = 6, threads = 2), NA)
  
  expect_error(bootstrap_filter(model, 1000, L = -2))
  expect_error(ll <- logLik(model, 10000, L = 3), NA)
  expect_equal(ll, -17, tol = 1)
  expect_error(out_bsf <- bootstrap_filter(model, 1000, L = 3), NA)
  expect_equal(ll, out_bsf$logLik, tol = 1)
  
  expect_error(out <- run_mcmc(model, iter = 500, 
    particles = 10, mcmc_type = "pm", L_f = 2), NA)
  expect_gt(out$acceptance_rate, 0)
  
  expect_error(out <- run_mcmc(model, iter = 500, 
    particles = 10, mcmc_type = "da", L_c = 2, L_f = 3), NA)
  expect_gt(out$acceptance_rate, 0)
  
  expect_error(out2 <- run_mcmc(model, iter = 500, 
    particles = 10, mcmc_type = "is2", L_c = 1, L_f = 2), NA)
  
  expect_gt(out2$acceptance_rate, 0)
  expect_equal(mean(colMeans(out$theta)-colMeans(out2$theta)), 0, tol = 1)
  
  expect_error(out2 <- run_mcmc(model, iter = 500, 
    particles = 10, mcmc_type = "is1", L_c = 1, L_f = 2, threads = 2), NA)
  
  expect_gt(out2$acceptance_rate, 0)
  expect_equal(mean(colMeans(out$theta)-colMeans(out2$theta)), 0, tol = 1)
})
