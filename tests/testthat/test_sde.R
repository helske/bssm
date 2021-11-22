context("Test SDE")
#' @srrstats {G5.0, G5.1, G5.4, G5.4a, G5.4b, G5.4c, BS7.2} GBM model and data 
#' as in Vihola, Helske, Franks (2020)
test_that("MCMC for SDE works", {
  skip_on_cran()
  
  pntrs <- cpp_example_model("sde_gbm")
  set.seed(42)
  n <- 50
  dt <- 1
  mu <- 0.05
  sigma_x <- 0.3
  sigma_y <- 1
  x <- 1
  y <- numeric(n)
  for (k in 1:n) {
    x <- x * exp((mu - 0.5 * sigma_x^2) * dt + 
        sqrt(dt) * rnorm(1, sd = sigma_x))
    y[k] <- rnorm(1, log(x), sigma_y)
  }

  model <- ssm_sde(y, pntrs$drift, pntrs$diffusion, 
    pntrs$ddiffusion, pntrs$obs_density,
    pntrs$prior, c(mu = 0.08, sigma_x = 0.4, sigma_y = 1.5), 
    x0 = 1, positive = TRUE)
  
  expect_error(out <- run_mcmc(model, iter = 2e4, burnin = 5000,
    particles = 50, mcmc_type = "is2", 
    L_c = 2, L_f = 6, threads = 2), NA)
  
  paper <- c(0.053, 0.253, 1.058, 1.254, 2.960)
  expect_equivalent(weighted_mean(out$theta, out$weights * out$counts), 
    paper[1:3], tol = 0.1)
  expect_equivalent(weighted_mean(t(out$alpha[c(1,50),1,]), 
    out$weights * out$counts), paper[4:5], tol = 0.1)
  
  expect_error(out <- run_mcmc(model, iter = 2e4, burnin = 5000,
    particles = 50, mcmc_type = "is2", 
    L_c = 2, L_f = 6, threads = -1))
  
  expect_error(out <- run_mcmc(model, iter = 2e4, burnin = 5000,
    particles = 50, mcmc_type = "is2", 
    L_c = 2, L_f = -1))
  
  expect_error(out <- run_mcmc(model, iter = 2e4, burnin = 5000,
    particles = 50, mcmc_type = "is2", 
    L_c = 2, L_f = 1))
  
  expect_error(out <- run_mcmc(model, iter = 2e4, burnin = 5000,
    particles = 50, mcmc_type = "pm", L_c = 0))
  
  expect_error(bootstrap_filter(model, 1000, L = -2))
  expect_error(particle_smoother(model, 1000, L = 0))
  expect_error(ll <- logLik(model, 10000, L = -3))
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
