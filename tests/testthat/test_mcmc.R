context("Test MCMC")

tol <- 1e-8
test_that("MCMC results for Gaussian model are correct", {
  set.seed(123)
  model_bssm <- bsm_lg(rnorm(10, 3), P1 = diag(2, 2), sd_slope = 0,
    sd_y = uniform(1, 0, 10), 
    sd_level = uniform(1, 0, 10))
  
  expect_error(mcmc_bsm <- run_mcmc(model_bssm, iter = 50, seed = 1), NA)
  
  expect_equal(run_mcmc(model_bssm, iter = 100, seed = 1)[-14], 
    run_mcmc(model_bssm, iter = 100, seed = 1)[-14])
  expect_equal(run_mcmc(model_bssm, iter = 100, seed = 1, 
    output_type = "summary")[-15], 
    run_mcmc(model_bssm, iter = 100, seed = 1, output_type = "summary")[-15])
  expect_equal(run_mcmc(model_bssm, iter = 100, seed = 1, 
    output_type = "theta")[-13], 
    run_mcmc(model_bssm, iter = 100, seed = 1, output_type = "theta")[-13])
  expect_equal(run_mcmc(model_bssm, iter = 100, seed = 1, 
    output_type = "theta")$theta, 
    run_mcmc(model_bssm, iter = 100, seed = 1, output_type = "summary")$theta)
  expect_equal(run_mcmc(model_bssm, iter = 100, seed = 1, 
    output_type = "theta")$acceptance_rate, 
    run_mcmc(model_bssm, iter = 100, seed = 1, 
      output_type = "summary")$acceptance_rate)
  
  expect_gt(mcmc_bsm$acceptance_rate, 0)
  expect_gte(min(mcmc_bsm$theta), 0)
  expect_lt(max(mcmc_bsm$theta), Inf)
  expect_true(is.finite(sum(mcmc_bsm$alpha)))
  
})


test_that("DA-MCMC results for Poisson model are correct", {
  set.seed(123)
  model_bssm <- bsm_ng(rpois(10, exp(0.2) * (2:11)), P1 = diag(2, 2), 
    sd_slope = 0, sd_level = uniform(2, 0, 10), u = 2:11, 
    distribution = "poisson")
  
  expect_error(mcmc_poisson <- run_mcmc(model_bssm, mcmc_type = "da", 
    iter = 100, particles = 5, seed = 42), NA)
  
  expect_warning(summary(mcmc_poisson, only_theta = TRUE))
  
  sumr <- expect_error(summary(mcmc_poisson, variable = "both"), NA)
  
  expect_lt(sum(abs(sumr$theta - c(0.25892090511681, 0.186796779799571))), 0.5)
  
  
  states <- expand_sample(mcmc_poisson, variable = "states")
  
  expect_equal(as.numeric(sumr$states$Mean[,1]), 
    as.numeric(colMeans(states$level)))
  
  expect_equal(run_mcmc(model_bssm, mcmc_type = "da", iter = 100, seed = 1, 
    particles = 5)[-14], 
    run_mcmc(model_bssm, mcmc_type = "da", iter = 100, seed = 1, 
      particles = 5)[-14])
  expect_equal(run_mcmc(model_bssm, mcmc_type = "da", iter = 100, seed = 1, 
    output_type = "summary", particles = 5)[-15], 
    run_mcmc(model_bssm, mcmc_type = "da", iter = 100, seed = 1, 
      output_type = "summary", particles = 5)[-15])
  expect_equal(run_mcmc(model_bssm, mcmc_type = "da", iter = 100, seed = 1, 
    output_type = "theta", particles = 5)[-13], 
    run_mcmc(model_bssm, mcmc_type = "da", iter = 100, seed = 1, 
      output_type = "theta", particles = 5)[-13])
  
  expect_gt(mcmc_poisson$acceptance_rate, 0)
  expect_gte(min(mcmc_poisson$theta), 0)
  expect_lt(max(mcmc_poisson$theta), Inf)
  expect_true(is.finite(sum(mcmc_poisson$alpha)))
  
})


test_that("MCMC results for SV model using IS-correction are correct", {
  set.seed(123)
  expect_error(model_bssm <- svm(rnorm(10), rho = uniform(0.95, -0.999, 0.999), 
    sd_ar = halfnormal(1, 5), sigma = halfnormal(1, 2)), NA)
  
  expect_equal(run_mcmc(model_bssm, iter = 100, particles = 10,
    mcmc_type = "is1", seed = 1)[-16], 
    run_mcmc(model_bssm, iter = 100, particles = 10, mcmc_type = "is1", 
      seed = 1)[-16])
  
  expect_equal(run_mcmc(model_bssm, iter = 100, particles = 10,
    mcmc_type = "is2", seed = 1)[-16], 
    run_mcmc(model_bssm, iter = 100, particles = 10, mcmc_type = "is2", 
      seed = 1)[-16])
  
  expect_equal(run_mcmc(model_bssm, iter = 100, particles = 10,
    mcmc_type = "is3", seed = 1)[-16], 
    run_mcmc(model_bssm, iter = 100, particles = 10, mcmc_type = "is3", 
      seed = 1)[-16])
  
  expect_equal(run_mcmc(model_bssm, iter = 100, particles = 10,
    mcmc_type = "is2", seed = 1, sampling_mcmc_type = "psi")[-16], 
    run_mcmc(model_bssm, iter = 100, particles = 10, 
      mcmc_type = "is2", seed = 1, sampling_mcmc_type = "psi")[-16])
  
  expect_equal(run_mcmc(model_bssm, iter = 100, particles = 10,
    mcmc_type = "is2", seed = 1, sampling_mcmc_type = "bsf")[-16], 
    run_mcmc(model_bssm, iter = 100, particles = 10, 
      mcmc_type = "is2", seed = 1, sampling_mcmc_type = "bsf")[-16])
  
  expect_equal(run_mcmc(model_bssm, iter = 100, particles = 10,
    mcmc_type = "is2", seed = 1, sampling_mcmc_type = "psi", 
    threads = 2L)[-16], 
    run_mcmc(model_bssm, iter = 100, particles = 10, 
      mcmc_type = "is2", seed = 1, sampling_mcmc_type = "psi", 
      threads = 2L)[-16])
  
  expect_error(mcmc_sv <- run_mcmc(model_bssm, iter = 100, particles = 10,
    mcmc_type = "is2", seed = 1, sampling_mcmc_type = "bsf"), NA)
  
  expect_warning(expand_sample(mcmc_sv))
  
  expect_gt(mcmc_sv$acceptance_rate, 0)
  expect_true(is.finite(sum(mcmc_sv$theta)))
  expect_true(is.finite(sum(mcmc_sv$alpha)))
  expect_gte(min(mcmc_sv$weights), 0)
  expect_lt(max(mcmc_sv$weights), Inf)
})
