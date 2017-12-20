context("Test MCMC")

tol <- 1e-8
test_that("MCMC results for Gaussian model are correct",{
  set.seed(123)
  model_bssm <- bsm(rnorm(10,3), P1 = diag(2,2), sd_slope = 0,
    sd_y = uniform(1, 0, 10), 
    sd_level = uniform(1, 0, 10))
  
  expect_equal(run_mcmc(model_bssm, n_iter = 10, seed = 1)[-13], 
    run_mcmc(model_bssm, n_iter = 10, seed = 1)[-13])
  
  expect_error(mcmc_bsm <- run_mcmc(model_bssm, n_iter = 50, seed = 1), NA)
  
  expect_gt(mcmc_bsm$acceptance_rate, 0)
  expect_gte(min(mcmc_bsm$theta), 0)
  expect_lt(max(mcmc_bsm$theta), Inf)
  expect_true(is.finite(sum(mcmc_bsm$alpha)))

})


test_that("MCMC results for Poisson model are correct",{
  set.seed(123)
  model_bssm <- ng_bsm(rpois(10, exp(0.2) * (2:11)), P1 = diag(2, 2), sd_slope = 0,
    sd_level = uniform(2, 0, 10), u = 2:11, distribution = "poisson")
  
  expect_equal(run_mcmc(model_bssm, n_iter = 10, nsim_states = 5, seed = 1)[-13], 
    run_mcmc(model_bssm, n_iter = 10, nsim_states = 5, seed = 1)[-13])
  
  expect_error(mcmc_poisson <- run_mcmc(model_bssm, n_iter = 10, nsim_states = 5, seed = 42), NA)
  
  expect_gt(mcmc_poisson$acceptance_rate, 0)
  expect_gte(min(mcmc_poisson$theta), 0)
  expect_lt(max(mcmc_poisson$theta), Inf)
  expect_true(is.finite(sum(mcmc_poisson$alpha)))
  
})


test_that("MCMC results for SV model using IS-correction are correct",{
  set.seed(123)
  expect_error(model_bssm <- svm(rnorm(10), rho = uniform(0.95,-0.999,0.999), 
    sd_ar = halfnormal(1, 5), sigma = halfnormal(1, 2)), NA)
  
  expect_equal(run_mcmc(model_bssm, n_iter = 100, nsim_states = 10,
    method = "is2", seed = 1)[-14], 
    run_mcmc(model_bssm, n_iter = 100, nsim_states = 10, method = "is2", seed = 1)[-14])
  
  expect_equal(run_mcmc(model_bssm, n_iter = 100, nsim_states = 10,
    method = "is2", seed = 1, simulation_method = "psi")[-14], 
    run_mcmc(model_bssm, n_iter = 100, nsim_states = 10, 
      method = "is2", seed = 1, simulation_method = "psi")[-14])
  
  expect_equal(run_mcmc(model_bssm, n_iter = 100, nsim_states = 10,
    method = "is2", seed = 1, simulation_method = "bsf")[-14], 
    run_mcmc(model_bssm, n_iter = 100, nsim_states = 10, 
      method = "is2", seed = 1, simulation_method = "bsf")[-14])
  
  expect_error(mcmc_sv <- run_mcmc(model_bssm, n_iter = 100, nsim_states = 10,
    method = "is2", seed = 1, simulation_method = "bsf"), NA)
      
  expect_gt(mcmc_sv$acceptance_rate, 0)
  expect_true(is.finite(sum(mcmc_sv$theta)))
  expect_true(is.finite(sum(mcmc_sv$alpha)))
  expect_gte(min(mcmc_sv$weights), 0)
  expect_lt(max(mcmc_sv$weights), Inf)
})
