context("Test importance_sample")

test_that("Test that poisson ng_bsm give identical results with ssm_mng",{
  
  expect_error(model_ssm_mng <- ssm_mng(y = 1:10, Z = matrix(c(1, 0), 2, 1),
    T = array(c(1, 0, 1, 1), c(2, 2, 1)), R = array(diag(2, 2), c(2, 2, 1)), 
    a1 = matrix(0, 2, 1), P1 = diag(2, 2), state_names = c("level", "slope"),
    distribution = "poisson"), NA)
  expect_error(sim_ssm_mng <- importance_sample(model_ssm_mng, 4, seed = 2), NA)
  expect_error(model_ng_bsm <- ng_bsm(1:10, sd_level = 2, sd_slope = 2, P1 = diag(2, 2), 
    distribution = "poisson"), NA)
  expect_error(sim_ng_bsm <- importance_sample(model_ng_bsm, 4, seed = 2), NA)
  expect_equal(sim_ng_bsm, sim_ssm_mng)
})

test_that("Test that svm still works",{
  data("exchange")
  model <- svm(exchange, rho = uniform(0.98,-0.999,0.999), 
    sd_ar = halfnormal(0.2, 5), sigma = halfnormal(1, 2))
  
  expect_error(sim <- importance_sample(model, 10, seed = 2), NA)
  
  expect_gte(min(sim$weights), 0)
  expect_lt(max(sim$weights), Inf)
  expect_true(is.finite(sum(sim$states)))
  expect_true(is.finite(sum(sim$weights)))
})

