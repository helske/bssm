context("Test that simulation smoother work")


test_that("Test that sim_smoother for LGSSM works as Kalman smoother", {
  y <- c(0.89, -0.05, -1.9, -1.9, 1.77, -0.22)
  expect_error(model_bsm <- bsm_lg(y, sd_level = 1, sd_slope = 0.01, 
    sd_y = 1, a1 = c(0, 0), P1 = diag(2)), NA)
  expect_error(sims <- sim_smoother(model_bsm, nsim = 10, 
    use_antithetic = TRUE), NA)
  expect_equal(smoother(model_bsm)$alphahat,
    as.ts(apply(sims, 1:2, mean)))
  expect_error(sims <- sim_smoother(model_bsm, nsim = 10, 
    use_antithetic = "blaa"))
  expect_error(sims <- sim_smoother(model_bsm, nsim = 10, 
    use_antithetic = 1))
})


test_that("sim_smoother for non-gaussian model works as Kalman smoother", {
  
    y <- c(11, 3, 1, 1, 354, 2)
    expect_error(model <- bsm_ng(y, sd_level = 1, 
      sd_seasonal = 0.1, period = 4, 
      P1 = diag(c(1, 0.1, 0.1, 0.1)), distribution = "poisson"), NA)
    expect_error(sims <- sim_smoother(model, nsim = 10, 
      use_antithetic = TRUE), NA)
    expect_equal(smoother(model)$alphahat,
      as.ts(apply(sims, 1:2, mean)))
})


