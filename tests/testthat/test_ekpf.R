context("Test EKF")


test_that("Particle filtering based on EKF works", {
  
  skip_on_cran()
  set.seed(1)
  n <- 10
  x <- y <- numeric(n)
  y[1] <- rnorm(1, exp(x[1]), 0.1)
  for(i in 1:(n-1)) {
    x[i+1] <- rnorm(1, sin(x[i]), 0.1)
    y[i+1] <- rnorm(1, exp(x[i+1]), 0.1)
  }
  
  pntrs <- cpp_example_model("nlg_sin_exp")
  
  expect_error(model_nlg <- ssm_nlg(y = y, a1 = pntrs$a1, P1 = pntrs$P1, 
    Z = pntrs$Z_fn, H = pntrs$H_fn, T = pntrs$T_fn, R = pntrs$R_fn, 
    Z_gn = pntrs$Z_gn, T_gn = pntrs$T_gn,
    theta = c(log_H = log(0.1), log_R = log(0.1)), 
    log_prior_pdf = pntrs$log_prior_pdf,
    n_states = 1, n_etas = 1, state_names = "state"), NA)
  
  expect_error(out <- ekpf_filter(model_nlg, 100), NA)
  expect_lt(out$logLik, 6)
  expect_gt(out$logLik, 1)
  expect_gte(min(out$w), 0-1e16)
  expect_lte(max(out$w), 1+1e16)
  expect_warning(ekpf_filter(model_nlg, nsim = 10))
  
  out_ekf <- particle_smoother(model_nlg, 1000, method = "ekf")
  out_psi <- particle_smoother(model_nlg, 1000, method = "psi")
  out_bsf <- particle_smoother(model_nlg, 1000, method = "bsf")
  expect_equal(out_ekf$alphahat[9:10], 
    c(0.0263875638744833, 0.0734903567971838), tol = 0.1)
  expect_equal(out_psi$alphahat[9:10], 
    c(0.0263875638744833, 0.0734903567971838), tol = 0.1)
  expect_equal(out_bsf$alphahat[9:10], 
    c(0.0263875638744833, 0.0734903567971838), tol = 0.1)
  
})

test_that("EKF and IEKF work", {
  skip_on_cran()
  set.seed(1)
  n <- 10
  x <- y <- numeric(n)
  y[1] <- rnorm(1, exp(x[1]), 0.1)
  for(i in 1:(n-1)) {
    x[i+1] <- rnorm(1, sin(x[i]), 0.1)
    y[i+1] <- rnorm(1, exp(x[i+1]), 0.1)
  }
  y[2:3] <- NA
  pntrs <- cpp_example_model("nlg_sin_exp")
  
  expect_error(model_nlg <- ssm_nlg(y = y, a1 = pntrs$a1, P1 = pntrs$P1, 
    Z = pntrs$Z_fn, H = pntrs$H_fn, T = pntrs$T_fn, R = pntrs$R_fn, 
    Z_gn = pntrs$Z_gn, T_gn = pntrs$T_gn,
    theta = c(log_H = log(0.1), log_R = log(0.1)), 
    log_prior_pdf = pntrs$log_prior_pdf,
    n_states = 1, n_etas = 1, state_names = "state"), NA)
  
  expect_equal(ekf(model_nlg)$logLik, 2.65163101109689)
  expect_equal(ekf(model_nlg, iekf_iter = 2)$logLik, 
    logLik(model_nlg, method = "ekf", iekf_iter = 2, particles = 0))
  expect_equal(ekf(model_nlg, iekf_iter = 1)$logLik, 2.61650080342709)
  expect_equal(ekf(model_nlg, iekf_iter = 1), 
    ekf(model_nlg, iekf_iter = 2))
  
  out_ekf1 <- ekf_smoother(model_nlg)
  out_ekf2 <- ekf_fast_smoother(model_nlg)
  expect_equal(out_ekf1$alphahat[9:10], 
    c(0.0333634309012196, 0.0797729159367873), tol = 0.1)
  expect_equal(out_ekf1$alphahat, out_ekf2)
  expect_equal(
    ekf_fast_smoother(model_nlg, iekf_iter = 2), 
    ekf_smoother(model_nlg, iekf_iter = 2)$alphahat)
  
  expect_error(ukf(model_nlg), NA)
  expect_error(ukf(model_nlg, alpha = -1))
  expect_error(ukf(model_nlg, beta = -1))
  expect_error(ukf(model_nlg, kappa = -1))
  expect_error(bootstrap_filter(model_nlg, 10), NA)
})
