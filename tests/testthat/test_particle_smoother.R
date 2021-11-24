
context("Test that particle smoothers work")

#' @srrstats {G5.9, G5.9a, G5.9b}

test_that("Test that trivial noise does not affect particle_smoother", {
  
  expect_error(model_bsm <- bsm_lg(rep(1, 5), sd_level = 0.05, sd_slope = 0.01, 
    sd_y = 0.01, a1 = c(1, 0), P1 = diag(0.01, 2)), NA)
  model_bsm2 <- model_bsm
  model_bsm2$y <- model_bsm2$y + .Machine$double.eps 
  expect_equal(
    particle_smoother(model_bsm, 1e5, seed = 1)$alphahat,
    particle_smoother(model_bsm2, 1e5, seed = 1)$alphahat)
})
test_that("Test that different seeds give comparable results", {
  
  expect_error(model_bsm <- bsm_lg(rep(1, 5), sd_level = 0.05, sd_slope = 0.01, 
    sd_y = 0.01, a1 = c(1, 0), P1 = diag(0.01, 2)), NA)
  expect_equal(
    particle_smoother(model_bsm, 1e5)$alphahat,
    particle_smoother(model_bsm, 1e5)$alphahat, tolerance = 0.001)
})

test_that("Test that particle_smoother for LGSSM works as Kalman smoother", {
  
  expect_error(model_bsm <- bsm_lg(rep(1, 5), sd_level = 0.05, sd_slope = 0.01, 
    sd_y = 0.01, a1 = c(1, 0), P1 = diag(0.01, 2)), NA)
  expect_equal(smoother(model_bsm)$alphahat, 
    particle_smoother(model_bsm, 1e5, seed = 1)$alphahat, tolerance = 1e-2)
})

test_that("Test that BSF&PSI particle_smoother for LGSSM are with MC error", {
  
  expect_error(model_bsm <- bsm_lg(rep(1, 5), sd_level = 0.05, sd_slope = 0.01, 
    sd_y = 0.01, a1 = c(1, 0), P1 = diag(0.01, 2)), NA)
  expect_error(out1 <- 
      particle_smoother(model_bsm, 1e4, method = "psi", seed = 1), NA)
  expect_error(out2 <- 
      particle_smoother(model_bsm, 1e4, method = "bsf", seed = 1), NA)
  expect_equal(out1$alphahat, 
    out2$alphahat, tolerance = 1e-2)
  expect_equal(out1$Vt, 
    out2$Vt, tolerance = 1e-2)
})

test_that("Particle smoother for LGSSM returns finite values", {
  
  expect_error(model_ssm_ulg <- ssm_ulg(y = 1:10, Z = matrix(c(1, 0), 2, 1),
    H = 2, T = array(c(1, 0, 1, 1), c(2, 2, 1)), 
    R = array(diag(2, 2), c(2, 2, 1)), 
    a1 = matrix(0, 2, 1), P1 = diag(2, 2), state_names = c("level", "slope")), 
    NA)
  expect_error(out <- particle_smoother(model_ssm_ulg, 10, seed = 1), 
    NA)
  expect_true(is.finite(sum(out$alpha)))
  expect_true(is.finite(sum(out$alphahat)))
  expect_true(is.finite(sum(out$Vt)))
})

test_that("Particle smoother for poisson bsm_ng returns finite values", {
  
  expect_error(model <- bsm_ng(1:10, sd_level = 2, sd_slope = 2, 
    P1 = diag(2, 2), distribution = "poisson"), NA)
  expect_error(out <- particle_smoother(model, 10, seed = 1), NA)
  expect_error(out <- particle_smoother(model, 10, method = "bsf", seed = 1), 
    NA)
  
  expect_true(is.finite(sum(out$alpha)))
  expect_true(is.finite(sum(out$alphahat)))
  expect_true(is.finite(sum(out$Vt)))
})

test_that("Particle smoother for binomial bsm_ng returns finite values", {
  
  expect_error(model <- bsm_ng(c(1, 0, 1, 1, 1, 0, 0, 0), sd_level = 2, 
    sd_slope = 2, P1 = diag(2, 2), 
    distribution = "binomial"), NA)
  expect_error(out <- particle_smoother(model, 10, seed = 1), NA)
  
  expect_true(is.finite(sum(out$alpha)))
  expect_true(is.finite(sum(out$alphahat)))
  expect_true(is.finite(sum(out$Vt)))
  
})

test_that("Particle smoother for NB bsm_ng returns finite values", {
  
  expect_error(model <- bsm_ng(c(1, 0, 1, 1, 1, 0, 0, 0), 
    sd_level = uniform(0.1,0,1), 
    sd_slope = halfnormal(0.1, 1), 
    P1 = diag(2, 2), phi = gamma(1, 2, 2),
    distribution = "negative binomial"), NA)
  
  expect_error(out <- particle_smoother(model, 10, seed = 1), NA)
  expect_true(is.finite(sum(out$alpha)))
  expect_true(is.finite(sum(out$alphahat)))
  expect_true(is.finite(sum(out$Vt)))
  
  expect_error(out <- particle_smoother(model, 10, method = "bsf", seed = 1), 
    NA)
  expect_true(is.finite(sum(out$alpha)))
  expect_true(is.finite(sum(out$alphahat)))
  expect_true(is.finite(sum(out$Vt)))
  
})


test_that("Particle smoother for svm returns finite values", {
  
  data("exchange")
  model <- svm(exchange[1:20], rho = uniform(0.98, -1, 1),
    sd_ar = halfnormal(0.01,0.1), mu = normal(0, 0, 1))
  
  expect_error(out1 <- 
      particle_smoother(model, 100, method = "psi", seed = 1), NA)
  expect_error(out2 <- 
      particle_smoother(model, 10000, method = "bsf", seed = 1), NA)
  
  expect_true(is.finite(sum(out1$alpha)))
  expect_true(is.finite(sum(out1$alphahat)))
  expect_true(is.finite(sum(out1$Vt)))
  
  expect_equal(out1$alphahat, out2$alphahat, tol = 0.1)
})

tol <- 1e-8
test_that("Test that linear-gaussian bsf smoother still works", {
  
  expect_error(model_ssm_ulg <- ssm_ulg(y = 1:10, Z = matrix(c(1, 0), 2, 1),
    H = 2, T = array(c(1, 0, 1, 1), c(2, 2, 1)), 
    R = array(diag(2, 2), c(2, 2, 1)), 
    a1 = matrix(0, 2, 1), P1 = diag(2, 2), state_names = c("level", "slope")), 
    NA)
  expect_error(bsf_ssm_ulg <- particle_smoother(model_ssm_ulg, 10, seed = 1, 
    method = "bsf"), 
    NA)
  expect_gte(min(bsf_ssm_ulg$weights), 0)
  expect_lt(max(bsf_ssm_ulg$weights), Inf)
  expect_true(is.finite(bsf_ssm_ulg$logLik))
  expect_true(is.finite(sum(bsf_ssm_ulg$alphahat)))
  expect_true(is.finite(sum(bsf_ssm_ulg$Vt)))
  
  expect_error(model_ar1_lg <- ar1_lg(y = 1:10, 
    rho = tnormal(0.6, 0, 0.5, -1, 1),
    sigma = gamma(1,2,2), sd_y = 0.1, mu = 1), NA)
  expect_error(bsf_ar1_lg <- particle_smoother(model_ar1_lg, 10, seed = 1, 
    method = "bsf"), NA)
  expect_gte(min(bsf_ar1_lg$weights), 0)
  expect_lt(max(bsf_ar1_lg$weights), Inf)
  expect_true(is.finite(bsf_ar1_lg$logLik))
  expect_true(is.finite(sum(bsf_ar1_lg$alphahat)))
  expect_true(is.finite(sum(bsf_ar1_lg$Vt)))
})


test_that("Test that binomial ar1_ng still works", {
  
  expect_error(model <- ar1_ng(c(1, 0, 1, 1, 1, 0, 0, 0), 
    rho = uniform(0.9, 0, 1), sigma = gamma(1, 2, 2), 
    mu = normal(1, 0, 1), 
    xreg = 1:8, beta = normal(0, 0, 0.1),
    distribution = "binomial"), NA)
  expect_error(bsf_binomial <- particle_smoother(model, 10, method = "bsf", 
    seed = 1), NA)
  
  expect_gte(min(bsf_binomial$weights), 0)
  expect_lt(max(bsf_binomial$weights), Inf)
  expect_true(is.finite(bsf_binomial$logLik))
  expect_true(is.finite(sum(bsf_binomial$alphahat)))
  expect_true(is.finite(sum(bsf_binomial$Vt)))
  
})
