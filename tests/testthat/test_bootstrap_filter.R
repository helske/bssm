
context("Test that bootstrap_filter works")


test_that("Test that bsm_lg gives identical results with ssm_ulg", {
  expect_error(model_ssm_ulg <- ssm_ulg(y = 1:10, Z = matrix(c(1, 0), 2, 1), 
    H = 2, T = array(c(1, 0, 1, 1), c(2, 2, 1)), 
    R = array(diag(2, 2), c(2, 2, 1)), 
    a1 = matrix(0, 2, 1), P1 = diag(2, 2), state_names = c("level", "slope")), 
    NA)
  expect_error(bsf_ssm_ulg <- bootstrap_filter(model_ssm_ulg, 10, seed = 1), 
    NA)
  expect_error(model_bsm <- bsm_lg(1:10, sd_level = 2, sd_slope = 2, sd_y = 2, 
    P1 = diag(2, 2)), NA)
  expect_error(bsf_bsm <- bootstrap_filter(model_bsm, 10, seed = 1), NA)
  expect_equal(bsf_bsm, bsf_ssm_ulg, tolerance = 1e-8)
})


tol <- 1e-8
test_that("Test that linear-gaussian bsf still works", {
  
  expect_error(model_ssm_ulg <- ssm_ulg(y = 1:10, Z = matrix(c(1, 0), 2, 1),
    H = 2, T = array(c(1, 0, 1, 1), c(2, 2, 1)), 
    R = array(diag(2, 2), c(2, 2, 1)), 
    a1 = matrix(0, 2, 1), P1 = diag(2, 2), state_names = c("level", "slope")), 
    NA)
  expect_error(bsf_ssm_ulg <- bootstrap_filter(model_ssm_ulg, 10, seed = 1), 
    NA)
  expect_gte(min(bsf_ssm_ulg$weights), 0)
  expect_lt(max(bsf_ssm_ulg$weights), Inf)
  expect_true(is.finite(bsf_ssm_ulg$logLik))
  expect_true(is.finite(sum(bsf_ssm_ulg$att)))
  expect_true(is.finite(sum(bsf_ssm_ulg$Ptt)))
  
  expect_error(model_ar1_lg <- ar1_lg(y = 1:10, 
    rho = tnormal(0.6, 0, 0.5, -1, 1),
    sigma = gamma(1,2,2), sd_y = 0.1, mu = 1), NA)
  expect_error(bsf_ar1_lg <- bootstrap_filter(model_ar1_lg, 10, seed = 1), 
    NA)
  expect_gte(min(bsf_ar1_lg$weights), 0)
  expect_lt(max(bsf_ar1_lg$weights), Inf)
  expect_true(is.finite(bsf_ar1_lg$logLik))
  expect_true(is.finite(sum(bsf_ar1_lg$att)))
  expect_true(is.finite(sum(bsf_ar1_lg$Ptt)))
})

test_that("Test that poisson bsm_ng still works", {
  
  expect_error(model <- bsm_ng(1:10, sd_level = 2, sd_slope = 2, 
    P1 = diag(2, 2), distribution = "poisson"), NA)
  expect_error(bsf_poisson <- bootstrap_filter(model, 10, seed = 1), NA)

  expect_gte(min(bsf_poisson$weights), 0)
  expect_lt(max(bsf_poisson$weights), Inf)
  expect_true(is.finite(bsf_poisson$logLik))
  expect_true(is.finite(sum(bsf_poisson$att)))
  expect_true(is.finite(sum(bsf_poisson$Ptt)))
})

test_that("Test that binomial ar1_ng still works", {
  
  expect_error(model <- ar1_ng(c(1, 0, 1, 1, 1, 0, 0, 0), 
    rho = uniform(0.9, 0, 1), sigma = gamma(1, 2, 2), 
    mu = normal(1, 0, 1), 
    xreg = 1:8, beta = normal(0, 0, 0.1),
    distribution = "binomial"), NA)
  expect_error(bsf_binomial <- bootstrap_filter(model, 10, seed = 1), NA)
  
  expect_gte(min(bsf_binomial$weights), 0)
  expect_lt(max(bsf_binomial$weights), Inf)
  expect_true(is.finite(bsf_binomial$logLik))
  expect_true(is.finite(sum(bsf_binomial$att)))
  expect_true(is.finite(sum(bsf_binomial$Ptt)))
  
})


test_that("Test that negative binomial bsm_ng still works", {
  
  expect_error(model <- bsm_ng(c(1, 0, 1, 1, 1, 0, 0, 0), sd_level = 2, 
    sd_slope = 2, P1 = diag(2, 2), 
    distribution = "negative binomial", phi = 0.1, u = 2), NA)
  expect_error(bsf_nbinomial <- bootstrap_filter(model, 10, seed = 1), NA)
  
  expect_gte(min(bsf_nbinomial$weights), 0)
  expect_lt(max(bsf_nbinomial$weights), Inf)
  expect_true(is.finite(bsf_nbinomial$logLik))
  expect_true(is.finite(sum(bsf_nbinomial$att)))
  expect_true(is.finite(sum(bsf_nbinomial$Ptt)))
})


test_that("Test that still svm works", {
  data("exchange")
  model <- svm(exchange, rho = uniform(0.98, -0.999, 0.999), 
    sd_ar = halfnormal(0.2, 5), sigma = halfnormal(1, 2))
  
  expect_error(bsf_svm <- bootstrap_filter(model, 10, seed = 1), NA)
  
  expect_gte(min(bsf_svm$weights), 0)
  expect_lt(max(bsf_svm$weights), Inf)
  expect_true(is.finite(bsf_svm$logLik))
  expect_true(is.finite(sum(bsf_svm$att)))
  expect_true(is.finite(sum(bsf_svm$Ptt)))
  
})

