context("Test as_bssm")


test_that("Test conversion from SSModel to ssm_ulg", {
  library(KFAS)
  model_KFAS <- SSModel(1:10 ~ SSMtrend(2, Q = list(2, 2), 
    P1 = diag(2e3, 2)), H = 2)
  expect_error(model_bssm <- ssm_ulg(y = ts(1:10), Z = matrix(c(1, 0), 2, 1), 
    H = sqrt(2), 
    T = array(c(1, 0, 1, 1), c(2, 2, 1)), 
    R = array(diag(sqrt(2), 2), c(2, 2, 1)), 
    a1 = matrix(0, 2, 1), P1 = diag(2e3, 2), init_theta = c(0, 0)), NA)
  expect_error(conv_model_bssm <- as_bssm(model_KFAS, 
    init_theta = c(0, 0)), NA)
  expect_equivalent(model_bssm, conv_model_bssm)
  expect_equivalent(logLik(conv_model_bssm), logLik(model_KFAS))
  
})

test_that("Test conversion from SSModel to ssm_ung", {
  library(KFAS)
  model_KFAS <- SSModel(1:10 ~ SSMtrend(2, Q = list(2, 2),
    P1 = diag(2e3, 2)), u = 2, distribution = "negative binomial")
  expect_error(model_bssm <- ssm_ung(y = ts(1:10), Z = matrix(c(1, 0), 2, 1), 
    phi = 2, T = array(c(1, 0, 1, 1), c(2, 2, 1)), 
    R = array(diag(sqrt(2), 2), c(2, 2, 1)), 
    a1 = matrix(0, 2, 1), P1 = diag(2e3, 2), 
    distribution = "negative binomial", 
    state_names = c("level", "slope"), init_theta = c(0, 0)), NA)
  expect_error(conv_model_bssm <- as_bssm(model_KFAS, init_theta = c(0, 0)), NA)
  expect_equivalent(model_bssm, conv_model_bssm)
})

test_that("Test conversion from SSModel to ssm_mng", {
  library(KFAS)
  set.seed(1)
  y <- matrix(rbinom(20, size = 10, prob = plogis(rnorm(20, sd = 0.2))), 10, 2)
  model_KFAS <- SSModel(y ~ SSMtrend(1, Q = diag(0.2^2, 2),
    P1 = diag(2)), u = 10, distribution = "binomial")
  expect_error(model_bssm <- ssm_mng(y, Z = diag(2), u = 10, 
    T = diag(2), R = array(diag(0.2, 2), c(2, 2, 1)), 
    a1 = matrix(0, 2, 1), P1 = diag(2), distribution = "binomial", 
    init_theta = c(0, 0)), NA)
  # to make the attributes match
  model_bssm$u <- as.ts(model_bssm$u)
  model_bssm$initial_mode <- as.ts(model_bssm$initial_mode)
  expect_error(conv_model_bssm <- as_bssm(model_KFAS, init_theta = c(0, 0)), NA)
  expect_equivalent(model_bssm, conv_model_bssm)
})

test_that("Test that time-varying parameters fail", {
  library(KFAS)
  model_KFAS <- SSModel(1:10 ~ 1, u = 1:10, distribution = "negative binomial")
  expect_error(as_bssm(model_KFAS))
  
  model_KFAS <- SSModel(1:10 ~ 1, u = 1:10, distribution = "gamma")
  expect_error(as_bssm(model_KFAS))
  
  model_KFAS <- SSModel(cbind(1:10, 1:10) ~ 1, u = matrix(1:20, 10, 2), 
    distribution = "negative binomial")
  expect_error(as_bssm(model_KFAS))
  
  model_KFAS <- SSModel(cbind(1:10, 1:10) ~ 1, u = matrix(1:20,10,2), 
    distribution = "gamma")
  expect_error(as_bssm(model_KFAS))
  
  model_KFAS <- SSModel(cbind(1:10, 1:10) ~ 1, u = cbind(1, 1:10), 
    distribution = c("gamma", "gaussian"))
  expect_error(as_bssm(model_KFAS))
  
  model_KFAS <- SSModel(cbind(1:10, 1:10) ~ 1, u = matrix(1:20,10,2), 
    distribution = c("binomial", "poisson"))
  expect_error(as_bssm(model_KFAS), NA)
})
