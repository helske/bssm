context("Test as_bssm")


test_that("Test conversion from SSModel to ssm_ulg",{
  library(KFAS)
  model_KFAS <- SSModel(1:10 ~ SSMtrend(2, Q = list(2, 2), 
    P1 = diag(2e3, 2)), H = 2)
  expect_error(model_bssm <- ssm_ulg(y = ts(1:10), Z = matrix(c(1, 0), 2, 1), H = sqrt(2), 
    T = array(c(1, 0, 1, 1), c(2, 2, 1)), R = array(diag(sqrt(2), 2), c(2, 2, 1)), 
    a1 = matrix(0, 2, 1), P1 = diag(2e3, 2), init_theta = c(0,0)), NA)
  expect_error(conv_model_bssm <- as_bssm(model_KFAS, init_theta = c(0, 0)), NA)
  expect_equivalent(model_bssm, conv_model_bssm)
  expect_equivalent(logLik(conv_model_bssm), logLik(model_KFAS))
  
})

test_that("Test conversion from SSModel to ssm_ung",{
  library(KFAS)
  model_KFAS <- SSModel(1:10 ~ SSMtrend(2, Q = list(2, 2),
    P1 = diag(2e3, 2)), u = 2, distribution = "negative binomial")
  expect_error(model_bssm <- ssm_ung(y = ts(1:10), Z = matrix(c(1, 0), 2, 1), phi = 2, 
    T = array(c(1, 0, 1, 1), c(2, 2, 1)), R = array(diag(sqrt(2), 2), c(2, 2, 1)), 
    a1 = matrix(0, 2, 1), P1 = diag(2e3, 2), distribution = "negative binomial", 
    state_names = c("level", "slope"), init_theta = c(0,0)), NA)
  expect_error(conv_model_bssm <- as_bssm(model_KFAS, init_theta = c(0,0)), NA)
  expect_equivalent(model_bssm, conv_model_bssm)
})

test_that("Test conversion from SSModel to ssm_mng",{
  library(KFAS)
  model_KFAS <- SSModel(cbind(1:10, 1:10) ~ SSMtrend(1, Q = 2,
    P1 = diag(2e3, 2)), u = 2, distribution = "negative binomial")
  expect_error(model_bssm <- ssm_mng(y = cbind(1:10, 1:10), Z = diag(2), phi = 2, 
    T = diag(2), R = array(diag(sqrt(2), 2), c(2, 2, 1)), 
    a1 = matrix(0, 2, 1), P1 = diag(2e3, 2), distribution = "negative binomial", 
    state_names = c("level", "slope"), init_theta = c(0,0)), NA)
  expect_error(conv_model_bssm <- as_bssm(model_KFAS, init_theta = c(0,0)), NA)
  expect_equivalent(model_bssm, conv_model_bssm)
})
