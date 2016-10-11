context("Test as_gssm")


test_that("Test conversion from SSModel to gssm",{
  library(KFAS)
  model_KFAS <- SSModel(1:10 ~ SSMtrend(2, Q = list(2, 2), 
    P1 = diag(2e3, 2)), H = 2)
  expect_error(model_bssm <- gssm(y = ts(1:10), Z = matrix(c(1, 0), 2, 1), H = sqrt(2), 
    T = array(c(1, 0, 1, 1), c(2, 2, 1)), R = array(diag(sqrt(2), 2), c(2, 2, 1)), 
    a1 = matrix(0, 2, 1), P1 = diag(2e3, 2)), NA)
  expect_error(conv_model_bssm <- as_gssm(model_KFAS), NA)
})


test_that("Test conversion from SSModel to ngssm",{
  library(KFAS)
  model_KFAS <- SSModel(1:10 ~ SSMtrend(2, Q = list(2, 2),
    P1 = diag(2e3, 2)), u = 2, distribution = "negative binomial")
  expect_error(model_bssm <- ngssm(y = ts(1:10), Z = matrix(c(1, 0), 2, 1), phi = 2, 
    T = array(c(1, 0, 1, 1), c(2, 2, 1)), R = array(diag(sqrt(2), 2), c(2, 2, 1)), 
    a1 = matrix(0, 2, 1), P1 = diag(2e3, 2), distribution = "negative binomial", 
    state_names = c("level", "slope")), NA)
  expect_error(conv_model_bssm <- as_ngssm(model_KFAS), NA)
  expect_equivalent(model_bssm, conv_model_bssm)
})
