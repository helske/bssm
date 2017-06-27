
context("Test that bootstrap_filter works")
tol <- 1e-8
test_that("Test that bsm gives identical results with gssm",{
  
  expect_error(model_gssm <- gssm(y = 1:10, Z = matrix(c(1, 0), 2, 1), H = 2, 
    T = array(c(1, 0, 1, 1), c(2, 2, 1)), R = array(diag(2, 2), c(2, 2, 1)), 
    a1 = matrix(0, 2, 1), P1 = diag(2, 2), state_names = c("level", "slope")), NA)
  expect_error(bsf_gssm <- bootstrap_filter(model_gssm, 10, seed = 1), NA)
  bsf_gssm_ref <- readRDS("bsf_reference/bsf_gssm_ref.rda")
  expect_equal(bsf_gssm$logLik, bsf_gssm_ref$logLik, tolerance = tol)
  
  expect_equal(bsf_gssm$alpha, bsf_gssm_ref$alpha, tolerance = tol)
  
  
  expect_error(model_bsm <- bsm(1:10, sd_level = 2, sd_slope = 2, sd_y = 2, 
    P1 = diag(2, 2)), NA)
  expect_error(bsf_bsm <- bootstrap_filter(model_bsm, 10, seed = 1), NA)
  expect_equal(bsf_bsm, bsf_gssm, tolerance = tol)
})

test_that("Test that poisson ng_bsm still works",{
  
  expect_error(model <- ng_bsm(1:10, sd_level = 2, sd_slope = 2, P1 = diag(2, 2), 
    distribution = "poisson"), NA)
  expect_error(bsf_poisson <- bootstrap_filter(model, 10, seed = 1), NA)
  bsf_poisson_ref <- readRDS("bsf_reference/bsf_poisson_ref.rda")
  expect_equal(bsf_poisson, bsf_poisson_ref, tolerance = tol)
})

test_that("Test that binomial ng_bsm still works",{
  
  expect_error(model <- ng_bsm(c(1,0,1,1,1,0,0,0), sd_level = 2, sd_slope = 2, P1 = diag(2, 2), 
    distribution = "binomial"), NA)
  expect_error(bsf_binomial <- bootstrap_filter(model, 10, seed = 1), NA)
  bsf_binomial_ref <- readRDS("bsf_reference/bsf_binomial_ref.rda")
  expect_equal(bsf_binomial, bsf_binomial_ref, tolerance = tol)
  
})



test_that("Test that negative binomial ng_bsm still works",{
  
  expect_error(model <- ng_bsm(c(1,0,1,1,1,0,0,0), sd_level = 2, sd_slope = 2, P1 = diag(2, 2), 
    distribution = "negative binomial", phi = 0.1, u = 2), NA)
  expect_error(bsf_nbinomial <- bootstrap_filter(model, 10, seed = 1), NA)
  bsf_nbinomial_ref <- readRDS("bsf_reference/bsf_nbinomial_ref.rda")
  expect_equal(bsf_nbinomial, bsf_nbinomial_ref, tolerance = tol)
})


test_that("Test that still svm works",{
  data("exchange")
  model <- svm(exchange, rho = uniform(0.98,-0.999,0.999), 
    sd_ar = halfnormal(0.2, 5), sigma = halfnormal(1, 2))
  
  expect_error(bsf_svm <- bootstrap_filter(model, 10, seed = 1), NA)
  bsf_svm_ref <- readRDS("bsf_reference/bsf_svm_ref.rda")
  expect_equal(bsf_svm, bsf_svm_ref, tolerance = tol)
  
})

