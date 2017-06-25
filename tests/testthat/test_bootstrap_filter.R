
context("Test that bootstrap_filter works")

test_that("Test that bsm gives identical results with gssm",{
  
  expect_error(model_gssm <- gssm(y = 1:10, Z = matrix(c(1, 0), 2, 1), H = 2, 
    T = array(c(1, 0, 1, 1), c(2, 2, 1)), R = array(diag(2, 2), c(2, 2, 1)), 
    a1 = matrix(0, 2, 1), P1 = diag(2, 2), state_names = c("level", "slope")), NA)
  expect_error(bsf_gssm <- bootstrap_filter(model_gssm, 2, seed = 2), NA)
  expect_equal(bsf_gssm$logLik, -36.6658020894499)
  
  testvalues <- c(-0.0319113702364748, 5.4722485818937, 0.936842667647643, 2.19849347885756)
  expect_equal(bsf_gssm$alpha[c(1, 2, 11, 20)], testvalues)
  
  
  expect_error(model_bsm <- bsm(1:10, sd_level = 2, sd_slope = 2, sd_y = 2, 
    P1 = diag(2, 2)), NA)
  expect_error(bsf_bsm <- bootstrap_filter(model_bsm, 2, seed = 2), NA)
  expect_equal(bsf_bsm, bsf_gssm)
})

test_that("Test that poisson ng_bsm give identical results with ngssm",{
  
  expect_error(model_ng_bsm <- ng_bsm(1:10, sd_level = 2, sd_slope = 2, P1 = diag(2, 2), 
    distribution = "poisson"), NA)
  expect_error(bsf_ng_bsm <- bootstrap_filter(model_ng_bsm, 100, seed = 2), NA)
  expect_equal(bsf_ng_bsm$logLik, -36.4165815198814)
  
})

test_that("Test that binomial ng_bsm give identical results with ngssm",{
  
  expect_error(model <- ng_bsm(c(1,0,1,1,1,0,0,0), sd_level = 2, sd_slope = 2, P1 = diag(2, 2), 
    distribution = "binomial"), NA)
  expect_error(out <- bootstrap_filter(model, 100, seed = 2), NA)
  expect_equal(out$logLik, -6.7024830631489)
  
})


test_that("Test that negative binomial ng_bsm give identical results with ngssm",{
  expect_error(model <- ng_bsm(c(1,0,1,1,1,0,0,0), sd_level = 2, sd_slope = 2, 
    P1 = diag(2, 2), distribution = "negative binomial", phi = 0.1, u = 2), NA)
  expect_error(out <- bootstrap_filter(model, 100, seed = 2), NA)
  expect_equal(out$logLik, -13.6549970967291)
})


test_that("Test that still svm works",{
  data("exchange")
  model <- svm(exchange, rho = uniform(0.98,-0.999,0.999), 
    sd_ar = halfnormal(0.2, 5), sigma = halfnormal(1, 2))
  
  expect_error(out <- bootstrap_filter(model, 10, seed = 2), NA)
  expect_equal(out$logLik, -933.056099469693)
  
})

