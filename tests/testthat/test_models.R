#' @srrstats {G5.8, G5.8a, G5.8b, G5.8c, G5.8d, BS2.1, BS2.1a}

context("Test models")

test_that("bad argument values for bsm throws an error", {
  expect_error(bsm_lg(numeric(0), 1, 1))
  expect_error(bsm_lg("character vector"))
  expect_error(bsm_lg(matrix(0, 2, 2)))
  expect_error(bsm_lg(1))
  expect_error(bsm_lg(c(1, Inf)))
  expect_error(bsm_lg(1:10, sd_y = "character"))
  expect_error(bsm_lg(1:10, sd_y = Inf))
  expect_error(bsm_lg(1:10, no_argument = 5))
  expect_error(bsm_lg(1:10, xreg = matrix(NA)))
  expect_error(bsm_lg(1:10, xreg = matrix(1:20), beta = uniform(0, 0, 1)))
  expect_error(bsm_lg(1:10, xreg = 1:10, beta = NA))
  expect_error(bsm_lg(1:10, 1, 1, 1, a1 = 1:4))
  expect_error(bsm_lg(1:10, 1, 1, 1, 1))
})

test_that("proper arguments for bsm don't throw an error", {
  expect_error(bsm_lg(1:10, 1, 1), NA)
  expect_error(bsm_lg(1:10, uniform(0, 0, 1), 1), NA)
  expect_error(bsm_lg(1:10, 1, 1, uniform(0, 0, 1)), NA)
  expect_error(bsm_lg(1:10, 1, 1, 1, 1, period = 3), NA)
  expect_error(bsm_lg(1:10, 1, 1, 1, 1, period = 3, xreg = matrix(1:10, 10), 
    beta = normal(0, 0, 10)), NA)
})


test_that("bad argument values for bsm_ng throws an error", {
  expect_error(bsm_ng(numeric(0), 1, 1, distribution = "poisson"))
  expect_error(bsm_ng("character vector", distribution = "poisson"))
  expect_error(bsm_ng(1:10, distribution = "poisson"))
  expect_error(bsm_ng(diag(2), distribution = "poisson", 
    sd_level = 1))
  expect_error(bsm_ng(1, distribution = "poisson"))
  expect_error(bsm_ng(c(1, Inf), distribution = "poisson"))
  expect_error(bsm_ng(1:10, sd_level = "character", distribution = "poisson"))
  expect_error(bsm_ng(1:10, sd_level = Inf, distribution = "poisson"))
  expect_error(bsm_ng(1:10, no_argument = 5, distribution = "poisson"))
  expect_error(bsm_ng(1:10, 1, 1, xreg = matrix(1:20), beta = uniform(0, 0, 1), 
    distribution = "poisson"))
  expect_error(bsm_ng(1:10, 1, 1, xreg = matrix(Inf, 10, 1), 
    beta = uniform(0, 0, 1), distribution = "poisson"))
  expect_error(bsm_ng(1:10, 1, 1, xreg = 1:10, beta = NA, 
    distribution = "poisson"))
  expect_error(bsm_ng(1:10, 1, 1, a1 = "a", distribution = "poisson"))
  expect_error(bsm_ng(1:2, 1, 1, 1, distribution = "poisson", period = 2))
  expect_error(bsm_ng(-(1:2), 1, 1, distribution = "poisson"))
  expect_error(bsm_ng(1:2 + 0.1, 1, 1,distribution = "poisson"))
  expect_error(bsm_ng(1:2, 1, sd_y = halfnormal(0, 1:2), 
    distribution = "poisson"))
})

test_that("proper arguments for ng_bsm don't throw an error", {
  expect_error(bsm_ng(1:10, 1, 1, distribution = "poisson"), NA)
  expect_error(bsm_ng(1:10, 1, 1, distribution = "POISSon"), NA)
  expect_error(bsm_ng(1:10, uniform(0, 0, 1), 1, distribution = "poisson"), NA)
  expect_error(bsm_ng(1:10, 1, uniform(0, 0, 1), distribution = "poisson"), NA)
  expect_error(bsm_ng(1:10, 1, 1, 1, period = 3, distribution = "poisson"), NA)
  expect_error(bsm_ng(1:10, 1, 1, 1, period = 3, xreg = matrix(1:10, 10), 
    beta = normal(0, 0, 10), distribution = "poisson"), NA)
})

test_that("bad argument values for svm throws an error", {
  expect_error(svm("character vector"))
  expect_error(svm(matrix(0, 2, 2)))
  expect_error(svm(1))
  expect_error(svm(c(1, Inf)))
  expect_error(svm(1:10, sd_level = "character"))
  expect_error(svm(1:10, rho = Inf))
  expect_error(svm(1:10, no_argument = 5))
  expect_error(svm(1:10, xreg = matrix(1:20), beta = uniform(0, 0, 1)))
  expect_error(svm(1:10, xreg = 1:10, beta = NA))
  expect_error(svm(1:10, 1, 1, a1 = 1))
})

test_that("proper arguments for svm don't throw an error", {
  expect_error(svm(1:10, rho = uniform(0.9, -0.9, 0.99), 
    mu = normal(0, 0, 2), sd_ar = halfnormal(1, 2)), NA)
})

test_that("multivariate non-gaussian model", {
  set.seed(1)
  y <- cbind(
    rpois(10, exp(cumsum(rnorm(10, sd = 0.1)))),
    rpois(10, exp(cumsum(rnorm(10, sd = 0.1)))))
  pfun <- function(theta) {
    dnorm(exp(theta), 0, 1, log = TRUE)
  }
  ufun <- function(theta) {
    list(R = array(diag(exp(theta)), c(2, 2, 1)))
  }
  
  expect_error(mng_model <- ssm_mng(y = data.frame(1:4,1:4), Z = diag(2),
    T = diag(2), R = 0.1 * diag(2), P1 = diag(2), distribution = "poisson",
    init_theta = log(c(0.1, 0.1)), prior_fn = pfun, update_fn = ufun))
  
  expect_error(mng_model <- ssm_mng(y = y - 10, Z = diag(2), T = diag(2), 
    R = 0.1 * diag(2), P1 = diag(2), distribution = "poisson",
    init_theta = log(c(0.1, 0.1)), prior_fn = pfun, update_fn = ufun))
  
  expect_error(ssm_mng(y = y + 0.1, Z = diag(2), T = diag(2), 
    R = 0.1 * diag(2), P1 = diag(2), distribution = "poisson",
    init_theta = log(c(0.1, 0.1)), prior_fn = pfun, update_fn = ufun))
  
  
  expect_error(mng_model <- ssm_mng(y = y, Z = diag(2), T = diag(2), 
    R = 0.1 * diag(2), P1 = diag(2), distribution = "poisson",
    init_theta = log(c(0.1, 0.1)), prior_fn = pfun, update_fn = ufun), NA)
  expect_error(logLik(mng_model, particles = 10), NA)
})
