context("Test basics")

#' @srrstats {G5.4, G5.4b, G5.6, G5.6a, G5.6b, G5.7} Compare with KFAS.

tol <- 1e-6

test_that("results for Gaussian models are comparable to KFAS", {
  library("KFAS")
  model_KFAS <- SSModel(1:10 ~ SSMtrend(2, Q = list(0.01^2, 0)), H = 2)
  model_KFAS$P1inf[] <- 0
  diag(model_KFAS$P1) <- 1e2

  expect_error(bsm_lg(1:10, P1 = diag(1e2, 2), sd_slope = 0,
    sd_level = 0.01))
  expect_error(bsm_lg(1:10, P1 = diag(1e2, 2), sd_slope = 0,
    sd_y = 0.01))
  model_bssm <- bsm_lg(1:10, P1 = diag(1e2, 2), sd_slope = 0,
    sd_level = 0.01, sd_y = sqrt(2))

  expect_equal(logLik(model_KFAS, convtol = 1e-12), logLik(model_bssm, 0))
  out_KFAS <- KFS(model_KFAS, filtering = "state", convtol = 1e-12)
  expect_error(out_bssm <- kfilter(model_bssm), NA)
  expect_equivalent(out_KFAS$a, out_bssm$at, tolerance = tol)
  expect_equivalent(out_KFAS$P, out_bssm$Pt, tolerance = tol)
  expect_error(out_bssm <- smoother(model_bssm), NA)
  expect_equivalent(out_KFAS$alphahat, out_bssm$alphahat, tolerance = tol)
  expect_equivalent(out_KFAS$V, out_bssm$Vt, tolerance = tol)
})

test_that("results for multivariate Gaussian model are comparable to KFAS", {
  library("KFAS")
  # From the help page of ?KFAS
  data("Seatbelts", package = "datasets")
  kfas_model <- SSModel(log(cbind(front, rear)) ~ -1 +
      log(PetrolPrice) + log(kms) +
      SSMregression(~law, data = Seatbelts, index = 1) +
      SSMcustom(Z = diag(2), T = diag(2), R = matrix(1, 2, 1),
        Q = matrix(1), P1inf = diag(2)) +
      SSMseasonal(period = 12, sea.type = "trigonometric"),
    data = Seatbelts, H = matrix(NA, 2, 2))

  diag(kfas_model$P1) <- 50
  diag(kfas_model$P1inf) <- 0
  kfas_model$H <- structure(c(0.00544500509177812, 0.00437558178720609,
    0.00437558178720609, 0.00885692410165593), .Dim = c(2L, 2L, 1L))
  kfas_model$R <- structure(c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0152150188066314, 0.0144897116711475
  ), .Dim = c(29L, 1L, 1L), .Dimnames = list(c("log(PetrolPrice).front",
    "log(kms).front", "log(PetrolPrice).rear", "log(kms).rear", "law.front",
    "sea_trig1.front", "sea_trig*1.front", "sea_trig2.front",
    "sea_trig*2.front", "sea_trig3.front", "sea_trig*3.front",
    "sea_trig4.front", "sea_trig*4.front",
    "sea_trig5.front", "sea_trig*5.front", "sea_trig6.front", "sea_trig1.rear",
    "sea_trig*1.rear", "sea_trig2.rear", "sea_trig*2.rear", "sea_trig3.rear",
    "sea_trig*3.rear", "sea_trig4.rear", "sea_trig*4.rear", "sea_trig5.rear",
    "sea_trig*5.rear", "sea_trig6.rear", "custom1", "custom2"), NULL,
    NULL))

  bssm_model <- as_bssm(kfas_model)
  expect_equivalent(logLik(kfas_model), logLik(bssm_model), tolerance = tol)
  expect_equivalent(KFS(kfas_model)$alphahat,
      smoother(bssm_model)$alphahat, tolerance = tol)

})

test_that("different smoothers give identical results", {
  model_bssm <- bsm_lg(log10(AirPassengers), P1 = diag(1e2, 13), sd_slope = 0,
    sd_y = uniform(0.005, 0, 10), sd_level = uniform(0.01, 0, 10),
    sd_seasonal = uniform(0.005, 0, 1))

  expect_error(out_bssm1 <- smoother(model_bssm), NA)
  expect_error(out_bssm2 <- fast_smoother(model_bssm), NA)
  expect_equivalent(out_bssm2, out_bssm1$alphahat, tolerance = tol)
})


test_that("results for Poisson model are comparable to KFAS", {
  library("KFAS")
  set.seed(1)
  model_KFAS <- SSModel(rpois(10, exp(0.2) * (2:11)) ~
      SSMtrend(2, Q = list(0.01^2, 0)),
    distribution = "poisson", u = 2:11)
  model_KFAS$P1inf[] <- 0
  diag(model_KFAS$P1) <- 1e2

  model_bssm <- bsm_ng(model_KFAS$y, P1 = diag(1e2, 2), sd_slope = 0,
    sd_level = 0.01, u = 2:11, distribution = "poisson")

  expect_equal(logLik(model_KFAS), logLik(model_bssm, 0))
  out_KFAS <- KFS(model_KFAS, filtering = "state")
  expect_error(out_bssm <- kfilter(model_bssm), NA)
  expect_equivalent(out_KFAS$a, out_bssm$at, tolerance = tol)
  expect_equivalent(out_KFAS$P, out_bssm$Pt, tolerance = tol)
  expect_error(out_bssm <- smoother(model_bssm), NA)
  expect_equivalent(out_KFAS$alphahat, out_bssm$alphahat, tolerance = tol)
  expect_equivalent(out_KFAS$V, out_bssm$Vt, tolerance = tol)
})


test_that("results for binomial model are comparable to KFAS", {
  library("KFAS")
  set.seed(1)
  model_KFAS <- SSModel(rbinom(10, 2:11, 0.4) ~
      SSMtrend(2, Q = list(0.01^2, 0)),
    distribution = "binomial", u = 2:11)
  model_KFAS$P1inf[] <- 0
  diag(model_KFAS$P1) <- 1e2

  model_bssm <- bsm_ng(model_KFAS$y, P1 = diag(1e2, 2), sd_slope = 0,
    sd_level = 0.01, u = 2:11, distribution = "binomial")

  expect_equal(logLik(model_KFAS), logLik(model_bssm, 0))
  out_KFAS <- KFS(model_KFAS, filtering = "state")
  expect_error(out_bssm <- kfilter(model_bssm), NA)
  expect_equivalent(out_KFAS$a, out_bssm$at, tolerance = tol)
  expect_equivalent(out_KFAS$P, out_bssm$Pt, tolerance = tol)
  expect_error(out_bssm <- smoother(model_bssm), NA)
  expect_equivalent(out_KFAS$alphahat, out_bssm$alphahat, tolerance = tol)
  expect_equivalent(out_KFAS$V, out_bssm$Vt, tolerance = tol)
})

test_that("results for bivariate non-Gaussian model are comparable to KFAS", {
  library("KFAS")
  set.seed(1)
  n <- 10
  x1 <- cumsum(rnorm(n))
  x2 <- cumsum(rnorm(n, sd = 0.2))
  u <- rep(c(1, 15), c(4, 6))
  y <- cbind(rbinom(n, size = u, prob = plogis(x1)),
    rpois(n, u * exp(x1 + x2)), rgamma(n, 10, 10 / exp(x2)), rnorm(n, x2, 0.1))

  model_KFAS <- SSModel(y ~
      SSMtrend(1, Q = 1, a1 = -0.5, P1 = 0.5, type = "common", index = 1:2) +
      SSMtrend(1, Q = 0.2^2, P1 = 1, type = "common", index = 2:4),
    distribution = c("binomial", "poisson", "gamma", "gaussian"),
    u = cbind(u, u, 10, 0.1^2))
  model_bssm <- as_bssm(model_KFAS)

  approx_bssm <- gaussian_approx(model_bssm, conv_tol = 1e-16)
  approx_KFAS <- approxSSM(model_KFAS, tol = 1e-16)

  expect_equivalent(approx_bssm$y, approx_KFAS$y, tolerance = tol)
  expect_equivalent(approx_bssm$H^2, approx_KFAS$H, tolerance = tol)

  expect_equivalent(logLik(model_KFAS, nsim = 0),
    logLik(model_bssm, particles = 0), tolerance = tol)
  expect_equivalent(logLik(model_KFAS, nsim = 100, seed = 1),
    logLik(model_bssm, particles = 100, method = "spdk", seed = 1),
    tolerance = 1)

  expect_equivalent(
    logLik(model_bssm, particles = 100, method = "psi", seed = 1),
    logLik(model_bssm, particles = 100, method = "spdk", seed = 1),
    tolerance = 1)

  # note large tolerance due to the sd of bsf
  expect_equivalent(
    logLik(model_bssm, particles = 100, method = "psi", seed = 1),
    logLik(model_bssm, particles = 100, method = "bsf", seed = 1),
    tolerance = 10)

  out_KFAS <- KFS(model_KFAS)
  expect_error(out_bssm <- smoother(model_bssm), NA)
  expect_equivalent(out_KFAS$alphahat, out_bssm$alphahat, tolerance = tol)
  expect_equivalent(out_KFAS$V, out_bssm$Vt, tolerance = tol)
  is_KFAS <- importanceSSM(model_KFAS, nsim = 1e4)
  expect_error(is_bssm <- importance_sample(model_bssm, nsim = 1e4), NA)
  expect_equivalent(apply(is_bssm$alpha, 1:2, mean)[1:n, ],
    apply(is_KFAS$samples, 1:2, mean), tolerance = 0.1)
  expect_equivalent(apply(is_bssm$alpha, 1:2, sd)[1:n, ],
    apply(is_KFAS$samples, 1:2, sd), tolerance = 0.1)

})


test_that("multivariate normal pdf works", {

  expect_equivalent(bssm:::dmvnorm(1, 3, matrix(2, 1, 1), TRUE, TRUE),
    dnorm(1, 3, 2, log = TRUE), tolerance = tol)
  expect_equivalent(bssm:::dmvnorm(1, 3, matrix(4, 1, 1), TRUE, TRUE),
      dnorm(1, 3, 4, log = TRUE), tolerance = tol)

  set.seed(1)
  a <- crossprod(matrix(rnorm(9), 3, 3))
  logp1 <- expect_error(bssm:::dmvnorm(1:3, -0.1 * (3:1), a, FALSE, TRUE), NA)
  expect_equivalent(logp1, -14.0607446337904, tolerance = 1e-6)

  chola <- t(chol(a))
  logp2 <- expect_error(bssm:::dmvnorm(1:3, -0.1 * (3:1), chola, TRUE, TRUE),
    NA)
  expect_equivalent(logp2, logp1, tolerance = tol)

  b <- matrix(0, 3, 3)
  constant <- bssm:::precompute_dmvnorm(a, b, 0:2)
  expect_equivalent(logp1,
    bssm:::fast_dmvnorm(1:3, -0.1 * (3:1), b, 0:2, constant), tolerance = 1e-8)

  a[2, ] <- a[, 2] <- 0
  logp3 <- expect_error(bssm:::dmvnorm(1:3, -0.1 * (3:1), a, FALSE, TRUE), NA)
  expect_equivalent(logp3, -12.5587625856078, tolerance = 1e-6)
})

test_that("asymptotic_var fails with improper weights", {
  x <- rnorm(10)
  expect_error(asymptotic_var(x, 0))
  expect_error(asymptotic_var(x, rep(0, length(x))))
  expect_error(asymptotic_var(x, c(-1, runif(9))))
  expect_error(asymptotic_var(x, c(Inf, runif(9))))
})
