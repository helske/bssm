context("Test bsm")

test_that("bad argument values throws an error",{
  expect_error(bsm("character vector"))
  expect_error(bsm(matrix(0, 2, 2)))
  expect_error(bsm(1))
  expect_error(bsm(c(1, Inf)))
  expect_error(bsm(1:10, sd_y = "character"))
  expect_error(bsm(1:10, sd_y = NA))
  expect_error(bsm(1:10, sd_level = Inf))
  expect_error(bsm(1:10, no_argument = 5))
  expect_error(bsm(1:10, xreg = "abc"))
  expect_error(bsm(1:10, xreg = NA))
  expect_error(bsm(1:10, xreg = 1:20))
  expect_error(bsm(1:10, xreg = 1:10, beta = NA))
})

test_that("results are comparable to KFAS",{
  library("KFAS")
  model_KFAS <- SSModel(log10(AirPassengers) ~ SSMtrend(2, Q = list(0.01^2, 0)) +
      SSMseasonal(12, 0.005^2), H = 0.005^2)
  model_KFAS$P1inf[] <- 0
  diag(model_KFAS$P1) <- 1e2
  model_bssm <- bsm(log10(AirPassengers), P1 = diag(1e2,13),
    sd_y = 0.005, sd_level = 0.01, sd_seasonal = 0.005)
  expect_equal(logLik(model_KFAS), logLik(model_bssm))
  out_KFAS <- KFS(model_KFAS)
  expect_error(out_bssm <- kfilter(model_bssm), NA)
  expect_equivalent(out_KFAS$a, out_bssm$at)
  expect_equivalent(out_KFAS$P, out_bssm$Pt)
  expect_error(out_bssm <- smoother(model_bssm), NA)
  expect_equivalent(out_KFAS$alphahat, out_bssm$alphahat)
  expect_equal(out_KFAS$V, out_bssm$Vt, tolerance = 1e-5, check.attributes = FALSE)
})


test_that("different smoothers give identical results",{
  model_bssm <- bsm(log10(AirPassengers), P1 = diag(1e2,13),
    sd_y = 0.005, sd_level = 0.01, sd_seasonal = 0.005)

  expect_error(out_bssm1 <- smoother(model_bssm), NA)
  expect_error(out_bssm2 <- fast_smoother(model_bssm), NA)
  expect_equivalent(out_bssm2, out_bssm1$alphahat)
})


test_that("MCMC results are correct",{
  set.seed(123)
  model_bssm <- bsm(rnorm(10,3), P1 = diag(2,2), sd_level = 2)

  expect_error(out <- run_mcmc(model_bssm, n_iter = 5), NA)

  testvalues <- structure(c(
    -21.1188759509962, -20.9618902905707, -20.5927926916911),
    .Dim = c(3L, 1L))
  expect_equivalent(testvalues, out$logLik)

  testvalues <- structure(c(
    1.07190610570214, 0.880984917781493, 0.869554983660202,
    2.16848541839909, 2.2395102983541, 2.07258174715106),
    .Dim = c(3L, 2L), .Dimnames = list(NULL, c("sd_y", "sd_level")),
    mcpar = c(3, 5, 1), class = "mcmc")
  expect_equivalent(testvalues, out$theta)

  testvalues <- c(
    1.400889326126, 1.90882445483498,
    0.192057468590735, 1.97901949320989,
    -0.971428708875681, 0.645910489882064)
  expect_equivalent(testvalues, out$alpha[c(1,10,20,25, 31, 60)])

})


