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
