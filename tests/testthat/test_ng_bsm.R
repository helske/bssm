context("Test ng_bsm")

test_that("bad argument values throws an error",{
  expect_error(ng_bsm("character vector"))
  expect_error(ng_bsm(matrix(0, 2, 2)))
  expect_error(ng_bsm(1))
  expect_error(ng_bsm(c(1, Inf)))
  expect_error(ng_bsm(1:10, sd_y = "character"))
  expect_error(ng_bsm(1:10, sd_level = Inf))
  expect_error(ng_bsm(1:10, no_argument = 5))
  expect_error(ng_bsm(1:10, xreg = "abc"))
  expect_error(ng_bsm(1:10, xreg = NA))
  expect_error(ng_bsm(1:10, xreg = 1:20))
  expect_error(ng_bsm(1:10, xreg = 1:10, beta = NA))
})

test_that("results are comparable to KFAS",{
  library("KFAS")
  model_KFAS <- SSModel(1:10 ~ SSMtrend(2, Q = list(0.01^2, 0)),
    distribution = "poisson", u = 2:11)
  model_KFAS$P1inf[] <- 0
  diag(model_KFAS$P1) <- 1e2
  model_bssm <- ng_bsm(1:10, P1 = diag(1e2,2),
    sd_level = 0.01, phi = 2:11, distribution = "poisson")
  expect_equal(logLik(model_KFAS,convtol = 1e-12), logLik(model_bssm,0))
  out_KFAS <- KFS(model_KFAS, filtering = "state", convtol = 1e-12)
  expect_error(out_bssm <- kfilter(model_bssm), NA)
  expect_equivalent(out_KFAS$a, out_bssm$at)
  expect_equivalent(out_KFAS$P, out_bssm$Pt)
  expect_error(out_bssm <- smoother(model_bssm), NA)
  expect_equivalent(out_KFAS$alphahat, out_bssm$alphahat)
  expect_equivalent(out_KFAS$V, out_bssm$Vt)
})

