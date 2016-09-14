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
    sd_y = uniform(0.005, 0, 10), sd_level = uniform(0.01, 0, 10), 
    sd_seasonal = uniform(0.005, 0, 1))
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
    sd_y = uniform(0.005, 0, 10), sd_level = uniform(0.01, 0, 10), 
    sd_seasonal = uniform(0.005, 0, 1))

  expect_error(out_bssm1 <- smoother(model_bssm), NA)
  expect_error(out_bssm2 <- fast_smoother(model_bssm), NA)
  expect_equivalent(out_bssm2, out_bssm1$alphahat)
})


test_that("MCMC results are correct",{
  set.seed(123)
  model_bssm <- bsm(rnorm(10,3), P1 = diag(2,2), sd_y = uniform(1, 0, 10), 
    sd_level = uniform(1, 0, 10))

  expect_error(out <- run_mcmc(model_bssm, n_iter = 5, nsim = 5), NA)

  testvalues <- structure(c(-23.8764619769274, -23.8837816806405, -24.3877459631945
  ), .Dim = c(3L, 1L))
  expect_equivalent(testvalues, out$posterior)

  testvalues <- structure(c(1.00257036620254, 1.11184780781035, 1.28864979167333, 
    1.20571811306111, 0.945143652508575, 0.952519531720265), .Dim = c(3L, 
      2L), .Dimnames = list(NULL, c("sd_y", "sd_level")), mcpar = c(3, 
        5, 1), class = "mcmc")
  expect_equivalent(testvalues, out$theta)

  testvalues <- c(0.648030303930946, 3.3218455394064, 0.968503953271118, 4.28022074618944, 
    0.195276265539814, -0.132712327951947)
  expect_equivalent(testvalues, out$alpha[c(1,10,20,25, 31, 60)])

})


