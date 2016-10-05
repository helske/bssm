context("Test basics")

test_that("results for gaussian model are comparable to KFAS",{
  library("KFAS")
  model_KFAS <- SSModel(1:10 ~ SSMtrend(2, Q = list(0.01^2, 0)), H = 2)
  model_KFAS$P1inf[] <- 0
  diag(model_KFAS$P1) <- 1e2
  
  model_bssm <- bsm(1:10, P1 = diag(1e2,2), sd_slope = 0,
    sd_level = 0.01, sd_y = sqrt(2))
  
  expect_equal(logLik(model_KFAS,convtol = 1e-12), logLik(model_bssm,0))
  out_KFAS <- KFS(model_KFAS, filtering = "state", convtol = 1e-12)
  expect_error(out_bssm <- kfilter(model_bssm), NA)
  expect_equivalent(out_KFAS$a, out_bssm$at)
  expect_equivalent(out_KFAS$P, out_bssm$Pt)
  expect_error(out_bssm <- smoother(model_bssm), NA)
  expect_equivalent(out_KFAS$alphahat, out_bssm$alphahat)
  expect_equivalent(out_KFAS$V, out_bssm$Vt)
})



test_that("different smoothers give identical results",{
  model_bssm <- bsm(log10(AirPassengers), P1 = diag(1e2,13), sd_slope = 0,
    sd_y = uniform(0.005, 0, 10), sd_level = uniform(0.01, 0, 10), 
    sd_seasonal = uniform(0.005, 0, 1))
  
  expect_error(out_bssm1 <- smoother(model_bssm), NA)
  expect_error(out_bssm2 <- fast_smoother(model_bssm), NA)
  expect_equivalent(out_bssm2, out_bssm1$alphahat)
})


test_that("results for poisson model are comparable to KFAS",{
  library("KFAS")
  model_KFAS <- SSModel(rpois(10, exp(0.2) * (2:11)) ~ SSMtrend(2, Q = list(0.01^2, 0)),
    distribution = "poisson", u = 2:11)
  model_KFAS$P1inf[] <- 0
  diag(model_KFAS$P1) <- 1e2
  
  model_bssm <- ng_bsm(model_KFAS$y, P1 = diag(1e2,2), sd_slope = 0,
    sd_level = 0.01, u = 2:11, distribution = "poisson")
  
  expect_equal(logLik(model_KFAS,convtol = 1e-12), logLik(model_bssm,0))
  out_KFAS <- KFS(model_KFAS, filtering = "state", convtol = 1e-12)
  expect_error(out_bssm <- kfilter(model_bssm), NA)
  expect_equivalent(out_KFAS$a, out_bssm$at)
  expect_equivalent(out_KFAS$P, out_bssm$Pt)
  expect_error(out_bssm <- smoother(model_bssm), NA)
  expect_equivalent(out_KFAS$alphahat, out_bssm$alphahat)
  expect_equivalent(out_KFAS$V, out_bssm$Vt)
})


test_that("results for binomial model are comparable to KFAS", {
  library("KFAS")
  model_KFAS <- SSModel(rbinom(10, 2:11, 0.4) ~ SSMtrend(2, Q = list(0.01^2, 0)),
    distribution = "binomial", u = 2:11)
  model_KFAS$P1inf[] <- 0
  diag(model_KFAS$P1) <- 1e2
  
  model_bssm <- ng_bsm(model_KFAS$y, P1 = diag(1e2,2), sd_slope = 0,
    sd_level = 0.01, u = 2:11, distribution = "binomial")
  
  expect_equal(logLik(model_KFAS,convtol = 1e-12), logLik(model_bssm,0))
  out_KFAS <- KFS(model_KFAS, filtering = "state", convtol = 1e-12)
  expect_error(out_bssm <- kfilter(model_bssm), NA)
  expect_equivalent(out_KFAS$a, out_bssm$at)
  expect_equivalent(out_KFAS$P, out_bssm$Pt)
  expect_error(out_bssm <- smoother(model_bssm), NA)
  expect_equivalent(out_KFAS$alphahat, out_bssm$alphahat)
  expect_equivalent(out_KFAS$V, out_bssm$Vt)
})


