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

test_that("results for poisson model are comparable to KFAS",{
  library("KFAS")
  model_KFAS <- SSModel(1:10 ~ SSMtrend(2, Q = list(0.01^2, 0)),
    distribution = "poisson", u = 2:11)
  model_KFAS$P1inf[] <- 0
  diag(model_KFAS$P1) <- 1e2
  
  model_bssm <- ng_bsm(1:10, P1 = diag(1e2,2), sd_slope = 0,
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


test_that("results for binomial model are comparable to KFAS",{
  library("KFAS")
  model_KFAS <- SSModel(1:10 ~ SSMtrend(2, Q = list(0.01^2, 0)),
    distribution = "binomial", u = 2:11)
  model_KFAS$P1inf[] <- 0
  diag(model_KFAS$P1) <- 1e2
  
  model_bssm <- ng_bsm(1:10, P1 = diag(1e2,2), sd_slope = 0,
    sd_level = 0.01, phi = 2:11, distribution = "binomial")
  
  expect_equal(logLik(model_KFAS,convtol = 1e-12), logLik(model_bssm,0))
  out_KFAS <- KFS(model_KFAS, filtering = "state", convtol = 1e-12)
  expect_error(out_bssm <- kfilter(model_bssm), NA)
  expect_equivalent(out_KFAS$a, out_bssm$at)
  expect_equivalent(out_KFAS$P, out_bssm$Pt)
  expect_error(out_bssm <- smoother(model_bssm), NA)
  expect_equivalent(out_KFAS$alphahat, out_bssm$alphahat)
  expect_equivalent(out_KFAS$V, out_bssm$Vt)
})


test_that("MCMC results for poisson model are correct",{
  set.seed(123)
  model_bssm <- ng_bsm(rpois(10,3), P1 = diag(2,2), sd_slope = 0,
    sd_level = uniform(2, 0, 10), phi = 22:31, distribution = "poisson")

  expect_error(out <- run_mcmc(model_bssm, n_iter = 10, nsim_states = 5), NA)

  testvalues <- structure(c(-31.9847662717918, -31.9847662717918, -31.9847662717918,
    -32.7146366602036, -31.6810801332842), .Dim = c(5L, 1L))
  expect_equivalent(testvalues, out$posterior)

  testvalues <- structure(c(1.89972367166746, 1.89972367166746, 1.89972367166746,
    1.82357173132233, 1.55941474919434), .Dim = c(5L, 1L), .Dimnames = list(
      NULL, "sd_level"), mcpar = c(6, 10, 1), class = "mcmc")
  expect_equivalent(testvalues, out$theta)

  testvalues <- c(-2.45678419725387, -3.58855684667486, 0.484418408546788, -1.37871770980265,
    0.484418408546788, 0.484418408546788)
  expect_equivalent(testvalues, out$alpha[c(1,10,20,25, 31, 60)])
})


