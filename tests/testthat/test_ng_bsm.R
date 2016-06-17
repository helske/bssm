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

test_that("MCMC results are correct",{
  set.seed(123)
  model_bssm <- ng_bsm(rpois(10,3), P1 = diag(2,2),
    sd_level = 2, phi = 22:31, distribution = "poisson")

  expect_error(out <- run_mcmc(model_bssm, n_iter = 5), NA)

  testvalues <- structure(c(-30.5898254659741,
    -30.559873871786, -30.5130394688329), .Dim = c(3L, 1L))
  expect_equivalent(testvalues, out$logLik)

  testvalues <- structure(c(1.92994662744397, 1.91792908167773, 1.89920013178759
  ), .Dim = c(3L, 1L), .Dimnames = list(NULL, "sd_level"), mcpar = c(3,
    5, 1), class = "mcmc")
  expect_equivalent(testvalues, out$theta)

  testvalues <- c(-1.75817378464323, -2.09820472708178, 0.0321910079468918,
    -1.78009326390443, -0.516269445583639, -0.0681578155549925)
  expect_equivalent(testvalues, out$alpha[c(1,10,20,25, 31, 60)])

  expect_error(out2 <- run_mcmc(model_bssm, n_iter = 5, nsim_states = 10), NA)

  testvalues <- c(
    -1.53451331164171, -1.65374734086221,
    -0.61385163351857, -1.18098534825777,
    -0.61385163351857, -0.61385163351857)
  expect_equivalent(testvalues, out2$alpha[c(1,10,20,25, 31, 60)])

})


