context("Test basics")

test_that("results for gaussian model are comparable to KFAS",{
  library("KFAS")
  model_KFAS <- SSModel(1:10 ~ SSMtrend(2, Q = list(0.01^2, 0)), H = 2)
  model_KFAS$P1inf[] <- 0
  diag(model_KFAS$P1) <- 1e2
  
  model_bssm <- bsm_lg(1:10, P1 = diag(1e2,2), sd_slope = 0,
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

test_that("results for multivariate gaussian model are comparable to KFAS",{
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
  kfas_model$H <- structure(c(0.00544500509177812, 0.00437558178720609, 0.00437558178720609, 
    0.00885692410165593), .Dim = c(2L, 2L, 1L))
  kfas_model$R <- structure(c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0152150188066314, 0.0144897116711475
  ), .Dim = c(29L, 1L, 1L), .Dimnames = list(c("log(PetrolPrice).front", 
    "log(kms).front", "log(PetrolPrice).rear", "log(kms).rear", "law.front", 
    "sea_trig1.front", "sea_trig*1.front", "sea_trig2.front", "sea_trig*2.front", 
    "sea_trig3.front", "sea_trig*3.front", "sea_trig4.front", "sea_trig*4.front", 
    "sea_trig5.front", "sea_trig*5.front", "sea_trig6.front", "sea_trig1.rear", 
    "sea_trig*1.rear", "sea_trig2.rear", "sea_trig*2.rear", "sea_trig3.rear", 
    "sea_trig*3.rear", "sea_trig4.rear", "sea_trig*4.rear", "sea_trig5.rear", 
    "sea_trig*5.rear", "sea_trig6.rear", "custom1", "custom2"), NULL, 
    NULL))
  
  bssm_model <- as_bssm(kfas_model)
  expect_equivalent(logLik(kfas_model),logLik(bssm_model))
  expect_equivalent(KFS(kfas_model)$alphahat, smoother(bssm_model)$alphahat)
  
})

test_that("different smoothers give identical results",{
  model_bssm <- bsm_lg(log10(AirPassengers), P1 = diag(1e2,13), sd_slope = 0,
    sd_y = uniform(0.005, 0, 10), sd_level = uniform(0.01, 0, 10), 
    sd_seasonal = uniform(0.005, 0, 1))
  
  expect_error(out_bssm1 <- smoother(model_bssm), NA)
  expect_error(out_bssm2 <- fast_smoother(model_bssm), NA)
  expect_equivalent(out_bssm2, out_bssm1$alphahat)
})


test_that("results for poisson model are comparable to KFAS",{
  library("KFAS")
  set.seed(1)
  model_KFAS <- SSModel(rpois(10, exp(0.2) * (2:11)) ~ SSMtrend(2, Q = list(0.01^2, 0)),
    distribution = "poisson", u = 2:11)
  model_KFAS$P1inf[] <- 0
  diag(model_KFAS$P1) <- 1e2
  
  model_bssm <- bsm_ng(model_KFAS$y, P1 = diag(1e2,2), sd_slope = 0,
    sd_level = 0.01, u = 2:11, distribution = "poisson")
  
  expect_equal(logLik(model_KFAS), logLik(model_bssm, 0))
  out_KFAS <- KFS(model_KFAS, filtering = "state")
  expect_error(out_bssm <- kfilter(model_bssm), NA)
  expect_equivalent(out_KFAS$a, out_bssm$at)
  expect_equivalent(out_KFAS$P, out_bssm$Pt)
  expect_error(out_bssm <- smoother(model_bssm), NA)
  expect_equivalent(out_KFAS$alphahat, out_bssm$alphahat)
  expect_equivalent(out_KFAS$V, out_bssm$Vt)
})


test_that("results for binomial model are comparable to KFAS", {
  library("KFAS")
  set.seed(1)
  model_KFAS <- SSModel(rbinom(10, 2:11, 0.4) ~ SSMtrend(2, Q = list(0.01^2, 0)),
    distribution = "binomial", u = 2:11)
  model_KFAS$P1inf[] <- 0
  diag(model_KFAS$P1) <- 1e2
  
  model_bssm <- bsm_ng(model_KFAS$y, P1 = diag(1e2,2), sd_slope = 0,
    sd_level = 0.01, u = 2:11, distribution = "binomial")
  
  expect_equal(logLik(model_KFAS), logLik(model_bssm, 0))
  out_KFAS <- KFS(model_KFAS, filtering = "state")
  expect_error(out_bssm <- kfilter(model_bssm), NA)
  expect_equivalent(out_KFAS$a, out_bssm$at)
  expect_equivalent(out_KFAS$P, out_bssm$Pt)
  expect_error(out_bssm <- smoother(model_bssm), NA)
  expect_equivalent(out_KFAS$alphahat, out_bssm$alphahat)
  expect_equivalent(out_KFAS$V, out_bssm$Vt)
})
