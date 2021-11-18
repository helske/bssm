context("Test importance_sample")

test_that("Test that bssm recovers the parameters of the Seatbelts model", {
  
  #' @srrstats {G5.6, G5.6a, G5.6b, G5.7, G5.9b} Replicate Durbin&Koopman (1997)

  model <- bsm_ng(Seatbelts[, "VanKilled"], distribution = "poisson",
    sd_level = 1, sd_seasonal = 1, xreg = Seatbelts[, "law"],
    beta = normal(0, 0, 1))
  
  obj <- function(theta) {
    model$beta[1] <- theta[1]
    model$R[1, 1, 1] <- theta[2]
    model$R[2, 2, 1] <- theta[3]
    -logLik(model, particles = 0)
  }
  
  fit <- optim(c(0, 0, 0), obj, method = "L-BFGS-B", 
    lower = c(-Inf, 0, 0), upper = c(10, 10, 10))
  
  DK1997 <- c(-0.278, 0.0245, 0) # From Durbin and Koopman (1997)
  expect_equal(fit$par, DK1997, tol = 0.01)
  
  # fixed seed for smooth likelihood optimization (enough only for "spdk")
  fixed_seed <- sample(1:1e6, size = 1)
  # Same but with importance sampling
  obj <- function(theta) {
    model$beta[1] <- theta[1]
    model$R[1, 1, 1] <- theta[2]
    model$R[2, 2, 1] <- theta[3]
    -logLik(model, particles = 10, method = "spdk", seed = fixed_seed)
  }
  
  fit_is <- optim(c(0, 0, 0), obj, method = "L-BFGS-B", 
    lower = c(-Inf, 0, 0), upper = c(10, 10, 10))
  
  # essentially identical results in this case
  expect_equal(fit_is$par, DK1997, tol = 0.01)
})

test_that("Test that poisson bsm_ng give identical results with ssm_ung", {
  
  expect_error(model_ssm_ung <- ssm_ung(y = 1:10, Z = matrix(c(1, 0), 2, 1),
    T = array(c(1, 0, 1, 1), c(2, 2, 1)), R = array(diag(2, 2), c(2, 2, 1)), 
    a1 = matrix(0, 2, 1), P1 = diag(2, 2), state_names = c("level", "slope"),
    distribution = "poisson"), NA)
  expect_error(sim_ssm_ung <- importance_sample(model_ssm_ung, 4, seed = 2), NA)
  expect_error(model_bsm_ng <- bsm_ng(1:10, sd_level = 2, sd_slope = 2, 
    P1 = diag(2, 2), distribution = "poisson"), NA)
  expect_error(sim_bsm_ng <- importance_sample(model_bsm_ng, 4, seed = 2), NA)
  expect_equal(sim_bsm_ng, sim_ssm_ung)
})

test_that("Test that svm still works", {
  data("exchange")
  model <- svm(exchange, rho = uniform(0.98, -0.999, 0.999), 
    sd_ar = halfnormal(0.2, 5), sigma = halfnormal(1, 2))
  
  expect_error(sim <- importance_sample(model, 10, seed = 2), NA)
  
  expect_gte(min(sim$weights), 0)
  expect_lt(max(sim$weights), Inf)
  expect_true(is.finite(sum(sim$states)))
  expect_true(is.finite(sum(sim$weights)))
})

