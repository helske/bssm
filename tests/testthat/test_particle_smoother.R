
context("Test that particle smoothers work")


test_that("Test that particle_smoother for LGSSM works as Kalman smoother", {
  
  expect_error(model_bsm <- bsm_lg(rep(1, 5), sd_level = 0.05, sd_slope = 0.01, 
    sd_y = 0.01, a1 = c(1, 0), P1 = diag(0.01, 2)), NA)
  expect_equal(smoother(model_bsm)$alphahat, 
    particle_smoother(model_bsm, 1e5, seed = 1)$alphahat, tolerance = 1e-2)
})

test_that("Test that BSF and PSI particle_smoother for LGSSM are with MC error", {
  
  expect_error(model_bsm <- bsm_lg(rep(1, 5), sd_level = 0.05, sd_slope = 0.01, 
    sd_y = 0.01, a1 = c(1, 0), P1 = diag(0.01, 2)), NA)
  expect_error(out1 <- 
      particle_smoother(model_bsm, 1e4, method = "psi", seed = 1), NA)
  expect_error(out2 <- 
      particle_smoother(model_bsm, 1e4, method = "bsf", seed = 1), NA)
  expect_equal(out$alphahat, 
    out2$alphahat, tolerance = 1e-2)
  expect_equal(out$Vt, 
    out2$Vtt, tolerance = 1e-2)
})

test_that("Particle smoother for LGSSM returns finite values", {
  
  expect_error(model_ssm_ulg <- ssm_ulg(y = 1:10, Z = matrix(c(1, 0), 2, 1),
    H = 2, T = array(c(1, 0, 1, 1), c(2, 2, 1)), 
    R = array(diag(2, 2), c(2, 2, 1)), 
    a1 = matrix(0, 2, 1), P1 = diag(2, 2), state_names = c("level", "slope")), 
    NA)
  expect_error(out <- particle_smoother(model_ssm_ulg, 10, seed = 1), 
    NA)
  expect_true(is.finite(sum(out$alpha)))
  expect_true(is.finite(sum(out$alphahat)))
  expect_true(is.finite(sum(out$Vt)))
})

test_that("Particle smoother for poisson bsm_ng returns finite values", {
  
  expect_error(model <- bsm_ng(1:10, sd_level = 2, sd_slope = 2, 
    P1 = diag(2, 2), distribution = "poisson"), NA)
  expect_error(out <- particle_smoother(model, 10, seed = 1), NA)
  
  expect_true(is.finite(sum(out$alpha)))
  expect_true(is.finite(sum(out$alphahat)))
  expect_true(is.finite(sum(out$Vt)))
})

test_that("Particle smoother for binomial bsm_ng returns finite values", {
  
  expect_error(model <- bsm_ng(c(1, 0, 1, 1, 1, 0, 0, 0), sd_level = 2, 
    sd_slope = 2, P1 = diag(2, 2), 
    distribution = "binomial"), NA)
  expect_error(out <- particle_smoother(model, 10, seed = 1), NA)
  
  expect_true(is.finite(sum(out$alpha)))
  expect_true(is.finite(sum(out$alphahat)))
  expect_true(is.finite(sum(out$Vt)))
  
})

test_that("Particle smoother for NB bsm_ng returns finite values", {
  
  expect_error(model <- bsm_ng(c(1, 0, 1, 1, 1, 0, 0, 0), sd_level = 2, 
    sd_slope = 2, P1 = diag(2, 2), 
    distribution = "negative binomial"), NA)
  expect_error(out <- particle_smoother(model, 10, seed = 1), NA)
  
  expect_true(is.finite(sum(out$alpha)))
  expect_true(is.finite(sum(out$alphahat)))
  expect_true(is.finite(sum(out$Vt)))
  
})


test_that("Particle smoother for svm returns finite values", {
  
  data("exchange")
  model <- svm(exchange[1:20], rho = uniform(0.98, -0.999, 0.999), 
    sd_ar = halfnormal(0.2, 5), sigma = halfnormal(0.2, 2))
  
  expect_error(out1 <- 
      particle_smoother(model, 1000, method = "psi", seed = 1), NA)
  expect_error(out2 <- 
      particle_smoother(model, 10000, method = "bsf", seed = 1), NA)
  
  expect_true(is.finite(sum(out1$alpha)))
  expect_true(is.finite(sum(out1$alphahat)))
  expect_true(is.finite(sum(out1$Vt)))
  
  expect_equal(out1$alphahat, out2$alphahat, tol = 1e-2)
})

