context("Test importance_sample")



test_that("Test that poisson ng_bsm give identical results with ngssm",{
  
  expect_error(model_ngssm <- ngssm(y = 1:10, Z = matrix(c(1, 0), 2, 1),
    T = array(c(1, 0, 1, 1), c(2, 2, 1)), R = array(diag(2, 2), c(2, 2, 1)), 
    a1 = matrix(0, 2, 1), P1 = diag(2, 2), state_names = c("level", "slope"),
    distribution = "poisson"), NA)
  expect_error(sim_ngssm <- importance_sample(model_ngssm, 4, seed = 2), NA)
  expect_error(model_ng_bsm <- ng_bsm(1:10, sd_level = 2, sd_slope = 2, P1 = diag(2, 2), 
    distribution = "poisson"), NA)
  expect_error(sim_ng_bsm <- importance_sample(model_ng_bsm, 4, seed = 2), NA)
  expect_equal(sim_ng_bsm, sim_ngssm)
  testvalues <- structure(c(1.33288345256985, 0.979030809094139, 0.326927291153189, 
    0.954068461233847), .Dim = c(4L, 1L))
  expect_equal(sim_ng_bsm$weights, testvalues)
})

test_that("Test that svm still works",{
  data("exchange")
  model <- svm(exchange, rho = uniform(0.98,-0.999,0.999), 
    sd_ar = halfnormal(0.2, 5), sigma = halfnormal(1, 2))
  
  expect_error(sim <- importance_sample(model, 4, seed = 2), NA)
  testvalues <- structure(c(5.86702179982002, 0.215967278599225, 0.0131080574094302, 
0.666553357641113), .Dim = c(4L, 1L))
  expect_equal(sim$weights, testvalues)
  
})
