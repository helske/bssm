# context("Test particle_simulate")
# 
# test_that("Test that bsm give identical results with gssm",{
#   
#   expect_error(model_gssm <- gssm(y = 1:10, Z = matrix(c(1, 0), 2, 1), H = 2, 
#     T = array(c(1, 0, 1, 1), c(2, 2, 1)), R = array(diag(2, 2), c(2, 2, 1)), 
#     a1 = matrix(0, 2, 1), P1 = diag(2, 2), state_names = c("level", "slope")), NA)
#   expect_error(sim_gssm <- particle_simulate(model_gssm, 2, seed = 2), NA)
#   expect_equal(sim_gssm$logLik, -36.6658020894499)
#   testvalues <- c(-0.0319113702364748, 2.01341894343523, 0.936842667647643, -1.4106951470175)
#   expect_equal(sim_gssm$alpha[c(1, 2, 11, 20)], testvalues)
#   expect_error(model_bsm <- bsm(1:10, sd_level = 2, sd_slope = 2, sd_y = 2, 
#     P1 = diag(2, 2)), NA)
#   expect_error(sim_bsm <- particle_simulate(model_bsm, 2, seed = 2), NA)
#   expect_equal(sim_bsm, sim_gssm)
# })
# 
# 
# test_that("Test that poisson ng_bsm give identical results with ngssm",{
#   
#   expect_error(model_ngssm <- ngssm(y = 1:10, Z = matrix(c(1, 0), 2, 1),
#     T = array(c(1, 0, 1, 1), c(2, 2, 1)), R = array(diag(2, 2), c(2, 2, 1)), 
#     a1 = matrix(0, 2, 1), P1 = diag(2, 2), state_names = c("level", "slope"),
#     distribution = "poisson"), NA)
#   expect_error(sim_ngssm <- particle_simulate(model_ngssm, 100, seed = 2), NA)
#   expect_equal(sim_ngssm$logLik, -36.4165815198814)
#   expect_error(model_ng_bsm <- ng_bsm(1:10, sd_level = 2, sd_slope = 2, P1 = diag(2, 2), 
#     distribution = "poisson"), NA)
#   expect_error(sim_ng_bsm <- particle_simulate(model_ng_bsm, 100, seed = 2), NA)
#   expect_equal(sim_ng_bsm, sim_ngssm)
# })
# 
# test_that("Test that binomial ng_bsm give identical results with ngssm",{
#   
#   expect_error(model_ngssm <- ngssm(y = c(1,0,1,1,1,0,0,0), Z = matrix(c(1, 0), 2, 1),
#     T = array(c(1, 0, 1, 1), c(2, 2, 1)), R = array(diag(2, 2), c(2, 2, 1)), 
#     a1 = matrix(0, 2, 1), P1 = diag(2, 2), state_names = c("level", "slope"),
#     distribution = "binomial"), NA)
#   expect_error(sim_ngssm <- particle_simulate(model_ngssm, 100, seed = 2), NA)
#   expect_equal(sim_ngssm$logLik, -6.7024830631489)
#   expect_error(model_ng_bsm <- ng_bsm(c(1,0,1,1,1,0,0,0), sd_level = 2, sd_slope = 2, P1 = diag(2, 2), 
#     distribution = "binomial"), NA)
#   expect_error(sim_ng_bsm <- particle_simulate(model_ng_bsm, 100, seed = 2), NA)
#   expect_equal(sim_ng_bsm, sim_ngssm)
# })
# 
# 
# test_that("Test that negative binomial ng_bsm give identical results with ngssm",{
#   
#   expect_error(model_ngssm <- ngssm(y = c(1,0,1,1,1,0,0,0), Z = matrix(c(1, 0), 2, 1),
#     T = array(c(1, 0, 1, 1), c(2, 2, 1)), R = array(diag(2, 2), c(2, 2, 1)), 
#     a1 = matrix(0, 2, 1), P1 = diag(2, 2), state_names = c("level", "slope"),
#     distribution = "negative binomial", phi = 0.1, u = 2), NA)
#   expect_error(sim_ngssm <- particle_simulate(model_ngssm, 100, seed = 2), NA)
#   expect_equal(sim_ngssm$logLik, -13.6549970967291)
#   expect_error(model_ng_bsm <- ng_bsm(c(1,0,1,1,1,0,0,0), sd_level = 2, sd_slope = 2, 
#     P1 = diag(2, 2), distribution = "negative binomial", phi = 0.1, u = 2), NA)
#   expect_error(sim_ng_bsm <- particle_simulate(model_ng_bsm, 100, seed = 2), NA)
#   expect_equal(sim_ng_bsm, sim_ngssm)
# })
# 
# 
# test_that("Test that svm still works",{
#   data("exchange")
#   model <- svm(exchange, rho = uniform(0.98,-0.999,0.999), 
#     sd_ar = halfnormal(0.2, 5), sigma = halfnormal(1, 2))
#   
#   expect_error(sim <- particle_simulate(model, 10, seed = 2), NA)
#   expect_equal(sim$logLik, -933.056099469693)
#   
# })
# 
# context("Test that particle_filter works")
# 
# test_that("Test that bsm give identical results with gssm",{
#   
#   expect_error(model_gssm <- gssm(y = 1:10, Z = matrix(c(1, 0), 2, 1), H = 2, 
#     T = array(c(1, 0, 1, 1), c(2, 2, 1)), R = array(diag(2, 2), c(2, 2, 1)), 
#     a1 = matrix(0, 2, 1), P1 = diag(2, 2), state_names = c("level", "slope")), NA)
#   expect_error(sim_gssm <- particle_filter(model_gssm, 2, seed = 2), NA)
#   expect_equal(sim_gssm$logLik, -36.6658020894499)
#   
#   testvalues <- c(-0.0319113702364748, 5.4722485818937, 0.936842667647643, 2.19849347885756)
#   expect_equal(sim_gssm$alpha[c(1, 2, 11, 20)], testvalues)
# 
#   
#   expect_error(model_bsm <- bsm(1:10, sd_level = 2, sd_slope = 2, sd_y = 2, 
#     P1 = diag(2, 2)), NA)
#   expect_error(sim_bsm <- particle_filter(model_bsm, 2, seed = 2), NA)
#   expect_equal(sim_bsm, sim_gssm)
# })
# 
# test_that("Test that poisson ng_bsm give identical results with ngssm",{
#   
#   expect_error(model_ng_bsm <- ng_bsm(1:10, sd_level = 2, sd_slope = 2, P1 = diag(2, 2), 
#     distribution = "poisson"), NA)
#   expect_error(sim_ng_bsm <- particle_filter(model_ng_bsm, 100, seed = 2), NA)
#  expect_equal(sim_ng_bsm$logLik, -36.4165815198814)
#  
# })
# 
# test_that("Test that binomial ng_bsm give identical results with ngssm",{
#   
#   expect_error(model <- ng_bsm(c(1,0,1,1,1,0,0,0), sd_level = 2, sd_slope = 2, P1 = diag(2, 2), 
#     distribution = "binomial"), NA)
#   expect_error(sim <- particle_simulate(model, 100, seed = 2), NA)
#   expect_equal(sim$logLik, -6.7024830631489)
#   
# })
# 
# 
# test_that("Test that negative binomial ng_bsm give identical results with ngssm",{
#   expect_error(model <- ng_bsm(c(1,0,1,1,1,0,0,0), sd_level = 2, sd_slope = 2, 
#     P1 = diag(2, 2), distribution = "negative binomial", phi = 0.1, u = 2), NA)
#   expect_error(sim <- particle_simulate(model, 100, seed = 2), NA)
#   expect_equal(sim$logLik, -13.6549970967291)
# })
# 
# 
# test_that("Test that still svm works",{
#   data("exchange")
#   model <- svm(exchange, rho = uniform(0.98,-0.999,0.999), 
#     sd_ar = halfnormal(0.2, 5), sigma = halfnormal(1, 2))
#   
#   expect_error(sim <- particle_filter(model, 10, seed = 2), NA)
#   expect_equal(sim$logLik, -933.056099469693)
#     expect_error(sim <- particle_filter(model, 10, seed = 2, filter_type = "psi"), NA)
#   expect_equal(sim$logLik, -928.645298308319)
#     expect_error(sim <- particle_smoother(model, 10, seed = 2), NA)
# expect_equal(sim$alpha[c(1,15,60,100)], c(-0.697336434621932, -0.228311104465841, 
#   -0.463805803000575, -1.57563774953562))
#    expect_error(sim <- particle_smoother(model, 10, seed = 2, filter_type = "psi"), NA)
#   expect_equal(sim$alpha[c(1,15,60,100)], c(0.0491680342406114, -1.23060147630784, 
#     -1.10951398031313, -1.38148633008125))
# 
# })
# 
