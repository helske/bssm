context("Test MCMC")


test_that("MCMC results for gaussian model are correct",{
  set.seed(123)
  model_bssm <- bsm(rnorm(10,3), P1 = diag(2,2), sd_slope = 0,
    sd_y = uniform(1, 0, 10), 
    sd_level = uniform(1, 0, 10))
  
  expect_error(out <- run_mcmc(model_bssm, n_iter = 5, nsim = 5), NA)
  
  expect_error(identical(run_mcmc(model_bssm, n_iter = 10, nsim_states = 5, seed = 1)[-8], 
    run_mcmc(model_bssm, n_iter = 10, nsim_states = 5, seed = 1)[-8]), NA)
  
  testvalues <- structure(c(-23.2747514859484, -23.4917121873825, -23.2613263226018), 
    .Dim = c(3L, 1L))
  expect_equivalent(testvalues, out$posterior)
  
  testvalues <- structure(c(0.834199058228936, 0.975961440040881, 0.836899897389733, 
    1.01308074724915, 0.883852023501684, 0.991296915615001), .Dim = c(3L, 2L), 
    .Dimnames = list(NULL, c("sd_y", "sd_level")), mcpar = c(3, 5, 1), class = "mcmc")
  expect_equivalent(testvalues, out$theta)
  
  testvalues <- c(1.97371748764053, 2.79020761994165, 0.100672315364547, 3.63895784712199, 
    -0.64692785732993, 0.0270753716556776)
  expect_equivalent(testvalues, out$alpha[c(1,10,20,25, 31, 60)])
  
})


test_that("MCMC results for Poisson model are correct",{
  set.seed(123)
  model_bssm <- ng_bsm(rpois(10, exp(0.2) * (2:11)), P1 = diag(2, 2), sd_slope = 0,
    sd_level = uniform(2, 0, 10), u = 2:11, distribution = "poisson")
  
  expect_error(out <- run_mcmc(model_bssm, n_iter = 10, nsim_states = 5), NA)
  expect_error(identical(run_mcmc(model_bssm, n_iter = 10, nsim_states = 5)[-8], 
    run_mcmc(model_bssm, n_iter = 10, nsim_states = 5)[-8]), NA)
  
  testvalues <- structure(c(-39.2859389395516, -39.2859389395516, -39.2859389395516, 
    -37.8081239720664, -37.846333182345), .Dim = c(5L, 1L))
  expect_equivalent(testvalues, out$posterior)
  
  testvalues <- structure(c(2.09962020707956, 2.09962020707956, 2.09962020707956, 
    1.49453752077392, 1.63439378916187), .Dim = c(5L, 1L), .Dimnames = list(
      NULL, "sd_level"), mcpar = c(6, 10, 1), class = "mcmc")
  expect_equivalent(testvalues, out$theta)
  
  testvalues <- c(-2.25051173382636, 0.401204374024394, -1.29957037760506, 0.70044534511549, 
    -1.29957037760506, -1.29957037760506)
  expect_equivalent(testvalues, out$alpha[c(1,10,20,25, 31, 60)])
})


test_that("MCMC results for binomial model are correct",{
  set.seed(123)
  model_bssm <- ng_bsm(rbinom(10, 22:31, 0.5), P1 = diag(2, 2), sd_slope = 0,
    sd_level = uniform(2, 0, 10), u = 22:31, distribution = "binomial")
  
  expect_error(out <- run_mcmc(model_bssm, n_iter = 10, nsim_states = 5), NA)
  expect_error(identical(run_mcmc(model_bssm, n_iter = 10, nsim_states = 5)[-8], 
    run_mcmc(model_bssm, n_iter = 10, nsim_states = 5)[-8]), NA)
  
  testvalues <- structure(c(-38.4510899893105, -38.4510899893105, -38.4510899893105, 
    -38.4510899893105, -38.2787954724052), .Dim = c(5L, 1L))
  expect_equivalent(testvalues, out$posterior)
  
  testvalues <- structure(c(2.0499145967585, 2.0499145967585, 2.0499145967585, 
    2.0499145967585, 2.03647937119522), .Dim = c(5L, 1L), .Dimnames = list(
      NULL, "sd_level"), mcpar = c(6, 10, 1), class = "mcmc")
  expect_equivalent(testvalues, out$theta)
  
  testvalues <- c(-0.129365208071896, -0.0764425225893368, 0.754866067120236, 
    -0.0826085226842411, 0.754866067120236, 0.754866067120236)
  expect_equivalent(testvalues, out$alpha[c(1,10,20,25, 31, 60)])
})


test_that("MCMC results for negative binomial model are correct",{
  set.seed(123)
  y <- rnbinom(100, mu = 5, size = 2)
  expect_error(model_bssm <- ng_bsm(y,  sd_level = 0, 
    phi = halfnormal(1, 2),
    distribution = 'negative binomial'), NA)
  
  expect_error(out <- run_mcmc(model_bssm, n_iter = 20, n_burnin = 15, nsim_states = 5), NA)
  expect_error(identical(run_mcmc(model_bssm, n_iter = 10, nsim_states = 5)[-8], 
    run_mcmc(model_bssm, n_iter = 10, nsim_states = 5)[-8]), NA)
  
  testvalues <- structure(c(-263.765826199904, -263.765826199904, -261.661097641368, 
    -262.375150730155, -262.375150730155), .Dim = c(5L, 1L))
  expect_equivalent(testvalues, out$posterior)
  
  testvalues <- structure(c(1.16553263865597, 1.16553263865597, 1.35971289542647, 
    1.28435318292968, 1.28435318292968), .Dim = c(5L, 1L), .Dimnames = list(
      NULL, NULL), mcpar = c(16, 20, 1), class = "mcmc")
  expect_equivalent(testvalues, out$theta)
  
  expect_equivalent(1.49337944854536, out$alpha[1])
})

