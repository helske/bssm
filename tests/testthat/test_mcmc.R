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


test_that("MCMC results for SV model are correct",{
  set.seed(123)
  expect_error(model_bssm <- svm(rnorm(10), rho = uniform(0.95,-0.999,0.999), 
    sd_ar = halfnormal(1, 5), sigma = halfnormal(1, 2)), NA)
  
  expect_error(out <- run_mcmc(model_bssm, n_iter = 20, n_burnin = 15, nsim_states = 5), NA)
  expect_error(identical(run_mcmc(model_bssm, n_iter = 10, nsim_states = 5)[-8], 
    run_mcmc(model_bssm, n_iter = 10, nsim_states = 5)[-8]), NA)
  
  testvalues <- structure(c(-17.2785499634381, -17.1522385567832, -17.2408376389957, 
    -17.2408376389957, -17.2529363969), .Dim = c(5L, 1L))
  expect_equivalent(testvalues, out$posterior)
  
  testvalues <- structure(c(0.611645727466498, 0.784126255249186, 0.739990910881518, 
    0.739990910881518, 0.538608198317944, 0.132471223920487, 0.260180586666946, 
    0.305401598068018, 0.305401598068018, 0.449188287046116, 0.732778575928857, 
    0.80046492329671, 0.817714439472359, 0.817714439472359, 1.02447292408947
  ), .Dim = c(5L, 3L), .Dimnames = list(NULL, c("rho", "sd_ar", 
    "sigma")), mcpar = c(16, 20, 1), class = "mcmc")
  expect_equivalent(testvalues, out$theta)
  
  testvalues <- c(-0.0830748149852014, -0.198578873285639, 0.178440933808639, 
    0.0256754390526584)
  expect_equivalent(testvalues, out$alpha[c(1,10, 20, 50)])
})


test_that("MCMC results for SV model using IS-correction are correct",{
  set.seed(123)
  expect_error(model_bssm <- svm(rnorm(10), rho = uniform(0.95,-0.999,0.999), 
    sd_ar = halfnormal(1, 5), sigma = halfnormal(1, 2)), NA)
  
  expect_error(out <- run_mcmc(model_bssm, n_iter = 20, n_burnin = 15, nsim_states = 5,
    method = "isc"), NA)
  expect_error(identical(run_mcmc(model_bssm, n_iter = 100, nsim_states = 10,
    method = "isc")[-8], 
    run_mcmc(model_bssm, n_iter = 100, nsim_states = 10, method = "isc")[-8]), NA)
  expect_error(identical(run_mcmc(model_bssm, n_iter = 100, nsim_states = 10,
    method = "isc", simulation_method = "psi")[-8], 
    run_mcmc(model_bssm, n_iter = 100, nsim_states = 10, method = "isc",
      simulation_method = "psi")[-8]), NA)
  expect_error(identical(run_mcmc(model_bssm, n_iter = 100, nsim_states = 10,
    method = "isc", simulation_method = "psi")[-8], 
    run_mcmc(model_bssm, n_iter = 100, nsim_states = 10, method = "isc", 
      simulation_method = "psi")[-8]), NA)
  testvalues <- structure(c(-19.0505020531966, -19.3483555773888, -19.2742098743496, 
-17.9587002271749), .Dim = c(4L, 1L))
  expect_equivalent(testvalues, out$posterior)
  
  testvalues <- structure(c(0.741988771923731, 0.697340092900775, 0.628452176724537, 
0.379346527686448, 1.19914545826558, 1.46878142788352, 1.56475733617601, 
1.41912717643135, 1.02106720262573, 0.5352855667731, 0.556124259401741, 
0.617628937905398), .Dim = c(4L, 3L), .Dimnames = list(NULL, 
    c("rho", "sd_ar", "sigma")))
  expect_equivalent(testvalues, out$theta)
  
  testvalues <- c(-0.357482373598444, 0.239536835878186, 1.39616333824964)
  expect_equivalent(testvalues, out$alpha[c(1,10, 20)])
})

