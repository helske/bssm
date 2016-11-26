context("Test MCMC")


test_that("MCMC results for Gaussian model are correct",{
  set.seed(123)
  model_bssm <- bsm(rnorm(10,3), P1 = diag(2,2), sd_slope = 0,
    sd_y = uniform(1, 0, 10), 
    sd_level = uniform(1, 0, 10))
  
  expect_error(out <- run_mcmc(model_bssm, n_iter = 5), NA)
  
  expect_error(identical(run_mcmc(model_bssm, n_iter = 10, seed = 1)[-8], 
    run_mcmc(model_bssm, n_iter = 10, seed = 1)[-8]), NA)
  
  testvalues <- structure(c(-23.2747514859484, -23.4917121873825, -23.2613263226018
  ), .Dim = c(3L, 1L))
  expect_equivalent(testvalues, out$posterior)
  
  testvalues <- structure(c(0.834199058228936, 0.975961440040881, 0.836899897389733, 
    1.01308074724915, 0.883852023501684, 0.991296915615001), .Dim = c(3L, 
      2L), .Dimnames = list(NULL, c("sd_y", "sd_level")), mcpar = c(3, 
        5, 1), class = "mcmc")
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
  
  testvalues <- structure(c(-39.3880935409099, -37.3318580117567, -37.6177815203621, 
    -38.8795560250992, -38.8795560250992), .Dim = c(5L, 1L))
  expect_equivalent(testvalues, out$posterior)
  
  testvalues <- structure(c(1.94161122336829, 1.45629893808125, 1.54833882701089, 
    1.86010843337335, 1.86010843337335), .Dim = c(5L, 1L), .Dimnames = list(
      NULL, "sd_level"), mcpar = c(6, 10, 1), class = "mcmc")
  expect_equivalent(testvalues, out$theta)
  
  testvalues <- c(-0.179742059090531, 0.190378831062351, -0.0451489240338361, 
    0.582554939438518, -0.0149794944120587, -0.42743617132243)
  expect_equivalent(testvalues, out$alpha[c(1,10,20,25, 31, 60)])
})


test_that("MCMC results for binomial model are correct",{
  set.seed(123)
  model_bssm <- ng_bsm(rbinom(10, 22:31, 0.5), P1 = diag(2, 2), sd_slope = 0,
    sd_level = uniform(2, 0, 10), u = 22:31, distribution = "binomial")
  
  expect_error(out <- run_mcmc(model_bssm, n_iter = 10, nsim_states = 5), NA)
  expect_error(identical(run_mcmc(model_bssm, n_iter = 10, nsim_states = 5)[-8], 
    run_mcmc(model_bssm, n_iter = 10, nsim_states = 5)[-8]), NA)
  
  testvalues <- structure(c(-37.5274138423666, -37.5274138423666, -37.5274138423666, 
    -37.5898467281197, -37.5898467281197), .Dim = c(5L, 1L))
  expect_equivalent(testvalues, out$posterior)
  
  testvalues <- structure(c(1.79609530935165, 1.79609530935165, 1.79609530935165, 
    1.80278017969165, 1.80278017969165), .Dim = c(5L, 1L), .Dimnames = list(
      NULL, "sd_level"), mcpar = c(6, 10, 1), class = "mcmc")
  expect_equivalent(testvalues, out$theta)
  
  testvalues <- c(-0.467178624242946, -0.216527057264181, 0.977319983890701, 
    1.04226099911296, 0.977319983890701, 0.977319983890701)
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
  
  testvalues <- structure(c(-259.408148930098, -260.865064698319, -261.14000172789, 
    -261.14000172789, -262.097829940278), .Dim = c(5L, 1L))
  expect_equivalent(testvalues, out$posterior)
  
  testvalues <- structure(c(1.78355552962326, 1.4643447573198, 1.4252130704839, 
    1.4252130704839, 1.3120054169815), .Dim = c(5L, 1L), .Dimnames = list(
      NULL, NULL), mcpar = c(16, 20, 1), class = "mcmc")
  expect_equivalent(testvalues, out$theta)
  
  expect_equivalent(1.52626449167856, out$alpha[1])
})


test_that("MCMC results for SV model are correct",{
  set.seed(123)
  expect_error(model_bssm <- svm(rnorm(10), rho = uniform(0.95,-0.999,0.999), 
    sd_ar = halfnormal(1, 5), sigma = halfnormal(1, 2)), NA)
  
  expect_error(out <- run_mcmc(model_bssm, n_iter = 20, n_burnin = 15, nsim_states = 5), NA)
  expect_error(identical(run_mcmc(model_bssm, n_iter = 10, nsim_states = 5)[-8], 
    run_mcmc(model_bssm, n_iter = 10, nsim_states = 5)[-8]), NA)
  
  testvalues <- structure(c(-18.644214166485, -18.644214166485, -18.644214166485, 
    -18.14059171039, -17.7427882105074), .Dim = c(5L, 1L))
  expect_equivalent(testvalues, out$posterior)
  
  testvalues <- structure(c(0.881217648431571, 0.881217648431571, 0.881217648431571, 
    0.75116941472834, 0.684070011216901, 0.463435906249714, 0.463435906249714, 
    0.463435906249714, 0.549836717261301, 0.353421831153397, 1.56694018288237, 
    1.56694018288237, 1.56694018288237, 1.3097950653206, 1.23093038495726
  ), .Dim = c(5L, 3L), .Dimnames = list(NULL, c("rho", "sd_ar", 
    "sigma")), mcpar = c(16, 20, 1), class = "mcmc")
  expect_equivalent(testvalues, out$theta)
  
  testvalues <- c(-0.857317424752647, -0.53536238357548, -0.53536238357548, -0.0228423519266524)
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

