context("Test MCMC")


test_that("MCMC results for Gaussian model are correct",{
  set.seed(123)
  model_bssm <- bsm(rnorm(10,3), P1 = diag(2,2), sd_slope = 0,
    sd_y = uniform(1, 0, 10), 
    sd_level = uniform(1, 0, 10))
  
  expect_error(out <- run_mcmc(model_bssm, n_iter = 5), NA)
  
  expect_error(identical(run_mcmc(model_bssm, n_iter = 10, seed = 1)[-8], 
    run_mcmc(model_bssm, n_iter = 10, seed = 1)[-8]), NA)
  
  testvalues <- structure(c(-23.1536983852141, -23.3159378063579, -24.2220816776382
  ), .Dim = c(3L, 1L))
  expect_equivalent(testvalues, out$posterior)
  
  testvalues <- structure(c(0.821203802977212, 0.488215820392417, 0.523049327126453, 
    0.745189722391033, 0.759448818738286, 0.522370859586638), .Dim = c(3L, 
      2L), .Dimnames = list(NULL, c("sd_y", "sd_level")))
  expect_equivalent(testvalues, out$theta)
  
  testvalues <- c(1.57047832116798, 1.4058958750444, 0.0673347434044258, 3.13476324356325, 
    0.157161964785528, 0.0178605248343667)
  expect_equivalent(testvalues, out$alpha[c(1,10,20,25, 31, 60)])
  
  expect_equivalent(matrix(1, 3, 1), out$counts)
  
})


test_that("MCMC results for Poisson model are correct",{
  set.seed(123)
  model_bssm <- ng_bsm(rpois(10, exp(0.2) * (2:11)), P1 = diag(2, 2), sd_slope = 0,
    sd_level = uniform(2, 0, 10), u = 2:11, distribution = "poisson")
  
  expect_error(out <- run_mcmc(model_bssm, n_iter = 10, nsim_states = 5), NA)
  expect_error(identical(run_mcmc(model_bssm, n_iter = 10, nsim_states = 5)[-8], 
    run_mcmc(model_bssm, n_iter = 10, nsim_states = 5)[-8]), NA)
  
  testvalues <- structure(c(-37.3203997981812, -36.1959001349018, -36.7410542226913
  ), .Dim = c(3L, 1L))
  expect_equivalent(testvalues, out$posterior)
  
  testvalues <- structure(c(1.499445222853, 1.16262771422103, 1.29967414191533
  ), .Dim = c(3L, 1L), .Dimnames = list(NULL, "sd_level"))
  expect_equivalent(testvalues, out$theta)
  
  testvalues <- c(0.43846059142326, -0.299120896882187, -0.14501280250231, 
    0.432926525017934, 0.482396611722603, 0.136167017762909)
  expect_equivalent(testvalues, out$alpha[c(1,10,20,25, 31, 60)])
})


test_that("MCMC results for binomial model are correct",{
  set.seed(123)
  model_bssm <- ng_bsm(rbinom(10, 22:31, 0.5), P1 = diag(2, 2), sd_slope = 0,
    sd_level = uniform(2, 0, 10), u = 22:31, distribution = "binomial")
  
  expect_error(out <- run_mcmc(model_bssm, n_iter = 10, nsim_states = 5), NA)
  expect_error(identical(run_mcmc(model_bssm, n_iter = 10, nsim_states = 5)[-8], 
    run_mcmc(model_bssm, n_iter = 10, nsim_states = 5)[-8]), NA)
  
  testvalues <- structure(c(-37.1851360231237, -34.2325393384421, 
    -32.0504836724325), .Dim = c(3L, 1L))
  expect_equivalent(testvalues, out$posterior)
  
  testvalues <- structure(c(1.71544296643094, 0.975301410431891, 0.434759249400714
  ), .Dim = c(3L, 1L), .Dimnames = list(NULL, "sd_level"))
  expect_equivalent(testvalues, out$theta)
  
  testvalues <- c(-1.39229506071483, -0.58130522651829, 0.0872358641475177, 
    0.227643509439814, 0.999071279612387, 0.118797362004142)
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
  
  testvalues <- structure(c(-258.904526353877, -258.901913842426, 
    -259.066975182541, -259.128475267148), .Dim = c(4L, 1L))
  expect_equivalent(testvalues, out$posterior)
  
  testvalues <- structure(c(2.1382365032609, 2.15793909737042, 2.49144057609899, 
    2.53825657504107), .Dim = c(4L, 1L), .Dimnames = list(NULL, NULL))
  expect_equivalent(testvalues, out$theta)
  
  expect_equivalent(1.37982138985796, out$alpha[1])
  expect_equivalent(matrix(c(2, 1, 1, 1)), out$counts)
})


test_that("MCMC results for SV model are correct",{
  set.seed(123)
  expect_error(model_bssm <- svm(rnorm(10), rho = uniform(0.95,-0.999,0.999), 
    sd_ar = halfnormal(1, 5), sigma = halfnormal(1, 2)), NA)
  
  expect_error(out <- run_mcmc(model_bssm, n_iter = 20, n_burnin = 15, nsim_states = 5), NA)
  expect_error(identical(run_mcmc(model_bssm, n_iter = 10, nsim_states = 5)[-8], 
    run_mcmc(model_bssm, n_iter = 10, nsim_states = 5)[-8]), NA)
  
  testvalues <- structure(c(-18.2523512011732, -18.1912052301928), .Dim = c(2L, 
    1L))
  expect_equivalent(testvalues, out$posterior)
  expect_equivalent(matrix(2:3, 2, 1), out$counts)
  
  testvalues <- structure(c(0.814869516598186, 0.779380668691182, 
    0.64925012566082, 0.487178396827435, 1.24731145087978, 1.35178973334319), 
    .Dim = 2:3, .Dimnames = list(NULL, c("rho", "sd_ar", "sigma")))
  expect_equivalent(testvalues, out$theta)
  
  testvalues <- c(-1.55093738190063, -0.526794309202934, -1.13699648749797)
  expect_equivalent(testvalues, out$alpha[c(1,10, 20)])
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
  
  testvalues <- structure(c(-19.5893006950704, -18.7083804690955, -18.6355215102079, 
    -17.5386138120297), .Dim = c(4L, 1L))
  expect_equivalent(testvalues, out$posterior)
  
  testvalues <- structure(c(0.696842492064601, 0.628422092207903, 0.381007029308564, 
    0.303730996699196, 1.46703192631662, 1.56221790236331, 1.41783102796158, 
    1.23078100977374, 0.541259456272363, 0.563511818877976, 0.624991330233118, 
    0.921759773813795), .Dim = c(4L, 3L), .Dimnames = list(NULL, 
      c("rho", "sd_ar", "sigma")))
  expect_equivalent(testvalues, out$theta)
  
  testvalues <- c(1.65781043896967, 1.13956749618722, -0.474485040911289)
  expect_equivalent(testvalues, out$alpha[c(1,10, 20)])
})

