## simple linear-Gaussian random walk model
set.seed(1)
y <- matrix(rnorm(100, cumsum(rnorm(100))), ncol = 1)
library("bssm")

## pointers for model equation functions
Rcpp::sourceCpp("nlg_example/model_functions.cpp")
xptrs <- create_xptrs()

## test that these work
Rcpp::sourceCpp("nlg_example/pointer_test.cpp")
pointer_test_vec(xptrs$Z_fn)
pointer_test_mat(xptrs$Z_gn)

model <- nlg_ssm(y = y, a1 = 0, P1 = matrix(1), Z = xptrs$Z_fn, 
  H = xptrs$H_fn, T = xptrs$T_fn, R = xptrs$R_fn, 
  Z_gn = xptrs$Z_gn, T_gn = xptrs$T_gn, theta = c(1, 1), 
  log_prior_pdf = xptrs$log_prior_pdf)

model$initial_mode <- matrix(0, 100, 1)
out <- run_mcmc(model, n_iter = 1e4, delayed_acceptance = FALSE, nsim = 10)

model2 <- bsm(y, sd_y = uniform(1, 0, 1000), sd_level = uniform(1, 0, 1000))
out2 <- run_mcmc(model2, n_iter = 1e4, delayed_acceptance = FALSE, sim_states = TRUE,seed=out$seed)
