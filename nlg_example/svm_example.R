library("bssm")

## Stochastic volatility model
data("exchange")

## pointers for model equation functions
Rcpp::sourceCpp("nlg_example/svm_functions.cpp")
xptrs <- create_xptrs()
# 
# ## test that these work
# Rcpp::sourceCpp("nlg_example/pointer_test.cpp")
# pointer_test_vec(xptrs$Z_fn)
# pointer_test_mat(xptrs$Z_gn)

model <- nlg_ssm(exchange, a1 = xptrs$a1_fn, P1 = xptrs$P1_fn, Z = xptrs$Z_fn, 
  H = xptrs$H_fn, T = xptrs$T_fn, R = xptrs$R_fn, 
  Z_gn = xptrs$Z_gn, T_gn = xptrs$T_gn, theta = c(0, 0.7, 0.5), 
  log_prior_pdf = xptrs$log_prior_pdf, initial_mode = log(pmax(1e-4, exchange^2)))

out <- run_mcmc(model, n_iter = 10, n_burnin=0, delayed_acceptance = FALSE, nsim = 100)

model2 <- svm(exchange, rho = uniform(0.7,-0.9999,0.9999), 
  sd_ar = halfnormal(0.5, 5), mu = normal(0, 0, 5))
out2 <- run_mcmc(model2,  n_iter = 10, n_burnin=0, delayed_acceptance = FALSE, 
  nsim = 100, seed=out$seed)

out$time
out2$time
