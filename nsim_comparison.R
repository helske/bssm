library("bssm")

set.seed(42)
y <- rnorm(100, cumsum(rnorm(100, sd = 0.1)), 1)
model <- bsm(y, sd_level = 0.1, sd_y = 1, slope = FALSE)

o1_1e4 <- run_mcmc(model, n_iter = 2e4,  nsim = 1, log_space = TRUE)
o10_1e4 <- run_mcmc(model, n_iter = 2e4, nsim = 10, log_space = TRUE)
o100_1e4 <- run_mcmc(model, n_iter = 2e4, nsim = 100, log_space = TRUE)

o1_1e5 <- run_mcmc(model, n_iter = 2e5,  nsim = 1, log_space = TRUE)
o10_1e5 <- run_mcmc(model, n_iter = 2e5, nsim = 10, log_space = TRUE)

coda::effectiveSize(o1_1e4$alpha[100,1,]) #1233.748 
coda::effectiveSize(o10_1e4$alpha[100,1,]) #12417.83 
coda::effectiveSize(o100_1e4$alpha[100,1,]) #420405.5
coda::effectiveSize(o1_1e5$alpha[100,1,]) #11973.6
coda::effectiveSize(o10_1e5$alpha[100,1,]) #11973.6
