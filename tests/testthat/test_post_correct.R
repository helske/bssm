
context("Post-correction and suggest_N")


test_that("Test post correction for AR1 model", {
   set.seed(1)
   n <- 14
   x1 <- sin((2 * pi / 12) * 1:n)
   x2 <- cos((2 * pi / 12) * 1:n)
   alpha <- numeric(n)
   alpha[1] <- 0
   rho <- 0.7
   sigma <- 2
   mu <- 1
   for(i in 2:n) {
     alpha[i] <- rnorm(1, mu * (1 - rho) + rho * alpha[i-1], sigma)
   }
   u <- rpois(n, 50)
   y <- rbinom(n, size = u, plogis(0.5 * x1 + x2 + alpha))
   
   expect_error(model <- ar1_ng(y, distribution = "binomial", 
     rho = uniform(0.5, -1, 1), sigma = gamma(1, 2, 0.001),
     mu = normal(0, 0, 10),
     xreg = cbind(x1,x2), beta = normal(c(0, 0), 0, 5),
     u = u), NA)
   
   
   expect_error(out_approx <- run_mcmc(model, mcmc_type = "approx", 
     local_approx = FALSE, iter = 1000, output_type = "summary"), NA)
   
   expect_error(estN <- suggest_N(model, out_approx, 
     replications = 10, candidates = c(5, 10)), NA)
   
   expect_identical(estN$N, 5)
   
   # Can't really test for correctness with limited time
   expect_error(out_is2 <- post_correct(model, out_approx, particles = estN$N,
     threads = 2), NA)
   expect_lt(sum(out_is2$theta), Inf)
   expect_lt(sum(out_is2$alphahat), Inf)
   expect_lt(sum(out_is2$Vt), Inf)
   expect_lt(max(out_is2$weights), Inf)
   expect_gt(max(out_is2$weights), 0)
})

test_that("Test post correction for non-linear model", {
   skip_on_cran()
   set.seed(1)
   n <- 10
   x <- y <- numeric(n)
   y[1] <- rnorm(1, exp(x[1]), 0.1)
   for(i in 1:(n-1)) {
      x[i+1] <- rnorm(1, sin(x[i]), 0.1)
      y[i+1] <- rnorm(1, exp(x[i+1]), 0.1)
   }
   y[2:3] <- NA
   pntrs <- cpp_example_model("nlg_sin_exp")
   
   expect_error(model <- ssm_nlg(y = y, a1 = pntrs$a1, P1 = pntrs$P1, 
      Z = pntrs$Z_fn, H = pntrs$H_fn, T = pntrs$T_fn, R = pntrs$R_fn, 
      Z_gn = pntrs$Z_gn, T_gn = pntrs$T_gn,
      theta = c(log_H = log(0.1), log_R = log(0.1)), 
      log_prior_pdf = pntrs$log_prior_pdf,
      n_states = 1, n_etas = 1, state_names = "state"), NA)
   
   
   expect_error(out_approx <- run_mcmc(model, mcmc_type = "approx", 
      local_approx = FALSE, iter = 1000, output_type = "full"), NA)
   
   expect_error(estN <- suggest_N(model, out_approx, 
      replications = 10, candidates = c(5, 10)), NA)
   
   expect_identical(estN$N, 5)
   
   # Can't really test for correctness with limited time
   expect_error(out_is2 <- post_correct(model, out_approx, particles = estN$N,
      threads = 2), NA)
   expect_lt(sum(out_is2$theta), Inf)
   expect_lt(sum(out_is2$alphahat), Inf)
   expect_lt(sum(out_is2$Vt), Inf)
   expect_lt(max(out_is2$weights), Inf)
   expect_gt(max(out_is2$weights), 0)
   
})
