
context("Post-correction and suggest_N")


test_that("Test that sim_smoother for LGSSM works as Kalman smoother", {
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

