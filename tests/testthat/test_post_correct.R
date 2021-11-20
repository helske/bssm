
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
   
   
   expect_error(post_correct(data.frame(1), out_approx, particles = estN$N,
      threads = 2))
   expect_error(post_correct(model, out_approx, particles = estN$N,
      threads = 2, particles = 1e12))
   expect_error(post_correct(model, out_approx, particles = estN$N,
      threads = 2, particles = 10, theta = diag(2)))
   expect_error(post_correct(model, out_approx, particles = estN$N,
      threads = 2, particles = 10, theta = rep(1:6)))
   expect_error(post_correct(model, 1:5, particles = estN$N,
      threads = 2))
   
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
   
   p1 <- 50 # population size at t = 1
   K <- 500 # carrying capacity
   H <- 1 # standard deviation of obs noise
   R_1 <- 0.05 # standard deviation of the noise on logit-growth
   R_2 <- 1 # standard deviation of the noise in population level
   #sample time
   dT <- .1
   
   #observation times
   t <- seq(0.1, 10, dT)
   n <- length(t)
   r <- plogis(cumsum(c(-1.5, rnorm(n - 1, sd = R_1))))
   p <- numeric(n)
   p[1] <- p1
   for(i in 2:n)
      p[i] <- rnorm(1, K * p[i-1] * exp(r[i-1] * dT) / 
            (K + p[i-1] * (exp(r[i-1] * dT) - 1)), R_2)
   # observations
   y <- p + rnorm(n, 0, H)
   y[2:15] <- NA
   pntrs <- cpp_example_model("nlg_growth")
   
   initial_theta <- c(log_H = 0, log_R1 = log(0.05), log_R2 = 0)
   
   # dT, K, a1 and the prior variances of 1st and 2nd state (logit r and and p)
   known_params <- c(dT = dT, K = K, a11 = -1, a12 = 50, P11 = 1, P12 = 100)
   
   expect_error(model <- ssm_nlg(y = y, a1=pntrs$a1, P1 = pntrs$P1, 
      Z = pntrs$Z_fn, H = pntrs$H_fn, T = pntrs$T_fn, R = pntrs$R_fn, 
      Z_gn = pntrs$Z_gn, T_gn = pntrs$T_gn,
      theta = initial_theta, log_prior_pdf = pntrs$log_prior_pdf,
      known_params = known_params, known_tv_params = matrix(1),
      n_states = 2, n_etas = 2, state_names = c("logit_r", "p")), NA)
   
   
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
