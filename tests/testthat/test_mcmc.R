context("Test MCMC")

#' @srrstats {G5.4, G5.4a, G5.4b, G5.4c, G5.5, BS7.0, BS7.1, BS7.2} 
#' Replicate Helske & Vihola (2021).

tol <- 1e-8

test_that("prior and posterior distributions coincide when no data is used", {
  
  skip_on_cran()
  
  set.seed(1)
  n <- 30
  x <- rnorm(n)
  model <- ar1_ng(rep(NA, n), 
    xreg = x,
    distribution = "negative binomial",
    rho = uniform_prior(0.9, 0, 1), 
    sigma = gamma_prior(1, 2, 10),
    mu = normal_prior(1, 0.2, 0.5), 
    phi = gamma_prior(0.4, 2, 1), 
    beta = normal_prior(0.5, 0, 1))
  
  prior_sumr <- as.data.frame(rbind(
    rho = c(0.5, sqrt(1/12)),
    sigma = c(0.2, sqrt(2)/10),
    mu = c(0.2, 0.5),
    phi = c(2, sqrt(2)),
    beta = c(0, 1)))
  
  # approx is enough here, the weights are uniform when there is no data
  fit <- run_mcmc(model, iter = 2e5,burnin = 1e4, mcmc_type = "approx")
  expect_equivalent(prior_sumr, 
    dplyr::arrange(summary(fit)[, c(2, 3)], order(rownames(prior_sumr))), 
    tol = 0.1)
})


test_that("MCMC results from bssm paper are still correct", {
  skip_on_cran()
  
  data(negbin_series)
  bssm_model <- bsm_ng(negbin_series[, 1],
    xreg = negbin_series[, 2],
    beta = normal(0, 0, 10),
    phi = halfnormal(1, 10),
    sd_level = halfnormal(0.1, 1),
    sd_slope = halfnormal(0.01, 0.1),
    a1 = c(0, 0), P1 = diag(c(10, 0.1)^2),
    distribution = "negative binomial")
  
  # run the MCMC
  fit_bssm <- run_mcmc(bssm_model, iter = 6e4, burnin = 1e4,
    particles = 10, seed = 1)
  expect_error(sumr_theta <- summary(fit_bssm)[, "Mean"], NA)
  paper_theta <- c(-0.912, 5.392, 0.092, 0.003)
  expect_equivalent(sumr_theta, paper_theta, tol = 0.01)
  
  expect_error(sumr_alpha <- summary(fit_bssm, 
    variable = "states", times = 200)$Mean, 
    NA)
  paper_alpha <- c(6.962, 0.006)
  expect_equivalent(sumr_alpha, paper_alpha, tol = 0.01)
  
})

#' @srrstats {BS7.3}
test_that("scaling is linear with respect to the length of the time series", {
  skip_on_cran()
  
  set.seed(1)
  n <- 2^14
  
  mu <- 2
  rho <- 0.7
  sd_y <- 0.1
  sigma <- 0.5
  beta <- -1
  x <- rnorm(n)
  z <- y <- numeric(n)
  z[1] <- rnorm(1, mu, sigma / sqrt(1 - rho^2))
  y[1] <- rnorm(1, beta * x[1] + z[1], sd_y)
  for(i in 2:n) {
    z[i] <- rnorm(1, mu * (1 - rho) + rho * z[i - 1], sigma)
    y[i] <- rnorm(1, beta * x[i] + z[i], sd_y)
  }
  # run the MCMC with various number observations
  m <- seq(1000, 10000, by = 3000)
  times <- numeric(length(m))
  for(i in seq_along(m)) {
    model <- ar1_lg(y[1:m[i]],
      xreg = x[1:m[i]],
      rho = uniform(0.5, -1, 1), 
      sigma = halfnormal(1, 10), 
      mu = normal(0, 0, 1), 
      sd_y = halfnormal(1, 10), 
      beta = normal(0, 0, 1))
    
    times[i] <- run_mcmc(model, iter = 2e4)$time[3]
  }
  # standard Kalman filter has complexity of O(n * m) where n is number of time 
  # points and m is the number of states
  expect_equivalent(min(times / m), max(times / m), tol = 0.1)
})


test_that("run_mcmc throws error with improper arguments", {
  set.seed(123)
  model_bssm <- bsm_lg(rnorm(10, 3), P1 = diag(2, 2), sd_slope = 0,
    sd_y = uniform(1, 0, 10), 
    sd_level = uniform(1, 0, 10))
  
  expect_error(mcmc_bsm <- run_mcmc(model_bssm, iter = 50, 
    end_adaptive_phase = 4),
    "Argument 'end_adaptive_phase' should be TRUE or FALSE.")

  out <- run_mcmc(model_bssm, iter = 10, output_type = "theta")
  expect_error(summary(out, return_se = 2), 
    "Argument 'return_se' should be TRUE or FALSE.")
  expect_error(summary(out, variable = "both"),
    "Cannot return summary of states as the MCMC type was not 'full'.")
  
  model_bssm$theta[] <- Inf
  expect_error(run_mcmc(model_bssm, iter = 1e4, particles = 10), 
    "Initial prior probability is not finite.")
})

#' @srrstats {BS2.13}
test_that("MCMC messages can be suppressed", {
  
  model_bssm <- bsm_lg(rnorm(10, 3), P1 = diag(2, 2), sd_slope = 0,
    sd_y = uniform(1, 0, 10), 
    sd_level = uniform(1, 0, 10))
  
  expect_equal(capture_output(run_mcmc(model_bssm, iter = 100, 
    verbose = FALSE)), "")
  expect_equal(capture_output(run_mcmc(model_bssm, iter = 10, 
    verbose = FALSE)), "")
})

test_that("MCMC results for Gaussian model are correct", {
  set.seed(123)
  model_bssm <- bsm_lg(rnorm(10, 3), P1 = diag(2, 2), sd_slope = 0,
    sd_y = uniform(1, 0, 10), 
    sd_level = uniform(1, 0, 10))
  
  expect_error(mcmc_bsm <- run_mcmc(model_bssm, iter = 50, seed = 1), NA)

  expect_equal(
    run_mcmc(model_bssm, iter = 100, seed = 1)[-14], 
    run_mcmc(model_bssm, iter = 100, seed = 1)[-14])
  expect_equal(
    run_mcmc(model_bssm, iter = 100, seed = 1, output_type = "summary")[-15], 
    run_mcmc(model_bssm, iter = 100, seed = 1, output_type = "summary")[-15])
  expect_equal(
    run_mcmc(model_bssm, iter = 100, seed = 1, output_type = "theta")[-13], 
    run_mcmc(model_bssm, iter = 100, seed = 1, output_type = "theta")[-13])
  expect_equal(
    run_mcmc(model_bssm, iter = 100, seed = 1, output_type = "theta")$theta, 
    run_mcmc(model_bssm, iter = 100, seed = 1, output_type = "summary")$theta)
  expect_equal(
    run_mcmc(model_bssm, iter = 100, seed = 1, 
      output_type = "theta")$acceptance_rate, 
    run_mcmc(model_bssm, iter = 100, seed = 1, 
      output_type = "summary")$acceptance_rate)
  
  expect_gt(mcmc_bsm$acceptance_rate, 0)
  expect_gte(min(mcmc_bsm$theta), 0)
  expect_lt(max(mcmc_bsm$theta), Inf)
  expect_true(is.finite(sum(mcmc_bsm$alpha)))
  
  set.seed(1)
  n <- 20
  x1 <- rnorm(n)
  x2 <- rnorm(n)
  b1 <- 1 + cumsum(rnorm(n, sd = 0.5))
  b2 <- 2 + cumsum(rnorm(n, sd = 0.1))
  y <- 1 + b1 * x1 + b2 * x2 + rnorm(n, sd = 0.1)
  
  Z <- rbind(1, x1, x2)
  H <- 0.1
  T <- diag(3)
  R <- diag(c(0, 1, 0.1))
  a1 <- rep(0, 3)
  P1 <- diag(10, 3)
  
  # updates the model given the current values of the parameters
  update_fn <- function(theta) {
    R <- diag(c(0, theta[1], theta[2]))
    dim(R) <- c(3, 3, 1)
    list(R = R, H = theta[3])
  }
  # prior for standard deviations as half-normal(1)
  prior_fn <- function(theta) {
    if(any(theta < 0)) {
      log_p <- -Inf 
    } else {
      log_p <- sum(dnorm(theta, 0, 1, log = TRUE))
    }
    log_p
  }
  
  model <- ssm_ulg(y, Z, H, T, R, a1, P1, 
    init_theta = c(1, 0.1, 0.1), 
    update_fn = update_fn, prior_fn = prior_fn)
  
  expect_error(out <- run_mcmc(model, iter = 50, seed = 1), NA)
  
  expect_gt(out$acceptance_rate, 0)
  expect_gte(min(out$theta), 0)
  expect_lt(max(out$theta), Inf)
  expect_true(is.finite(sum(out$alpha)))
  
  model2 <- ssm_ulg(y, Z, H, T, R, a1, P1)
  expect_error(run_mcmc(model2, iter = 50))
  
  expect_equal(
    run_mcmc(model, iter = 100, seed = 1)[-14], 
    run_mcmc(model, iter = 100, seed = 1)[-14])
  expect_equal(
    run_mcmc(model, iter = 100, seed = 1, output_type = "summary")[-15], 
    run_mcmc(model, iter = 100, seed = 1, output_type = "summary")[-15])
  expect_equal(
    run_mcmc(model, iter = 100, seed = 1, output_type = "theta")[-13], 
    run_mcmc(model, iter = 100, seed = 1, output_type = "theta")[-13])
  expect_equal(
    run_mcmc(model, iter = 100, seed = 1, output_type = "theta")$theta, 
    run_mcmc(model, iter = 100, seed = 1, output_type = "summary")$theta)
  expect_equal(
    run_mcmc(model, iter = 100, seed = 1, 
      output_type = "theta")$acceptance_rate, 
    run_mcmc(model, iter = 100, seed = 1, 
      output_type = "summary")$acceptance_rate)
})

test_that("MCMC for ssm_mng work", {
  
  set.seed(1)
  n <- 20
  x <- cumsum(rnorm(n, sd = 0.5))
  phi <- 2
  y <- cbind(rgamma(n, shape = phi, scale = exp(x) / phi),
    rbinom(n, 1, plogis(x)))
  
  Z <- matrix(1, 2, 1)
  T <- 1
  R <- 0.5
  a1 <- 0
  P1 <- 1
  
  update_fn <- function(theta) {
    list(R = array(theta[1], c(1, 1, 1)), phi = c(theta[2], 1))
  }
  
  prior_fn <- function(theta) {
    ifelse(all(theta > 0), sum(dnorm(theta, 0, 1, log = TRUE)), -Inf)
  }
  
  expect_error(model <- ssm_mng(y, Z, T, R, a1, P1, phi = c(2, 1), 
    init_theta = c(0.5, 2), 
    distribution = c("gamma", "binomial"),
    update_fn = update_fn, prior_fn = prior_fn), NA)
  
  expect_error(run_mcmc(model, iter = 50, 
    local_approx = 4))
  expect_error(run_mcmc(model, iter = 50, 
    particles = 1))
  
  for(type in c("pm", "da", "is1", "is3", "is3", "approx")) {
    for(method in c("psi", "bsf", "spdk")) {
      for(output in c("full", "summary", "theta")) {
        expect_error(
          out <- run_mcmc(model, mcmc_type = type, sampling_method = method,
            output_type = output, iter = 1000, seed = 1, particles = 10), NA)
        expect_equal(sum(is.na(out$theta)), 0)
        expect_equal(sum(is.na(out$alpha)), 0)
        expect_equal(sum(!is.finite(out$theta)), 0)
        expect_equal(sum(!is.finite(out$alpha)), 0)
        expect_equal(sum(!is.finite(out$posterior)), 0)
        expect_gt(out$acceptance_rate, 0)
        expect_gte(min(out$theta), 0)
      }
    }
  }
  
  expect_error(bootstrap_filter(model, 10), NA)
})

test_that("MCMC results with psi-APF for Poisson model are correct", {
  
  set.seed(123)
  model_bssm <- bsm_ng(rpois(10, exp(0.2) * (2:11)), P1 = diag(2, 2), 
    sd_slope = 0, sd_level = uniform(2, 0, 10), u = 2:11, 
    distribution = "poisson")
  
  expect_error(mcmc_poisson <- run_mcmc(model_bssm, mcmc_type = "da", 
    iter = 100, particles = 5, seed = 42), NA)
  
  expect_gt(mcmc_poisson$acceptance_rate, 0)
  expect_gte(min(mcmc_poisson$theta), 0)
  expect_lt(max(mcmc_poisson$theta), Inf)
  expect_true(is.finite(sum(mcmc_poisson$alpha)))
  
  sumr <- expect_error(summary(mcmc_poisson, variable = "both"), NA)
  
  expect_lt(sum(abs(sumr$theta[1, 2:3] - 
      c(0.25892090511681, 0.186796779799571))), 0.5)
  
  states <- expand_sample(mcmc_poisson, variable = "states")
  
  expect_error(expand_sample(mcmc_poisson, variable = "blaablaa"))
  expect_error(expand_sample(mcmc_poisson, variable = "states", by_states = 2))
  expect_error(expand_sample(mcmc_poisson, variable = "states", times = 0))
  expect_error(expand_sample(mcmc_poisson, variable = "states", times = 1:100))
  expect_error(expand_sample(mcmc_poisson, variable = "states", states = 0))
  expect_error(expand_sample(mcmc_poisson, variable = "states", states = "a"))
  expect_error(expand_sample(mcmc_poisson, variable = "states", 
    states = list(4)))
  
  
  expect_equal(sumr$states$Mean[sumr$states$variable == "level"], 
    as.numeric(colMeans(states$level)))
  
  expect_error(as_draws(mcmc_poisson), NA)
  expect_error(d <- as.data.frame(mcmc_poisson, variable = "state"), NA)
  x <- dplyr::pull(dplyr::summarise(
    dplyr::group_by(
      dplyr::filter(d, variable == "level"), time), 
    mean = mean(value)), mean)
  expect_equal(x, as.numeric(colMeans(states$level)))
  
  for(type in c("pm", "da", "is1", "is3", "is3", "approx")) {
    z <- 2*type%in%c("is1", "is3", "is3", "approx")
    expect_equal(
      run_mcmc(model_bssm, mcmc_type = type, iter = 100, seed = 1, 
        particles = 5)[-14 - z], 
      run_mcmc(model_bssm, mcmc_type = type, iter = 100, seed = 1, 
        particles = 5)[-14 - z])
    expect_equal(
      run_mcmc(model_bssm, mcmc_type = type, iter = 100, seed = 1, 
        output_type = "summary", particles = 5)[-15 - z], 
      run_mcmc(model_bssm, mcmc_type = type, iter = 100, seed = 1, 
        output_type = "summary", particles = 5)[-15 - z])
    expect_equal(
      run_mcmc(model_bssm, mcmc_type = type, iter = 100, seed = 1, 
        output_type = "theta", particles = 5)[-13 - z], 
      run_mcmc(model_bssm, mcmc_type = type, iter = 100, seed = 1, 
        output_type = "theta", particles = 5)[-13 - z])
  }
  
  expect_error(expand_sample(run_mcmc(model_bssm, iter = 100, seed = 1, 
    output_type = "theta", mcmc_type = "approx"), variable = "states"))
})



test_that("MCMC using SPDK for Gamma model works", {
  
  set.seed(123)
  n <- 20
  u <- rgamma(n, 3, 1)
  phi <- 5
  x <- cumsum(rnorm(n, 0, 0.5))
  y <- rgamma(n, shape = phi, scale = u * exp(x) / phi)
  model_bssm <- bsm_ng(y, 
    sd_level = gamma(0.1, 2, 10), u = u, phi = gamma(2, 2, 0.1),
    distribution = "gamma", P1 = 2)
  
  expect_error(mcmc_gamma <- run_mcmc(model_bssm, sampling_method = "spdk",
    iter = 1000, particles = 5, seed = 42), NA)
  
  expect_gt(mcmc_gamma$acceptance_rate, 0)
  expect_gte(min(mcmc_gamma$theta), 0)
  expect_lt(max(mcmc_gamma$theta), Inf)
  expect_true(is.finite(sum(mcmc_gamma$alpha)))
  
  expect_lt(sum(abs(summary(mcmc_gamma)$Mean - 
      c(12.353642743311, 0.542149368711246))), 2)
  
})

test_that("MCMC results for SV model using IS-correction are correct", {
  set.seed(123)
  
  expect_error(svm(rnorm(10), rho = uniform(0.95, -0.999, 0.999), 
    sd_ar = halfnormal(1, 5), mu = 4, sigma = halfnormal(1, 2)))
  
  expect_error(model_bssm <- svm(rnorm(10), rho = uniform(0.95, -0.999, 0.999), 
    sd_ar = halfnormal(1, 5), sigma = halfnormal(1, 2)), NA)
  expect_error(logLik(model_bssm, particles = 0, method = "bsf"))
  
  
  expect_equal(run_mcmc(model_bssm, iter = 100, particles = 10,
    mcmc_type = "is1", seed = 1)[-16], 
    run_mcmc(model_bssm, iter = 100, particles = 10, mcmc_type = "is1", 
      seed = 1)[-16])
  
  expect_equal(run_mcmc(model_bssm, iter = 100, particles = 10,
    mcmc_type = "is2", seed = 1)[-16], 
    run_mcmc(model_bssm, iter = 100, particles = 10, mcmc_type = "is2", 
      seed = 1)[-16])
  
  expect_equal(run_mcmc(model_bssm, iter = 100, particles = 10,
    mcmc_type = "is3", seed = 1)[-16], 
    run_mcmc(model_bssm, iter = 100, particles = 10, mcmc_type = "is3", 
      seed = 1)[-16])
  
  expect_equal(run_mcmc(model_bssm, iter = 100, particles = 10,
    mcmc_type = "is2", seed = 1, sampling_mcmc_type = "psi")[-16], 
    run_mcmc(model_bssm, iter = 100, particles = 10, 
      mcmc_type = "is2", seed = 1, sampling_mcmc_type = "psi")[-16])
  
  expect_equal(run_mcmc(model_bssm, iter = 100, particles = 10,
    mcmc_type = "is2", seed = 1, sampling_mcmc_type = "bsf")[-16], 
    run_mcmc(model_bssm, iter = 100, particles = 10, 
      mcmc_type = "is2", seed = 1, sampling_mcmc_type = "bsf")[-16])
  
  expect_equal(run_mcmc(model_bssm, iter = 100, particles = 10,
    mcmc_type = "is2", seed = 1, sampling_mcmc_type = "psi", 
    threads = 2L)[-16], 
    run_mcmc(model_bssm, iter = 100, particles = 10, 
      mcmc_type = "is2", seed = 1, sampling_mcmc_type = "psi", 
      threads = 2L)[-16])
  
  expect_error(mcmc_sv <- run_mcmc(model_bssm, iter = 100, particles = 10,
    mcmc_type = "is2", seed = 1, sampling_mcmc_type = "bsf"), NA)
  
  expect_warning(expand_sample(mcmc_sv))
  expect_error(summary(mcmc_sv, variable = "both"), NA)
  expect_gt(mcmc_sv$acceptance_rate, 0)
  expect_true(is.finite(sum(mcmc_sv$theta)))
  expect_true(is.finite(sum(mcmc_sv$alpha)))
  expect_gte(min(mcmc_sv$weights), 0)
  expect_lt(max(mcmc_sv$weights), Inf)
  
  expect_equal(run_mcmc(model_bssm, iter = 100, particles = 10,
    mcmc_type = "pm", seed = 1, sampling_mcmc_type = "psi",
    output_type = "summary")[-15], 
    run_mcmc(model_bssm, iter = 100, particles = 10, 
      mcmc_type = "pm", seed = 1, sampling_mcmc_type = "psi", 
      output_type = "summary")[-15])
  
  expect_equal(run_mcmc(model_bssm, iter = 100, particles = 10,
    mcmc_type = "is2", seed = 1, sampling_mcmc_type = "psi", 
    output_type = "summary",
    threads = 2L)[-17], 
    run_mcmc(model_bssm, iter = 100, particles = 10, 
      mcmc_type = "is2", seed = 1, sampling_mcmc_type = "psi", 
      output_type = "summary",
      threads = 2L)[-17])
})

test_that("MCMC for nonlinear models work", {
  skip_on_cran()
  set.seed(1)
  n <- 10
  x <- y <- numeric(n)
  y[1] <- rnorm(1, exp(x[1]), 0.1)
  for(i in 1:(n-1)) {
    x[i+1] <- rnorm(1, sin(x[i]), 0.1)
    y[i+1] <- rnorm(1, exp(x[i+1]), 0.1)
  }
  
  pntrs <- cpp_example_model("nlg_sin_exp")
  
  expect_error(model_nlg <- ssm_nlg(y = y, a1 = pntrs$a1, P1 = pntrs$P1, 
    Z = pntrs$Z_fn, H = pntrs$H_fn, T = pntrs$T_fn, R = pntrs$R_fn, 
    Z_gn = pntrs$Z_gn, T_gn = pntrs$T_gn,
    theta = c(log_H = log(0.1), log_R = log(0.1)), 
    log_prior_pdf = pntrs$log_prior_pdf,
    n_states = 1, n_etas = 1, state_names = "state"), NA)
  
  
  for(type in c("pm", "da", "is1", "is3", "is3", "approx", "ekf")) {
    for(method in c("psi", "bsf", "ekf")) {
      for(output in c("full", "summary", "theta")) {
        if(type %in% c("is1", "is2", "is3") && method == "ekf") {
          expect_error(run_mcmc(model_nlg, mcmc_type = type, 
            sampling_method = method, output_type = output, iter = 100, 
            seed = 1, particles = 5))
        } else {
          expect_error(
            run_mcmc(out <- model_nlg, mcmc_type = type, 
              sampling_method = method, output_type = output, iter = 100, 
              seed = 1, particles = 5), NA)
          expect_equal(sum(is.na(out$theta)), 0)
          expect_equal(sum(is.na(out$alpha)), 0)
          expect_equal(sum(!is.finite(out$theta)), 0)
          expect_equal(sum(!is.finite(out$alpha)), 0)
          expect_equal(sum(!is.finite(out$posterior)), 0)
        }
      }
    }
  }
  
  expect_equal(run_mcmc(model_nlg, iter = 100, particles = 10,
    mcmc_type = "is2", seed = 1, sampling_method = "psi", 
    threads = 2L)[-16], 
    run_mcmc(model_nlg, iter = 100, particles = 10, 
      mcmc_type = "is2", seed = 1, sampling_method = "psi", 
      threads = 2L)[-16])
  
  expect_equal(
    run_mcmc(model_nlg, iter = 100, particles = 10,
      mcmc_type = "is1", seed = 1, sampling_method = "psi", 
      output_type = "summary",
      threads = 2L)[-17], 
    run_mcmc(model_nlg, iter = 100, particles = 10, 
      mcmc_type = "is1", seed = 1, sampling_method = "psi", 
      output_type = "summary",
      threads = 2L)[-17])
  
})
