# Get MAP estimate of theta
get_map <- function(x) {
  x$theta[which.max(x$posterior),]
}

#' Suggest Number of Particles for \eqn{\psi}-APF Post-correction
#'
#' Function \code{estimate_N} estimates suitable number particles needed for accurate 
#' post-correction of approximate MCMC
#' 
#' Function \code{suggest_N} estimates the standard deviation of the 
#' logarithm of the post-correction weights at approximate MAP of theta, 
#' using various particle sizes and suggest smallest number of particles 
#' which still leads standard deviation less than 1. Similar approach was suggested in
#' the context of pseudo-marginal MCMC by Doucet et al. (2015), but see also 
#' Section 10.3 in Vihola et al (2020).
#' 
#' @param model Model of class \code{nongaussian} or \code{ssm_nlg}.
#' @param mcmc_output An output from \code{run_mcmc} used to compute the MAP estimate of theta. 
#' While the intended use assumes this is from approximate MCMC, it is not actually checked, i.e., 
#' it is also possible to input previous (asymptotically) exact output.
#' @param candidates Vector containing the candidate number of particles to test. Default 
#' is \code{seq(10, 100, by = 10)}. 
#' @param replications How many replications should be used for computing the standard deviations? 
#' Default is 100.
#' @param seed Seed for the random number generator.
#' @return List with suggested number of particles \code{N} and matrix containing 
#' estimated standard deviations of the log-weights and corresponding number of particles.
#' @references A. Doucet, M. K. Pitt, G. Deligiannidis, R. Kohn, 
#' Efficient implementation of Markov chain Monte Carlo when using an unbiased likelihood estimator, 
#' Biometrika, Volume 102, Issue 2, 2015, Pages 295–313, https://doi.org/10.1093/biomet/asu075
#' Vihola, M, Helske, J, Franks, J. Importance sampling type estimators based on approximate marginal Markov chain Monte Carlo. 
#' Scand J Statist. 2020; 1– 38. https://doi.org/10.1111/sjos.12492
#' @export
#' @examples 
#' \dontrun{
#' set.seed(1)
#' n <- 300
#' x1 <- sin((2 * pi / 12) * 1:n)
#' x2 <- cos((2 * pi / 12) * 1:n)
#' alpha <- numeric(n)
#' alpha[1] <- 0
#' rho <- 0.7
#' sigma <- 2
#' mu <- 1
#' for(i in 2:n) {
#'   alpha[i] <- rnorm(1, mu * (1 - rho) + rho * alpha[i-1], sigma)
#' }
#' u <- rpois(n, 50)
#' y <- rbinom(n, size = u, plogis(0.5 * x1 + x2 + alpha))
#' 
#' ts.plot(y / u)
#' 
#' model <- ar1_ng(y, distribution = "binomial", 
#'   rho = uniform(0.5, -1, 1), sigma = gamma(1, 2, 0.001),
#'   mu = normal(0, 0, 10),
#'   xreg = cbind(x1,x2), beta = normal(c(0, 0), 0, 5),
#'   u = u)
#' 
#' out_approx <- run_mcmc(model, mcmc_type = "approx", 
#'  iter = 5000)
#' 
#' estN <- suggest_N(model, out_approx, candidates = seq(10, 50, by = 10))
#' plot(x = estN$results$N, y = estN$results$sd, type = "b")
#' estN$N
#' }
suggest_N <- function(model, mcmc_output, candidates = seq(10, 100, by = 10), 
  replications = 100, seed = sample(.Machine$integer.max, size = 1)) {
  
  if (!inherits(mcmc_output, "mcmc_output")) stop("Object 'mcmc_output' is not valid output from 'run_mcmc'.")
  theta <- get_map(mcmc_output)
  
  if (inherits(model, "nongaussian")) {
    model$distribution <- pmatch(model$distribution,
      c("svm", "poisson", "binomial", "negative binomial", "gamma", "gaussian"), 
      duplicates.ok = TRUE) - 1
    
    out <- suggest_n_nongaussian(model, theta, candidates, replications, seed, model_type(model))
  } else {
    if (inherits(model, "ssm_nlg")) {
      out <- suggest_n_nonlinear(t(model$y), model$Z, model$H, model$T, 
        model$R, model$Z_gn, model$T_gn, model$a1, model$P1, 
        model$theta, model$log_prior_pdf, model$known_params, 
        model$known_tv_params, model$n_states, model$n_etas, 
        as.integer(model$time_varying), 
        theta, candidates, replications, seed)
    } else stop("Function 'suggest_N' is only available for models of class 'nongaussian' and 'nlg_ssm'.")
  }
  list(N = candidates[which(out < 1)[1]], results = data.frame(N = candidates, sd = out))
}
#' Run Post-correction for Approximate MCMC using \eqn{\psi}-APF
#'
#' Function \code{post_correct} updates previously obtained approximate MCMC output 
#' with post-correction weights leading to asymptotically exact weighted posterior, 
#' and returns updated MCMC output where components \code{weights}, \code{posterior}, 
#' \code{alpha}, \code{alphahat}, and \code{Vt} are updated (depending on the original output type).
#' 
#' @param model Model of class \code{nongaussian} or \code{ssm_nlg}.
#' @param mcmc_output An output from \code{run_mcmc} used to compute the MAP estimate of theta. 
#' While the intended use assumes this is from approximate MCMC, it is not actually checked, i.e., 
#' it is also possible to input previous (asymptotically) exact output.
#' @param particles Number of particles for \eqn{\psi}-APF. 
#' @param threads Number of parallel threads.
#' @param is_type Type of IS-correction. Possible choices are 
#'\code{"is3"} for simple importance sampling (weight is computed for each MCMC iteration independently),
#' \code{"is2"} for jump chain importance sampling type weighting (default), or
#' \code{"is1"} for importance sampling type weighting where the number of particles used for
#' weight computations is proportional to the length of the jump chain block.
#' @param seed Seed for the random number generator.
#' @return List with suggested number of particles \code{N} and matrix containing 
#' estimated standard deviations of the log-weights and corresponding number of particles.
#' @references A. Doucet, M. K. Pitt, G. Deligiannidis, R. Kohn, 
#' Efficient implementation of Markov chain Monte Carlo when using an unbiased likelihood estimator, 
#' Biometrika, Volume 102, Issue 2, 2015, Pages 295–313, https://doi.org/10.1093/biomet/asu075
#' Vihola, M, Helske, J, Franks, J. Importance sampling type estimators based on approximate marginal Markov chain Monte Carlo. 
#' Scand J Statist. 2020; 1– 38. https://doi.org/10.1111/sjos.12492
#' @export
#' @examples 
#' \dontrun{
#' set.seed(1)
#' n <- 300
#' x1 <- sin((2 * pi / 12) * 1:n)
#' x2 <- cos((2 * pi / 12) * 1:n)
#' alpha <- numeric(n)
#' alpha[1] <- 0
#' rho <- 0.7
#' sigma <- 2
#' mu <- 1
#' for(i in 2:n) {
#'   alpha[i] <- rnorm(1, mu * (1 - rho) + rho * alpha[i-1], sigma)
#' }
#' u <- rpois(n, 50)
#' y <- rbinom(n, size = u, plogis(0.5 * x1 + x2 + alpha))
#' 
#' ts.plot(y / u)
#' 
#' model <- ar1_ng(y, distribution = "binomial", 
#'   rho = uniform(0.5, -1, 1), sigma = gamma(1, 2, 0.001),
#'   mu = normal(0, 0, 10),
#'   xreg = cbind(x1,x2), beta = normal(c(0, 0), 0, 5),
#'   u = u)
#' 
#' out_approx <- run_mcmc(model, mcmc_type = "approx", 
#'   local_approx = FALSE, iter = 50000)
#' 
#' out_is2 <- post_correct(model, out_approx, particles = 30,
#'   threads = 2)
#' out_is2$time
#' 
#' summary(out_approx, return_se = TRUE)
#' summary(out_is2, return_se = TRUE)
#' 
#' # latent state
#' library("dplyr")
#' library("ggplot2")
#' state_approx <- as.data.frame(out_approx, variable = "states") %>% 
#'   group_by(time) %>%
#'   summarise(mean = mean(value))
#'   
#' state_exact <- as.data.frame(out_is2, variable = "states") %>% 
#'   group_by(time) %>%
#'   summarise(mean = weighted.mean(value, weight))
#' 
#' dplyr::bind_rows(approx = state_approx, 
#'   exact = state_exact, .id = "method") %>%
#'   filter(time > 200) %>%
#' ggplot(aes(time, mean, colour = method)) + 
#'   geom_line() + 
#'   theme_bw()
#'
#' # posterior means
#' p_approx <- predict(out_approx, model, type = "mean", 
#'   nsim = 1000, future = FALSE) %>% 
#'   group_by(time) %>%
#'   summarise(mean = mean(value))
#' p_exact <- predict(out_is2, model, type = "mean", 
#'   nsim = 1000, future = FALSE) %>% 
#'   group_by(time) %>%
#'   summarise(mean = mean(value))
#' 
#' dplyr:: bind_rows(approx = p_approx, 
#'   exact = p_exact, .id = "method") %>%
#'   filter(time > 200) %>%
#' ggplot(aes(time, mean, colour = method)) + 
#'   geom_line() + 
#'   theme_bw() 
#' }
post_correct <- function(model, mcmc_output, particles, threads = 1L, 
  is_type = "is2", seed = sample(.Machine$integer.max, size = 1)) {
  
  if (!inherits(mcmc_output, "mcmc_output")) stop("Object 'mcmc_output' is not valid output from 'run_mcmc'.")
  is_type <- pmatch(match.arg(is_type, paste0("is", 1:3)), paste0("is", 1:3))
  
  a <- proc.time()
  if (inherits(model, "nongaussian")) {
    model$distribution <- pmatch(model$distribution,
      c("svm", "poisson", "binomial", "negative binomial", "gamma", "gaussian"), 
      duplicates.ok = TRUE) - 1
    
    out <- postcorrection_nongaussian(model, model_type(model), 
      mcmc_output$output_type, 
      particles,
      seed, threads, is_type, mcmc_output$counts, t(mcmc_output$theta), mcmc_output$modes)
  } else {
    if (inherits(model, "ssm_nlg")) {
      out <- postcorrection_nonlinear(t(model$y), model$Z, model$H, model$T, 
        model$R, model$Z_gn, model$T_gn, model$a1, model$P1, 
        model$theta, model$log_prior_pdf, model$known_params, 
        model$known_tv_params, model$n_states, model$n_etas, 
        as.integer(model$time_varying),
        mcmc_output$output_type, 
        particles,
        seed, threads, is_type,
        mcmc_output$counts, t(mcmc_output$theta), mcmc_output$modes)
    } else stop("Function 'post_correct' is only available for models of class 'nongaussian' and 'ssm_nlg'.")
  }
  mcmc_output$weights <- out$weights
  mcmc_output$posterior <- mcmc_output$posterior + out$posterior
  if (mcmc_output$output_type == 1) {
    mcmc_output$alpha <- out$alpha
    colnames(mcmc_output$alpha) <- names(model$a1)
  } else {
    if (mcmc_output$output_type == 2) {
      mcmc_output$alphahat <- out$alphahat
      mcmc_output$Vt <- out$Vt
      colnames(mcmc_output$alphahat) <- colnames(mcmc_output$Vt) <- rownames(mcmc_output$Vt) <-
        names(model$a1)
      mcmc_output$alphahat <- ts(mcmc_output$alphahat, start = start(model$y),
        frequency = frequency(model$y))
    }
  }
  mcmc_output$time <- 
    rbind("approx" = mcmc_output$time, "postcorrection" = proc.time() - a)[,1:3]
  mcmc_output$mcmc_type <- paste0("is", is_type)
  mcmc_output$seed <- c(mcmc_output$seed, seed)
  mcmc_output$call <- c(mcmc_output$call, match.call()) 
  mcmc_output
}
