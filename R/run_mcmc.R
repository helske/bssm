#' Bayesian Inference of State Space Models
#'
#' Adaptive Markov chain Monte Carlo simulation of state space models using
#' Robust Adaptive Metropolis algorithm by Vihola (2012). 
#' See specific methods for various model types for details.
#'
#' @importFrom stats tsp
#' @param model State space model model of \code{bssm} package.
#' @param iter Number of MCMC iterations.
#' @param ... Parameters to specific methods. See \code{\link{run_mcmc.gaussian}},
#' \code{\link{run_mcmc.nongaussian}}, \code{\link{run_mcmc.ssm_nlg}}, 
#' and \code{\link{run_mcmc.ssm_sde}} for details.
#' @export
#' @rdname run_mcmc
#' @references Matti Vihola (2012). "Robust adaptive Metropolis algorithm with
#' coerced acceptance rate". Statistics and Computing, Volume 22, Issue 5,
#' pages 997--1008.
run_mcmc <- function(model, iter, ...) {
  UseMethod("run_mcmc", model)
}
#' Bayesian Inference of Linear-Gaussian State Space Models
#'
#' @method run_mcmc gaussian
#' @rdname run_mcmc_g
#' @param model Model model.
#' @param iter Number of MCMC iterations.
#' @param output_type Type of output. Default is \code{"full"}, which returns
#' samples from the posterior \eqn{p(\alpha, \theta)}. Option \code{"summary"} does not simulate
#' states directly but computes the posterior means and variances of states using
#' fast Kalman smoothing. This is slightly faster, more memory efficient and
#' more accurate than calculations based on simulation smoother. Using option \code{"theta"} will only
#' return samples from the marginal posterior of the hyperparameters \eqn{\theta}.
#' @param burnin Length of the burn-in period which is disregarded from the
#' results. Defaults to \code{iter / 2}. Note that all MCMC algorithms of \code{bssm}
#'  used adaptive MCMC during the burn-in period in order to find good proposal.
#' @param thin Thinning rate. All MCMC algorithms in \code{bssm} use the jump chain
#' representation, and the thinning is applied to these blocks.
#' Defaults to 1.
#' @param gamma Tuning parameter for the adaptation of RAM algorithm. Must be
#' between 0 and 1 (not checked).
#' @param target_acceptance Target acceptance ratio for RAM. Defaults to 0.234.
#' @param S Initial value for the lower triangular matrix of RAM
#' algorithm, so that the covariance matrix of the Gaussian proposal
#' distribution is \eqn{SS'}. Note that for some parameters 
#' (currently the standard deviation and dispersion parameters of bsm_lg models) the sampling
#' is done for transformed parameters with internal_theta = log(theta).
#' @param end_adaptive_phase If \code{TRUE}, S is held fixed after the burnin period. Default is \code{FALSE}.
#' @param threads Number of threads for state simulation.
#' @param seed Seed for the random number generator.
#' @param ... Ignored.
#' @references 
#' Vihola, M, Helske, J, Franks, J. Importance sampling type estimators based on approximate marginal Markov chain Monte Carlo. 
#' Scand J Statist. 2020; 1– 38. https://doi.org/10.1111/sjos.12492
#' @export
#' @examples 
#' model <- ar1_lg(LakeHuron, rho = uniform(0.5,-1,1), 
#'   sigma = halfnormal(1, 10), mu = normal(500, 500, 500), 
#'   sd_y = halfnormal(1, 10))
#' 
#' mcmc_results <- run_mcmc(model, iter = 2e4)
#' summary(mcmc_results, return_se = TRUE)
#' 
#' require("dplyr")
#' sumr <- as.data.frame(mcmc_results, variable = "states") %>%
#'   group_by(time) %>%
#'   summarise(mean = mean(value), 
#'     lwr = quantile(value, 0.025), 
#'     upr = quantile(value, 0.975))
#' require("ggplot2")
#' sumr %>% ggplot(aes(time, mean)) + 
#'   geom_ribbon(aes(ymin = lwr, ymax = upr),alpha=0.25) + 
#'   geom_line() + theme_bw() +
#'   geom_point(data = data.frame(mean = LakeHuron, time = time(LakeHuron)),
#'     col = 2)
run_mcmc.gaussian <- function(model, iter, output_type = "full",
  burnin = floor(iter / 2), thin = 1, gamma = 2/3,
  target_acceptance = 0.234, S, end_adaptive_phase = FALSE, threads = 1,
  seed = sample(.Machine$integer.max, size = 1), ...) {
  
  
  if(length(model$theta) == 0) stop("No unknown parameters ('model$theta' has length of zero).")
  a <- proc.time()
  
  check_target(target_acceptance)
  
  output_type <- pmatch(output_type, c("full", "summary", "theta"))
  
  if (inherits(model, "bsm_lg")) {
    names_ind <- !model$fixed & c(TRUE, TRUE, model$slope, model$seasonal)
    model$theta[c("sd_y", "sd_level", "sd_slope", "sd_seasonal")[names_ind]] <- 
      log(pmax(1e-8, model$theta[c("sd_y", "sd_level", "sd_slope", "sd_seasonal")[names_ind]]))
  }
  
  if (missing(S)) {
    S <- diag(0.1 * pmax(0.1, abs(model$theta)), length(model$theta))
  }
  
  
  out <- gaussian_mcmc(model, output_type,
    iter, burnin, thin, gamma, target_acceptance, S, seed,
    end_adaptive_phase, threads, model_type(model))
  
  if (output_type == 1) {
    colnames(out$alpha) <- names(model$a1)
  } else {
    if (output_type == 2) {
      colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <-
        names(model$a1)
      out$alphahat <- ts(out$alphahat, start = start(model$y),
        frequency = frequency(model$y))
    }
  }
  
  colnames(out$theta) <- rownames(out$S) <- colnames(out$S) <- names(model$theta)
  
  if (inherits(model, "bsm_lg")) {
    out$theta[, c("sd_y", "sd_level", "sd_slope", "sd_seasonal")[names_ind]] <- 
      exp(out$theta[, c("sd_y", "sd_level", "sd_slope", "sd_seasonal")[names_ind]])
  }
  out$call <- match.call()
  out$seed <- seed
  out$iter <- iter
  out$burnin <- burnin
  out$thin <- thin
  out$mcmc_type <- "gaussian_mcmc"
  out$output_type <- output_type
  out$time <- proc.time() - a
  class(out) <- "mcmc_output"
  attr(out, "model_type") <- class(model)[1]
  attr(out, "ts") <- 
    list(start = start(model$y), end = end(model$y), frequency=frequency(model$y))
  out
}


#' Bayesian Inference of Non-Gaussian State Space Models 
#'
#' Methods for posterior inference of states and parameters.
#'
#' @method run_mcmc nongaussian
#' @rdname run_mcmc_ng
#' @export
#' @param model Model model.
#' @param iter Number of MCMC iterations.
#' @param particles Number of state samples per MCMC iteration.
#' Ignored if \code{mcmc_type} is \code{"approx"}.
#' @param output_type Either \code{"full"} 
#' (default, returns posterior samples of states alpha and hyperparameters theta), 
#' \code{"theta"} (for marginal posterior of theta), 
#' or \code{"summary"} (return the mean and variance estimates of the states 
#' and posterior samples of theta).
#' @param mcmc_type What MCMC algorithm to use? Possible choices are
#' \code{"pm"} for pseudo-marginal MCMC,
#' \code{"da"} for delayed acceptance version of PMCMC , 
#' \code{"approx"} for approximate inference based on the Gaussian approximation of the model,
#' or one of the three importance sampling type weighting schemes:
#' \code{"is3"} for simple importance sampling (weight is computed for each MCMC iteration independently),
#' \code{"is2"} for jump chain importance sampling type weighting (default), or
#' \code{"is1"} for importance sampling type weighting where the number of particles used for
#' weight computations is proportional to the length of the jump chain block.
#' @param sampling_method If \code{"psi"}, \eqn{\psi}-APF is used for state sampling
#' (default). If \code{"spdk"}, non-sequential importance sampling based
#' on Gaussian approximation is used. If \code{"bsf"}, bootstrap filter
#' is used.
#' @param burnin Length of the burn-in period which is disregarded from the
#' results. Defaults to \code{iter / 2}.
#' @param thin Thinning rate. Defaults to 1. Increase for large models in
#' order to save memory. For IS-corrected methods, larger
#' value can also be statistically more effective. 
#' Note: With \code{output_type = "summary"}, the thinning does not affect the computations 
#' of the summary statistics in case of pseudo-marginal methods.
#' @param gamma Tuning parameter for the adaptation of RAM algorithm. Must be
#' between 0 and 1 (not checked).
#' @param target_acceptance Target acceptance ratio for RAM. Defaults to 0.234. 
#' @param S Initial value for the lower triangular matrix of RAM
#' algorithm, so that the covariance matrix of the Gaussian proposal
#' distribution is \eqn{SS'}. Note that for some parameters 
#' (currently the standard deviation and dispersion parameters of bsm_ng models) the sampling
#' is done for transformed parameters with internal_theta = log(theta).
#' @param end_adaptive_phase If \code{TRUE}, S is held fixed after the burnin period. Default is \code{FALSE}.
#' @param local_approx If \code{TRUE} (default), Gaussian approximation needed for
#' importance sampling is performed at each iteration. If \code{FALSE}, approximation is updated only
#' once at the start of the MCMC.
#' @param threads Number of threads for state simulation.
#' @param seed Seed for the random number generator.
#' @param max_iter Maximum number of iterations used in Gaussian approximation.
#' @param conv_tol Tolerance parameter used in Gaussian approximation.
#' @param ... Ignored.
#' @examples
#' set.seed(1)
#' n <- 50 
#' slope <- cumsum(c(0, rnorm(n - 1, sd = 0.001)))
#' level <- cumsum(slope + c(0, rnorm(n - 1, sd = 0.2)))
#' y <- rpois(n, exp(level))
#' poisson_model <- bsm_ng(y, 
#'   sd_level = halfnormal(0.01, 1), 
#'   sd_slope = halfnormal(0.01, 0.1), 
#'   P1 = diag(c(10, 0.1)), distribution = "poisson")
#'   
#' # Note small number of iterations for CRAN checks
#' mcmc_is <- run_mcmc(poisson_model, iter = 1000, particles = 10, 
#'   mcmc_type = "da")
#' summary(mcmc_is, what = "theta", return_se = TRUE)
#' 
#' set.seed(123)
#' n <- 50
#' sd_level <- 0.1
#' drift <- 0.01
#' beta <- -0.9
#' phi <- 5
#' 
#' level <- cumsum(c(5, drift + rnorm(n - 1, sd = sd_level)))
#' x <- 3 + (1:n) * drift + sin(1:n + runif(n, -1, 1))
#' y <- rnbinom(n, size = phi, mu = exp(beta * x + level))
#' 
#' model <- bsm_ng(y, xreg = x,
#'   beta = normal(0, 0, 10),
#'   phi = halfnormal(1, 10),
#'   sd_level = halfnormal(0.1, 1), 
#'   sd_slope = halfnormal(0.01, 0.1),
#'   a1 = c(0, 0), P1 = diag(c(10, 0.1)^2), 
#'   distribution = "negative binomial")
#' 
#' # run IS-MCMC
#' # Note small number of iterations for CRAN checks
#' fit <- run_mcmc(model, iter = 5000,
#'   particles = 10, mcmc_type = "is2", seed = 1)
#'
#' # extract states   
#' d_states <- as.data.frame(fit, variable = "states", time = 1:n)
#' 
#' library("dplyr")
#' library("ggplot2")
#' 
#'  # compute summary statistics
#' level_sumr <- d_states %>% 
#'   filter(variable == "level") %>%
#'   group_by(time) %>%
#'   summarise(mean = Hmisc::wtd.mean(value, weight, normwt = TRUE), 
#'     lwr = Hmisc::wtd.quantile(value, weight, 
#'       0.025, normwt = TRUE), 
#'     upr = Hmisc::wtd.quantile(value, weight, 
#'       0.975, normwt = TRUE))
#' 
#' # visualize
#' level_sumr %>% ggplot(aes(x = time, y = mean)) + 
#'   geom_line() +
#'   geom_line(aes(y = lwr), linetype = "dashed", na.rm = TRUE) +
#'   geom_line(aes(y = upr), linetype = "dashed", na.rm = TRUE) +
#'   theme_bw() + 
#'   theme(legend.title = element_blank()) + 
#'   xlab("Time") + ylab("Level")
#' 
#' # Bivariate Poisson model:
#' 
#' set.seed(1)
#' x <- cumsum(c(3, rnorm(19, sd = 0.5)))
#' y <- cbind(
#'   rpois(20, exp(x)), 
#'   rpois(20, exp(x)))
#' 
#' prior_fn <- function(theta) {
#'   # half-normal prior using transformation
#'   dnorm(exp(theta), 0, 1, log = TRUE) + theta # plus jacobian term
#' }
#' 
#' update_fn <- function(theta) {
#'   list(R = array(exp(theta), c(1, 1, 1)))
#' }
#' 
#' model <- ssm_mng(y = y, Z = matrix(1,2,1), T = 1, 
#'   R = 0.1, P1 = 1, distribution = "poisson",
#'   init_theta = log(0.1), 
#'   prior_fn = prior_fn, update_fn = update_fn)
#'   
#' # Note small number of iterations for CRAN checks
#' out <- run_mcmc(model, iter = 5000, mcmc_type = "approx")
#' 
#' sumr <- as.data.frame(out, variable = "states") %>% 
#'   group_by(time) %>% mutate(value = exp(value)) %>%
#'   summarise(mean = mean(value), 
#'     ymin = quantile(value, 0.05), ymax = quantile(value, 0.95))
#' ggplot(sumr, aes(time, mean)) + 
#' geom_ribbon(aes(ymin = ymin, ymax = ymax),alpha = 0.25) + 
#' geom_line() + 
#' geom_line(data = data.frame(mean = y[, 1], time = 1:20), colour = "tomato") + 
#' geom_line(data = data.frame(mean = y[, 2], time = 1:20), colour = "tomato") +
#' theme_bw()
#' 
run_mcmc.nongaussian <- function(model, iter, particles, output_type = "full",
  mcmc_type = "is2", sampling_method = "psi", burnin = floor(iter/2),
  thin = 1, gamma = 2/3, target_acceptance = 0.234, S, end_adaptive_phase = FALSE,
  local_approx  = TRUE, threads = 1,
  seed = sample(.Machine$integer.max, size = 1), max_iter = 100, conv_tol = 1e-8, ...) {
  
  if(missing(particles)) {
    nsim <- eval(match.call(expand.dots = TRUE)$nsim)
    if (!is.null(nsim)) {
      warning("Argument `nsim` is deprecated. Use argument `particles` instead.")
      particles <- nsim
    }
  }
  
  if(length(model$theta) == 0) stop("No unknown parameters ('model$theta' has length of zero).")
  a <- proc.time()
  check_target(target_acceptance)
  
  output_type <- pmatch(output_type, c("full", "summary", "theta"))
  mcmc_type <- match.arg(mcmc_type, c("pm", "da", paste0("is", 1:3), "approx"))
  if (mcmc_type == "approx") particles <- 0
  if (particles < 2 && mcmc_type != "approx") 
    stop("Number of state samples less than 2, use 'mcmc_type' 'approx' instead.")
  
  sampling_method <- pmatch(match.arg(sampling_method, c("psi", "bsf", "spdk")), 
    c("psi", "bsf", "spdk"))
  
  model$max_iter <- max_iter
  model$conv_tol <- conv_tol
  model$local_approx <- local_approx
  
  if(inherits(model, "bsm_ng")) {
    names_ind <-
      c(!model$fixed & c(TRUE, model$slope, model$seasonal), model$noise)
    transformed <- c(c("sd_level", "sd_slope", "sd_seasonal", "sd_noise")[names_ind], 
      if (model$distribution == "negative binomial") "phi")
    model$theta[transformed] <- log(pmax(1e-8, model$theta[transformed]))
  }
  
  if (missing(S)) {
    S <- diag(0.1 * pmax(0.1, abs(model$theta)), length(model$theta))
  }
  
  model$distribution <- pmatch(model$distribution,
    c("svm", "poisson", "binomial", "negative binomial", "gamma", "gaussian"), 
    duplicates.ok = TRUE) - 1
  
  switch(mcmc_type,
    "da" = {
      out <- nongaussian_da_mcmc(model, 
        output_type, particles, iter, burnin, thin, gamma, target_acceptance, S,
        seed, end_adaptive_phase, threads,
        sampling_method, model_type(model))
    },
    "pm" = {
      out <- nongaussian_pm_mcmc(model, output_type,
        particles, iter, burnin, thin, gamma, target_acceptance, S,
        seed, end_adaptive_phase, threads, 
        sampling_method, model_type(model))
    },
    "is1" =,
    "is2" =,
    "is3" = {
      out <- nongaussian_is_mcmc(model, output_type,
        particles, iter, burnin, thin, gamma, target_acceptance, S,
        seed, end_adaptive_phase, threads, 
        sampling_method,
        pmatch(mcmc_type, paste0("is", 1:3)), model_type(model), FALSE)
    },
    "approx" = {
      out <- nongaussian_is_mcmc(model, output_type,
        particles, iter, burnin, thin, gamma, target_acceptance, S,
        seed, end_adaptive_phase, threads, 
        sampling_method, 2, model_type(model), TRUE)
    })
  if (output_type == 1) {
    colnames(out$alpha) <- names(model$a1)
  } else {
    if (output_type == 2) {
      colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <-
        names(model$a1)
      out$alphahat <- ts(out$alphahat, start = start(model$y),
        frequency = frequency(model$y))
    }
  }
  
  colnames(out$theta) <- rownames(out$S) <- colnames(out$S) <- names(model$theta)
  if(inherits(model, "bsm_ng")) {
    out$theta[, transformed] <- exp(out$theta[, transformed])
  }
  out$iter <- iter
  out$burnin <- burnin
  out$thin <- thin
  out$mcmc_type <- mcmc_type
  out$output_type <- output_type
  out$call <- match.call()
  out$seed <- seed
  out$time <- proc.time() - a
  class(out) <- "mcmc_output"
  attr(out, "model_type") <- class(model)[1]
  attr(out, "ts") <- 
    list(start = start(model$y), end = end(model$y), frequency=frequency(model$y))
  out
}
#' Bayesian Inference of non-linear state space models 
#'
#' Methods for posterior inference of states and parameters.
#'
#' @method run_mcmc ssm_nlg
#' @param model Model model.
#' @param iter Number of MCMC iterations.
#' @param particles Number of state samples per MCMC iteration. 
#' Ignored if \code{mcmc_type} is \code{"approx"} or \code{"ekf"}.
#' @param output_type Either \code{"full"} 
#' (default, returns posterior samples of states alpha and hyperparameters theta), 
#' \code{"theta"} (for marginal posterior of theta), 
#' or \code{"summary"} (return the mean and variance estimates of the states 
#' and posterior samples of theta). 
#' @param mcmc_type What MCMC algorithm to use? Possible choices are
#' \code{"pm"} for pseudo-marginal MCMC,
#' \code{"da"} for delayed acceptance version of pseudo-marginal MCMC, 
#' \code{"approx"} for approximate inference based on the Gaussian approximation of the model,
#' \code{"ekf"} for approximate inference using extended Kalman filter, 
#' or one of the three importance sampling type weighting schemes:
#' \code{"is3"} for simple importance sampling (weight is computed for each MCMC iteration independently),
#' \code{"is2"} for jump chain importance sampling type weighting (default), or
#' \code{"is1"} for importance sampling type weighting where the number of particles used for
#' weight computations is proportional to the length of the jump chain block.
#' @param sampling_method If \code{"bsf"} (default), bootstrap filter is used for state sampling. 
#' If \code{"ekf"}, particle filter based on EKF-proposals are used. 
#' If \code{"psi"}, \eqn{\psi}-APF is used.
#' @param burnin Length of the burn-in period which is disregarded from the
#' results. Defaults to \code{iter / 2}.
#' @param thin Thinning rate. Defaults to 1. Increase for large models in
#' order to save memory. For IS-corrected methods, larger
#' value can also be statistically more effective. 
#' Note: With \code{output_type = "summary"}, the thinning does not affect the computations 
#' of the summary statistics in case of pseudo-marginal methods.
#' @param gamma Tuning parameter for the adaptation of RAM algorithm. Must be
#' between 0 and 1 (not checked).
#' @param target_acceptance Target acceptance ratio for RAM. Defaults to 0.234.
#' @param S Initial value for the lower triangular matrix of RAM
#' algorithm, so that the covariance matrix of the Gaussian proposal
#' distribution is \eqn{SS'}. Note that for some parameters 
#' (currently the standard deviation and dispersion parameters of bsm_ng models) the sampling
#' is done for transformed parameters with internal_theta = log(theta).
#' @param end_adaptive_phase If \code{TRUE}, S is held fixed after the burnin period. Default is \code{FALSE}.
#' @param threads Number of threads for state simulation.
#' @param seed Seed for the random number generator.
#' @param max_iter Maximum number of iterations used in Gaussian approximation.
#' @param conv_tol Tolerance parameter used in Gaussian approximation.
#' @param iekf_iter If \code{iekf_iter > 0}, iterated extended Kalman filter is used with
#' \code{iekf_iter} iterations in place of standard EKF. Defaults to zero.
#' @param ... Ignored.
#' @export
#' @references 
#' Vihola, M, Helske, J, Franks, J. Importance sampling type estimators based on approximate marginal Markov chain Monte Carlo. 
#' Scand J Statist. 2020; 1– 38. https://doi.org/10.1111/sjos.12492
run_mcmc.ssm_nlg <-  function(model, iter, particles, output_type = "full",
  mcmc_type = "is2", sampling_method = "bsf",
  burnin = floor(iter/2), thin = 1,
  gamma = 2/3, target_acceptance = 0.234, S, end_adaptive_phase = FALSE,
  threads = 1, seed = sample(.Machine$integer.max, size = 1), max_iter = 100,
  conv_tol = 1e-8, iekf_iter = 0, ...) {
  
  if(missing(particles)) {
    nsim <- eval(match.call(expand.dots = TRUE)$nsim)
    if (!is.null(nsim)) {
      warning("Argument `nsim` is deprecated. Use argument `particles` instead.")
      particles <- nsim
    }
  }
  
  if(length(model$theta) == 0) stop("No unknown parameters ('model$theta' has length of zero).")
  a <- proc.time()
  check_target(target_acceptance)
  
  output_type <- pmatch(output_type, c("full", "summary", "theta"))
  mcmc_type <- match.arg(mcmc_type, c("pm", "da", paste0("is", 1:3), "ekf", "approx"))
  if(mcmc_type %in% c("ekf", "approx")) particles <- 0
  sampling_method <- pmatch(match.arg(sampling_method, c("psi", "bsf", "ekf")), 
    c("psi", "bsf", NA, "ekf"))
  
  if (missing(S)) {
    S <- diag(0.1 * pmax(0.1, abs(model$theta)), length(model$theta))
  }
  
  if (particles < 2 && !(mcmc_type %in% c("ekf", "approx")))
     stop("Number of state samples less than 2, use 'mcmc_type' 'approx' or 'ekf' instead.")
 
  
  out <- switch(mcmc_type,
    "da" = {
      nonlinear_da_mcmc(t(model$y), model$Z, model$H, model$T,
        model$R, model$Z_gn, model$T_gn, model$a1, model$P1,
        model$theta, model$log_prior_pdf, model$known_params,
        model$known_tv_params, as.integer(model$time_varying),
        model$n_states, model$n_etas, seed,
        particles, iter, burnin, thin, gamma, target_acceptance, S,
        end_adaptive_phase, threads, max_iter, conv_tol,
        sampling_method,iekf_iter, output_type, 
        default_update_fn, default_prior_fn)
    },
    "pm" = {
      nonlinear_pm_mcmc(t(model$y), model$Z, model$H, model$T,
        model$R, model$Z_gn, model$T_gn, model$a1, model$P1,
        model$theta, model$log_prior_pdf, model$known_params,
        model$known_tv_params, as.integer(model$time_varying),
        model$n_states, model$n_etas, seed,
        particles, iter, burnin, thin, gamma, target_acceptance, S,
        end_adaptive_phase, threads, max_iter, conv_tol,
        sampling_method,iekf_iter, output_type, 
        default_update_fn, default_prior_fn)
    },
    "is1" =,
    "is2" =,
    "is3" = {
      if (sampling_method == 4)
        stop("IS-MCMC with extended particle filter is (not yet) supported.")
      nonlinear_is_mcmc(t(model$y), model$Z, model$H, model$T,
        model$R, model$Z_gn, model$T_gn, model$a1, model$P1,
        model$theta, model$log_prior_pdf, model$known_params,
        model$known_tv_params, as.integer(model$time_varying),
        model$n_states, model$n_etas, seed,
        particles, iter, burnin, thin, gamma, target_acceptance, S,
        end_adaptive_phase, threads, pmatch(mcmc_type, paste0("is", 1:3)),
        sampling_method, max_iter, conv_tol, iekf_iter, 
        output_type, default_update_fn, 
        default_prior_fn, FALSE)
    },
    "ekf" = {
      nonlinear_ekf_mcmc(t(model$y), model$Z, model$H, model$T,
        model$R, model$Z_gn, model$T_gn, model$a1, model$P1,
        model$theta, model$log_prior_pdf, model$known_params,
        model$known_tv_params, as.integer(model$time_varying),
        model$n_states, model$n_etas, seed,
        iter, burnin, thin, gamma, target_acceptance, S,
        end_adaptive_phase,  threads, iekf_iter, output_type, 
        default_update_fn, default_prior_fn)
    },
    "approx" = {
      nonlinear_is_mcmc(t(model$y), model$Z, model$H, model$T,
        model$R, model$Z_gn, model$T_gn, model$a1, model$P1,
        model$theta, model$log_prior_pdf, model$known_params,
        model$known_tv_params, as.integer(model$time_varying),
        model$n_states, model$n_etas, seed,
        particles, iter, burnin, thin, gamma, target_acceptance, S,
        end_adaptive_phase, threads, 2,
        sampling_method, max_iter, conv_tol, 
        iekf_iter, output_type, default_update_fn, 
        default_prior_fn, TRUE)
    }
  )
  if (output_type == 1) {
    colnames(out$alpha) <- model$state_names
  } else {
    if (output_type == 2) {
      colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <-
        model$state_names
      out$alphahat <- ts(out$alphahat, start = start(model$y),
        frequency = frequency(model$y))
    }
  }
  
  
  colnames(out$theta) <- rownames(out$S) <- colnames(out$S) <- names(model$theta)
  
  out$iter <- iter
  out$burnin <- burnin
  out$thin <- thin
  out$mcmc_type <- mcmc_type
  out$output_type <- output_type
  out$call <- match.call()
  out$seed <- seed
  out$time <- proc.time() - a
  class(out) <- "mcmc_output"
  attr(out, "model_type") <- "ssm_nlg"
  attr(out, "ts") <- 
    list(start = start(model$y), end = end(model$y), frequency=frequency(model$y))
  out
}
#' Bayesian Inference of SDE 
#'
#' Methods for posterior inference of states and parameters.
#'
#' @method run_mcmc ssm_sde
#' @param model Model model.
#' @param iter Number of MCMC iterations.
#' @param particles Number of state samples per MCMC iteration.
#' @param output_type Either \code{"full"} 
#' (default, returns posterior samples of states alpha and hyperparameters theta), 
#' \code{"theta"} (for marginal posterior of theta), 
#' or \code{"summary"} (return the mean and variance estimates of the states 
#' and posterior samples of theta). If \code{particles = 0}, this is argument ignored and set to \code{"theta"}.
#' @param mcmc_type What MCMC algorithm to use? Possible choices are
#' \code{"pm"} for pseudo-marginal MCMC,
#' \code{"da"} for delayed acceptance version of pseudo-marginal MCMC, 
#' or one of the three importance sampling type weighting schemes:
#' \code{"is3"} for simple importance sampling (weight is computed for each MCMC iteration independently),
#' \code{"is2"} for jump chain importance sampling type weighting (default), or
#' \code{"is1"} for importance sampling type weighting where the number of particles used for
#' weight computations is proportional to the length of the jump chain block.
#' @param burnin Length of the burn-in period which is disregarded from the
#' results. Defaults to \code{iter / 2}.
#' @param thin Thinning rate. Defaults to 1. Increase for large models in
#' order to save memory. For IS-corrected methods, larger
#' value can also be statistically more effective. 
#' Note: With \code{output_type = "summary"}, the thinning does not affect the computations 
#' of the summary statistics in case of pseudo-marginal methods.
#' @param gamma Tuning parameter for the adaptation of RAM algorithm. Must be
#' between 0 and 1 (not checked).
#' @param target_acceptance Target acceptance ratio for RAM. Defaults to 0.234.
#' @param S Initial value for the lower triangular matrix of RAM
#' algorithm, so that the covariance matrix of the Gaussian proposal
#' distribution is \eqn{SS'}. Note that for some parameters 
#' (currently the standard deviation and dispersion parameters of bsm_ng models) the sampling
#' is done for transformed parameters with internal_theta = log(theta).
#' @param end_adaptive_phase If \code{TRUE}, S is held fixed after the burnin period. Default is \code{FALSE}.
#' @param threads Number of threads for state simulation.
#' @param L_c,L_f Integer values defining the discretization levels for first and second stages (defined as 2^L). 
#' For PM methods, maximum of these is used.
#' @param seed Seed for the random number generator.
#' @param ... Ignored.
#' @export
#' @references 
#' Vihola, M, Helske, J, Franks, J. Importance sampling type estimators based on approximate marginal Markov chain Monte Carlo. 
#' Scand J Statist. 2020; 1– 38. https://doi.org/10.1111/sjos.12492
run_mcmc.ssm_sde <-  function(model, iter, particles, output_type = "full",
  mcmc_type = "is2", L_c, L_f,
  burnin = floor(iter/2), thin = 1,
  gamma = 2/3, target_acceptance = 0.234, S, end_adaptive_phase = FALSE,
  threads = 1, seed = sample(.Machine$integer.max, size = 1), ...) {
  
  if(any(c(model$drift, model$diffusion, model$ddiffusion,
    model$prior_pdf, model$obs_pdf) %in% c("<pointer: (nil)>", "<pointer: 0x0>"))) {
    stop("NULL pointer detected, please recompile the pointer file and reconstruct the model.")
  }
  
  if(missing(particles)) {
    nsim <- eval(match.call(expand.dots = TRUE)$nsim)
    if (!is.null(nsim)) {
      warning("Argument `nsim` is deprecated. Use argument `particles` instead.")
      particles <- nsim
    }
  }
  
  if(length(model$theta) == 0) stop("No unknown parameters ('model$theta' has length of zero).")
  a <- proc.time()
  check_target(target_acceptance)
  if(particles <= 0) stop("particles should be positive integer.")
  
  output_type <- pmatch(output_type, c("full", "summary", "theta"))
  mcmc_type <- match.arg(mcmc_type, c("pm", "da", paste0("is", 1:3)))
  
  if (missing(S)) {
    S <- diag(0.1 * pmax(0.1, abs(model$theta)), length(model$theta))
  }
  
  if (mcmc_type == "da"){
    if (L_f <= L_c) stop("L_f should be larger than L_c.")
    if(L_c < 1) stop("L_c should be at least 1")
    out <- sde_da_mcmc(model$y, model$x0, model$positive,
      model$drift, model$diffusion, model$ddiffusion,
      model$prior_pdf, model$obs_pdf, model$theta,
      particles, L_c, L_f, seed,
      iter, burnin, thin, gamma, target_acceptance, S,
      end_adaptive_phase, output_type)
  } else {
    if(mcmc_type == "pm") {
      if (missing(L_c)) L_c <- 0
      if (missing(L_f)) L_f <- 0
      L <- max(L_c, L_f)
      if(L <= 0) stop("L should be positive.")
      out <- sde_pm_mcmc(model$y, model$x0, model$positive,
        model$drift, model$diffusion, model$ddiffusion,
        model$prior_pdf, model$obs_pdf, model$theta,
        particles, L, seed,
        iter, burnin, thin, gamma, target_acceptance, S,
        end_adaptive_phase, output_type)
    } else {
      if (L_f <= L_c) stop("L_f should be larger than L_c.")
      if(L_c < 1) stop("L_c should be at least 1")
      
      out <- sde_is_mcmc(model$y, model$x0, model$positive,
        model$drift, model$diffusion, model$ddiffusion,
        model$prior_pdf, model$obs_pdf, model$theta,
        particles, L_c, L_f, seed,
        iter, burnin, thin, gamma, target_acceptance, S,
        end_adaptive_phase, pmatch(mcmc_type, paste0("is", 1:3)), 
        threads, output_type)
    }
  }
  colnames(out$alpha) <- model$state_names
  
  colnames(out$theta) <- rownames(out$S) <- colnames(out$S) <- names(model$theta)
  
  out$iter <- iter
  out$burnin <- burnin
  out$thin <- thin
  out$mcmc_type <- mcmc_type
  out$output_type <- output_type
  out$call <- match.call()
  out$seed <- seed
  out$time <- proc.time() - a
  class(out) <- "mcmc_output"
  attr(out, "model_type") <- "ssm_sde"
  attr(out, "ts") <- 
    list(start = start(model$y), end = end(model$y), frequency=frequency(model$y))
  out
}
