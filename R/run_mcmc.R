#' Bayesian Inference of State Space Models
#'
#' Adaptive Markov chain Monte Carlo simulation of state space models using
#' Robust Adaptive Metropolis algorithm by Vihola (2012).
#'
#' @param object State space model object of \code{bssm} package.
#' @param n_iter Number of MCMC iterations.
#' @param ... Parameters to specific methods. See \code{\link{run_mcmc.gssm}} and
#' \code{\link{run_mcmc.ngssm}} for details.
#' @export
#' @rdname run_mcmc
#' @references Matti Vihola (2012). "Robust adaptive Metropolis algorithm with
#' coerced acceptance rate". Statistics and Computing, Volume 22, Issue 5,
#' pages 997--1008.
#' Matti Vihola, Jouni Helske, Jordan Franks (2016). "Importance sampling type
#' correction of Markov chain Monte Carlo and exact approximations."
#' ArXiv:1609.02541.
run_mcmc <- function(object, n_iter, ...) {
  UseMethod("run_mcmc", object)
}
#' Bayesian Inference of Linear-Gaussian State Space Models
#'
#' @method run_mcmc gssm
#' @rdname run_mcmc_g
#' @param object Model object.
#' @param n_iter Number of MCMC iterations.
#' @param sim_states Simulate states of Gaussian state space models. Default is \code{TRUE}. 
#' Ignored if argument \code{type = "summary"}.
#' @param type Type of output. Default is \code{"full"}, which returns
#' samples from the posterior \eqn{p(\alpha, \theta)}. 
#'  Option \code{"summary"} does not simulate
#' states directly but computes the posterior means and variances of states using
#' fast Kalman smoothing. This is slightly faster, memory  efficient and
#' more accurate than calculations based on simulation smoother.
#' \eqn{\theta}. Optional for \code{bsm} objects.
#' @param n_burnin Length of the burn-in period which is disregarded from the
#' results. Defaults to \code{n_iter / 2}. Note that all MCMC algorithms of \code{bssm}
#'  used adaptive MCMC during the burn-in period in order to find good proposal.
#' @param n_thin Thinning rate. All MCMC algoritms in \code{bssm} use the jump chain
#' representation, and the thinning is applied to these blocks.
#' This defaults to 1, but for IS-corrected methods, larger
#' value is often computationally more effective.
#' @param gamma Tuning parameter for the adaptation of RAM algorithm. Must be
#' between 0 and 1 (not checked).
#' @param target_acceptance Target acceptance ratio for RAM. Defaults to 0.234.
#' @param S Initial value for the lower triangular matrix of RAM
#' algorithm, so that the covariance matrix of the Gaussian proposal
#' distribution is \eqn{SS'}. Note that for some parameters 
#' (currently the standard deviation and dispersion parameters of bsm models) the sampling
#' is done for transformed parameters with internal_theta = log(1 + theta).
#' @param end_adaptive_phase If \code{TRUE} (default), $S$ is held fixed after the burnin period.
#' @param n_threads Number of threads for state simulation.
#' @param seed Seed for the random number generator.
#' @param ... Ignored.
#' @export
run_mcmc.gssm <- function(object, n_iter, sim_states = TRUE, type = "full",
  n_burnin = floor(n_iter / 2), n_thin = 1, gamma = 2/3,
  target_acceptance = 0.234, S, end_adaptive_phase = TRUE, n_threads = 1,
  seed = sample(.Machine$integer.max, size = 1), ...) {
  
  a <- proc.time()
  
  check_target(target_acceptance)
  
  type <- match.arg(type, c("full", "summary"))
  
  if (missing(S)) {
    S <- diag(0.1 * pmax(0.1, abs(object$theta)), length(object$theta))
  }
  out <- switch(type,
    full = {
      out <- gaussian_mcmc(object, sim_states,
        n_iter, n_burnin, n_thin, gamma, target_acceptance, S, seed,
        end_adaptive_phase, n_threads, model_type = 1L,
        object$Z_ind, object$H_ind, object$T_ind, object$R_ind)
      if (sim_states) {
        colnames(out$alpha) <- names(object$a1)
      }
      out
    },
    summary = {
      out <- gaussian_mcmc_summary(object, 
        n_iter, n_burnin, n_thin, gamma, target_acceptance, S, seed,
        end_adaptive_phase, n_threads, model_type = 1L,
        object$Z_ind, object$H_ind, object$T_ind, object$R_ind)
      colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <-
        names(object$a1)
      out$alphahat <- ts(out$alphahat, start = start(object$y),
        frequency = frequency(object$y))
      out
    }
  )
  out$call <- match.call()
  out$seed <- seed
  out$n_iter <- n_iter
  out$n_burnin <- n_burnin
  out$n_thin <- n_thin
  out$mcmc_type <- "gaussian_mcmc"
  out$time <- proc.time() - a
  class(out) <- "mcmc_output"
  attr(out, "model_type") <- "gssm"
  out
}

#' @method run_mcmc bsm
#' @rdname run_mcmc_g
#' @inheritParams run_mcmc.gssm
#' @export
run_mcmc.bsm <- function(object, n_iter, sim_states = TRUE, type = "full",
  n_burnin = floor(n_iter/2), n_thin = 1, gamma = 2/3,
  target_acceptance = 0.234, S, end_adaptive_phase = TRUE,
  n_threads = 1, seed = sample(.Machine$integer.max, size = 1), ...) {
  
  a <- proc.time()
  check_target(target_acceptance)
  
  type <- match.arg(type, c("full", "summary"))
  
  names_ind <- !object$fixed & c(TRUE, TRUE, object$slope, object$seasonal)
  object$theta[c("sd_y", "sd_level", "sd_slope", "sd_seasonal")[names_ind]] <- 
    log(1 + object$theta[c("sd_y", "sd_level", "sd_slope", "sd_seasonal")[names_ind]])
  if (missing(S)) {
    S <- diag(0.1 * pmax(0.1, abs(object$theta)), length(object$theta))
  }
  
  
  out <- switch(type,
    full = {
      out <- gaussian_mcmc(object, sim_states,
        n_iter, n_burnin, n_thin, gamma, target_acceptance, S, seed,
        end_adaptive_phase, n_threads, model_type = 2L, 0, 0, 0, 0)
      
      if (sim_states) {
        colnames(out$alpha) <- names(object$a1)
      }
      out
    },
    summary = {
      out <- gaussian_mcmc_summary(object,
        n_iter, n_burnin, n_thin, gamma, target_acceptance, S, seed,
        end_adaptive_phase, n_threads, model_type = 2L, 0, 0, 0, 0)
      colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <-
        names(object$a1)
      out$alphahat <- ts(out$alphahat, start = start(object$y),
        frequency = frequency(object$y))
      out
    })
  
  
  colnames(out$theta) <- rownames(out$S) <- colnames(out$S) <- names(object$theta)
  out$theta[, c("sd_y", "sd_level", "sd_slope", "sd_seasonal")[names_ind]] <- 
    exp(out$theta[, c("sd_y", "sd_level", "sd_slope", "sd_seasonal")[names_ind]]) - 1
  out$call <- match.call()
  out$seed <- seed
  out$n_iter <- n_iter
  out$n_burnin <- n_burnin
  out$n_thin <- n_thin
  out$mcmc_type <- "gaussian_mcmc"
  out$time <- proc.time() - a
  class(out) <- "mcmc_output"
  attr(out, "model_type") <- "bsm"
  out
}



#' Bayesian inference of non-Gaussian or non-linear state space models using MCMC
#'
#' Methods for posterior inference of states and parameters.
#'
#' @method run_mcmc ngssm
#' @rdname run_mcmc_ng
#' @param object Model object.
#' @param n_iter Number of MCMC iterations.
#' @param nsim_states Number of state samples per MCMC iteration.
#' If <2, approximate inference based on Gaussian approximation is performed.
#' @param type Either \code{"full"} (default), or \code{"summary"}. The
#' former produces samples of states whereas the latter gives the mean and
#' variance estimates of the states.
#' @param method What MCMC algorithm to use? Possible choices are
#' \code{"pm"} for pseudo-marginal MCMC,
#' \code{"da"} for delayed acceptance version of PMCMC (default), or one of the three
#' importance sampling type weighting schemes:
#' \code{"is3"} for simple importance sampling (weight is computed for each MCMC iteration independently),
#' \code{"is2"} for jump chain importance sampling type weighting, or
#' \code{"is1"} for importance sampling type weighting where the number of particles used for
#' weight computations is proportional to the length of the jump chain block.
#' @param simulation_method If \code{"spdk"}, non-sequential importance sampling based
#' on Gaussian approximation is used. If \code{"bsf"}, bootstrap filter
#' is used (default for \code{"nlg_ssm"} and only option for \code{"sde_ssm"}),
#' and if \code{"psi"}, psi-auxiliary particle filter is used
#' (default for models with linear-Gaussian state equation).
#' @param n_burnin Length of the burn-in period which is disregarded from the
#' results. Defaults to \code{n_iter / 2}.
#' @param n_thin Thinning rate. Defaults to 1. Increase for large models in
#' order to save memory.
#' @param gamma Tuning parameter for the adaptation of RAM algorithm. Must be
#' between 0 and 1 (not checked).
#' @param target_acceptance Target acceptance ratio for RAM. Defaults to 0.234.
#' @param S Initial value for the lower triangular matrix of RAM
#' algorithm, so that the covariance matrix of the Gaussian proposal
#' distribution is \eqn{SS'}.
#' @param end_adaptive_phase If \code{TRUE} (default), $S$ is held fixed after the burnin period.
#' @param local_approx If \code{TRUE} (default), Gaussian approximation needed for
#' importance sampling is performed at each iteration. If false, approximation is updated only
#' once at the start of the MCMC. Not used for non-linear models.
#' @param n_threads Number of threads for state simulation.
#' @param seed Seed for the random number generator.
#' @param max_iter Maximum number of iterations used in Gaussian approximation. Used psi-PF.
#' @param conv_tol Tolerance parameter used in Gaussian approximation. Used psi-PF.
#' @param iekf_iter If zero (default), first approximation for non-linear
#' Gaussian models is obtained from extended Kalman filter. If
#' \code{iekf_iter > 0}, iterated extended Kalman filter is used with
#' \code{iekf_iter} iterations.
#' @param ... Ignored.
#' @export
run_mcmc.ngssm <- function(object, n_iter, nsim_states, type = "full",
  method = "da", simulation_method = "psi", n_burnin = floor(n_iter/2),
  n_thin = 1, gamma = 2/3, target_acceptance = 0.234, S, end_adaptive_phase = TRUE,
  local_approx  = TRUE, n_threads = 1,
  seed = sample(.Machine$integer.max, size = 1), max_iter = 100, conv_tol = 1e-8, ...) {
  
  a <- proc.time()
  check_target(target_acceptance)
  
  type <- match.arg(type, c("full", "summary"))
  method <- match.arg(method, c("pm", "da", paste0("is", 1:3)))
  simulation_method <- match.arg(simulation_method, c("psi", "bsf", "spdk"))
  
  if (nsim_states < 2) {
    method <- "is2"
  }
  
  if (missing(S)) {
    S <- diag(0.1 * pmax(0.1, abs(object$theta)), length(object$theta))
  }
  
  object$distribution <- pmatch(object$distribution,
    c("poisson", "binomial", "negative binomial"))
  
  out <-  switch(type,
    full = {
      if (method == "da") {
        out <- nongaussian_da_mcmc(object,
          nsim_states, n_iter, n_burnin, n_thin, gamma, target_acceptance, S,
          seed, end_adaptive_phase, n_threads, local_approx, object$initial_mode,
          max_iter, conv_tol, pmatch(simulation_method, c("psi", "bsf", "spdk")),
          model_type = 1L, object$Z_ind, object$T_ind, object$R_ind)
      } else {
        if(method == "pm"){
          out <- nongaussian_pm_mcmc(object, 
            nsim_states, n_iter, n_burnin, n_thin, gamma, target_acceptance, S,
            seed, end_adaptive_phase, n_threads, local_approx, object$initial_mode,
            max_iter, conv_tol, pmatch(simulation_method, c("psi", "bsf", "spdk")),
            model_type = 1L, object$Z_ind, object$T_ind, object$R_ind)
        } else {
          out <- nongaussian_is_mcmc(object, 
            nsim_states, n_iter, n_burnin, n_thin, gamma, target_acceptance, S,
            seed, end_adaptive_phase, n_threads, local_approx, object$initial_mode,
            max_iter, conv_tol, pmatch(simulation_method, c("psi", "bsf", "spdk")),
            pmatch(method, paste0("is", 1:3)),
            model_type = 1L, object$Z_ind, object$T_ind, object$R_ind)
        }
      }
      colnames(out$alpha) <- names(object$a1)
      out
    },
    summary = {
      stop("summary method for general models is not yet implemented.")
    })
  
  out$n_iter <- n_iter
  out$n_burnin <- n_burnin
  out$n_thin <- n_thin
  out$mcmc_type <- method
  out$call <- match.call()
  out$seed <- seed
  out$time <- proc.time() - a
  class(out) <- "mcmc_output"
  attr(out, "model_type") <- "ngssm"
  out
}


#' @method run_mcmc ng_bsm
#' @rdname run_mcmc_ng
#' @export
run_mcmc.ng_bsm <-  function(object, n_iter, nsim_states, type = "full",
  method = "da", simulation_method = "psi",
  n_burnin = floor(n_iter/2), n_thin = 1,
  gamma = 2/3, target_acceptance = 0.234, S, end_adaptive_phase = TRUE,
  local_approx  = TRUE, n_threads = 1,
  seed = sample(.Machine$integer.max, size = 1), max_iter = 100, conv_tol = 1e-8, ...) {
  
  a <- proc.time()
  check_target(target_acceptance)
  
  type <- match.arg(type, c("full", "summary"))
  method <- match.arg(method, c("pm", "da", paste0("is", 1:3)))
  simulation_method <- match.arg(simulation_method, c("psi", "bsf", "spdk"))
  
  if (nsim_states < 2) {
    #approximate inference
    method <- "is2"
  }
  
  names_ind <-
    c(!object$fixed & c(TRUE, object$slope, object$seasonal), object$noise)
  transformed <- c(c("sd_level", "sd_slope", "sd_seasonal", "sd_noise")[names_ind], 
    if (object$distribution == "negative binomial") "nb_dispersion")
  object$theta[transformed] <- log(1 + object$theta[transformed])
  
  if (missing(S)) {
    S <- diag(0.1 * pmax(0.1, abs(object$theta)), length(object$theta))
  }
  
  object$distribution <- pmatch(object$distribution,
    c("poisson", "binomial", "negative binomial"))
  
  out <-  switch(type,
    full = {
      if (method == "da") {
        out <- nongaussian_da_mcmc(object,
          nsim_states, n_iter, n_burnin, n_thin, gamma, target_acceptance, S,
          seed, end_adaptive_phase, n_threads, local_approx, object$initial_mode,
          max_iter, conv_tol, pmatch(simulation_method, c("psi", "bsf", "spdk")),
          model_type = 2L, 0, 0, 0)
      } else {
        if(method == "pm") {
          out <- nongaussian_pm_mcmc(object, 
            nsim_states, n_iter, n_burnin, n_thin, gamma, target_acceptance, S,
            seed, end_adaptive_phase, n_threads, local_approx, object$initial_mode,
            max_iter, conv_tol, pmatch(simulation_method, c("psi", "bsf", "spdk")),
            model_type = 2L, 0, 0, 0)
        } else {
          out <- nongaussian_is_mcmc(object,
            nsim_states, n_iter, n_burnin, n_thin, gamma, target_acceptance, S,
            seed, end_adaptive_phase, n_threads, local_approx, object$initial_mode,
            max_iter, conv_tol, pmatch(simulation_method, c("psi", "bsf", "spdk")),
            pmatch(method, paste0("is", 1:3)),
            model_type = 2L, 0, 0, 0)
        }
      }
      colnames(out$alpha) <- names(object$a1)
      out
    },
    summary = {
      stop("summary method not yet re-implemented.")
      # if (method == "pm"){
      #   out <- ng_bsm_run_mcmc_summary(object, priors$prior_types, priors$params, n_iter,
      #     nsim_states, n_burnin, n_thin, gamma, target_acceptance, S,
      #     object$initial_mode, seed,  n_threads, end_adaptive_phase, local_approx,
      #     delayed_acceptance, pmatch(simulation_method, c("psi", "bsf", "spdk")))
      # } else {
      #
      #   out <- ng_bsm_run_mcmc_summary_is(object, priors$prior_types, priors$params, n_iter,
      #     nsim_states, n_burnin, n_thin, gamma, target_acceptance, S,
      #     object$initial_mode, seed,  n_threads, end_adaptive_phase, local_approx,
      #     pmatch(simulation_method, c("psi", "bsf", "spdk")), pmatch(method, paste0("is", 1:3)),
      #     sample(.Machine$integer.max, size = n_threads))
      # }
      colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <- names(object$a1)
      out$alphahat <- ts(out$alphahat, start = start(object$y), frequency = frequency(object$y))
      out$muhat <- ts(out$muhat, start = start(object$y), frequency = frequency(object$y))
      out
    })
  
  
  colnames(out$theta) <- rownames(out$S) <- colnames(out$S) <- names(object$theta)
  out$theta[, transformed] <- exp(out$theta[, transformed]) - 1
  
  out$n_iter <- n_iter
  out$n_burnin <- n_burnin
  out$n_thin <- n_thin
  out$mcmc_type <- method
  out$call <- match.call()
  out$seed <- seed
  out$time <- proc.time() - a
  class(out) <- "mcmc_output"
  attr(out, "model_type") <- "ng_bsm"
  out
}

#' @method run_mcmc ng_ar1
#' @rdname run_mcmc_ng
#' @export
run_mcmc.ng_ar1 <-  function(object, n_iter, nsim_states, type = "full",
  method = "da", simulation_method = "psi",
  n_burnin = floor(n_iter/2), n_thin = 1,
  gamma = 2/3, target_acceptance = 0.234, S, end_adaptive_phase = TRUE,
  local_approx  = TRUE, n_threads = 1,
  seed = sample(.Machine$integer.max, size = 1), max_iter = 100, conv_tol = 1e-8, ...) {
  
  a <- proc.time()
  check_target(target_acceptance)
  
  type <- match.arg(type, c("full", "summary"))
  method <- match.arg(method, c("pm", "da", paste0("is", 1:3)))
  simulation_method <- match.arg(simulation_method, c("psi", "bsf", "spdk"))
  
  if (nsim_states < 2) {
    #approximate inference
    method <- "is2"
  }
  
  if (missing(S)) {
    S <- diag(0.1 * pmax(0.1, abs(object$theta)), length(object$theta))
  }
  
  
  object$distribution <- pmatch(object$distribution,
    c("poisson", "binomial", "negative binomial"))
  
  out <-  switch(type,
    full = {
      if (method == "da") {
        out <- nongaussian_da_mcmc(object, 
          nsim_states, n_iter, n_burnin, n_thin, gamma, target_acceptance, S,
          seed, end_adaptive_phase, n_threads, local_approx, object$initial_mode,
          max_iter, conv_tol, pmatch(simulation_method, c("psi", "bsf", "spdk")),
          model_type = 4L, 0, 0, 0)
      } else {
        if(method == "pm") {
          out <- nongaussian_pm_mcmc(object, 
            nsim_states, n_iter, n_burnin, n_thin, gamma, target_acceptance, S,
            seed, end_adaptive_phase, n_threads, local_approx, object$initial_mode,
            max_iter, conv_tol, pmatch(simulation_method, c("psi", "bsf", "spdk")),
            model_type = 4L, 0, 0, 0)
        } else {
          out <- nongaussian_is_mcmc(object, 
            nsim_states, n_iter, n_burnin, n_thin, gamma, target_acceptance, S,
            seed, end_adaptive_phase, n_threads, local_approx, object$initial_mode,
            max_iter, conv_tol, pmatch(simulation_method, c("psi", "bsf", "spdk")),
            pmatch(method, paste0("is", 1:3)),
            model_type = 4L, 0, 0, 0)
        }
      }
      colnames(out$alpha) <- names(object$a1)
      out
    },
    summary = {
      stop("summary method not yet re-implemented.")
      colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <- names(object$a1)
      out$alphahat <- ts(out$alphahat, start = start(object$y), frequency = frequency(object$y))
      out$muhat <- ts(out$muhat, start = start(object$y), frequency = frequency(object$y))
      out
    })
  
  
  colnames(out$theta) <- rownames(out$S) <- colnames(out$S) <- names(object$theta)
  out$n_iter <- n_iter
  out$n_burnin <- n_burnin
  out$n_thin <- n_thin
  out$mcmc_type <- method
  out$call <- match.call()
  out$seed <- seed
  out$time <- proc.time() - a
  class(out) <- "mcmc_output"
  attr(out, "model_type") <- "ng_ar1"
  out
}

#' @method run_mcmc ar1
#' @rdname run_mcmc
#' @export
run_mcmc.ar1 <-  function(object, n_iter, sim_states = TRUE, type = "full",
  n_burnin = floor(n_iter/2), n_thin = 1,
  gamma = 2/3, target_acceptance = 0.234, S, end_adaptive_phase = TRUE,
  n_threads = 1, seed = sample(.Machine$integer.max, size = 1), ...) {
  
  a <- proc.time()
  check_target(target_acceptance)
  
  type <- match.arg(type, c("full", "summary"))
  
  if (missing(S)) {
    S <- diag(0.1 * pmax(0.1, abs(object$theta)), length(object$theta))
  }
  
  out <- switch(type,
    full = {
      out <- gaussian_mcmc(object, sim_states,
        n_iter, n_burnin, n_thin, gamma, target_acceptance, S, seed,
        end_adaptive_phase, n_threads, model_type = 3L, 0, 0, 0, 0)
      
      if (sim_states) {
        colnames(out$alpha) <- names(object$a1)
      }
      out
    },
    summary = {
      out <- gaussian_mcmc_summary(object,
        n_iter, n_burnin, n_thin, gamma, target_acceptance, S, seed,
        end_adaptive_phase, n_threads, model_type = 3L, 0, 0, 0, 0)
      colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <-
        names(object$a1)
      out$alphahat <- ts(out$alphahat, start = start(object$y),
        frequency = frequency(object$y))
      out
    })
  
  
  colnames(out$theta) <- rownames(out$S) <- colnames(out$S) <- names(object$theta)
  out$n_iter <- n_iter
  out$n_burnin <- n_burnin
  out$n_thin <- n_thin
  out$mcmc_type <- "gaussian_mcmc"
  out$call <- match.call()
  out$seed <- seed
  out$time <- proc.time() - a
  class(out) <- "mcmc_output"
  attr(out, "model_type") <- "ar1"
  out
}

#' @method run_mcmc svm
#' @rdname run_mcmc_ng
#' @inheritParams run_mcmc.ngssm
#' @export
#'
run_mcmc.svm <-  function(object, n_iter, nsim_states, type = "full",
  method = "da", simulation_method = "psi",
  n_burnin = floor(n_iter/2),
  n_thin = 1, gamma = 2/3, target_acceptance = 0.234, S, end_adaptive_phase = TRUE,
  local_approx  = TRUE, n_threads = 1,
  seed = sample(.Machine$integer.max, size = 1), max_iter = 100, conv_tol = 1e-8,...) {
  
  a <- proc.time()
  check_target(target_acceptance)
  type <- match.arg(type, c("full", "summary"))
  method <- match.arg(method, c("pm", "da", paste0("is", 1:3)))
  simulation_method <- match.arg(simulation_method, c("psi", "bsf", "spdk"))
  
  
  if (nsim_states < 2) {
    #approximate inference
    method <- "is2"
  }
  
  if (missing(S)) {
    S <- diag(0.1 * pmax(0.1, abs(object$theta)), length(object$theta))
  }
  
  
  out <-  switch(type,
    full = {
      if (method == "da"){
        out <- nongaussian_da_mcmc(object,
          nsim_states, n_iter, n_burnin, n_thin, gamma, target_acceptance, S,
          seed, end_adaptive_phase, n_threads, local_approx, object$initial_mode,
          max_iter, conv_tol, pmatch(simulation_method, c("psi", "bsf", "spdk")),
          model_type = 3L, 0, 0, 0)
      } else {
        if (method == "pm") {
          out <- nongaussian_pm_mcmc(object, 
            nsim_states, n_iter, n_burnin, n_thin, gamma, target_acceptance, S,
            seed, end_adaptive_phase, n_threads, local_approx, object$initial_mode,
            max_iter, conv_tol, pmatch(simulation_method, c("psi", "bsf", "spdk")),
            model_type = 3L, 0, 0, 0)
        } else {
          out <- nongaussian_is_mcmc(object, 
            nsim_states, n_iter, n_burnin, n_thin, gamma, target_acceptance, S,
            seed, end_adaptive_phase, n_threads, local_approx, object$initial_mode,
            max_iter, conv_tol, pmatch(simulation_method, c("psi", "bsf", "spdk")),
            pmatch(method, paste0("is", 1:3)),
            model_type = 3L, 0, 0, 0)
        }
      }
      
      colnames(out$alpha) <- names(object$a1)
      out
    },
    summary = {
      stop("summary for SV models not yet implemented.")
    })
  
  colnames(out$theta) <- rownames(out$S) <- colnames(out$S) <- names(object$theta)
  
  out$n_iter <- n_iter
  out$n_burnin <- n_burnin
  out$n_thin <- n_thin
  out$mcmc_type <- method
  
  out$call <- match.call()
  out$seed <- seed
  
  out$time <- proc.time() - a
  class(out) <- "mcmc_output"
  attr(out, "model_type") <- "svm"
  out
}


#' @method run_mcmc nlg_ssm
#' @rdname run_mcmc_ng
#' @export
run_mcmc.nlg_ssm <-  function(object, n_iter, nsim_states, type = "full",
  method = "da", simulation_method = "psi",
  n_burnin = floor(n_iter/2), n_thin = 1,
  gamma = 2/3, target_acceptance = 0.234, S, end_adaptive_phase = TRUE,
  n_threads = 1, seed = sample(.Machine$integer.max, size = 1), max_iter = 100,
  conv_tol = 1e-4, iekf_iter = 0, ...) {
  
  a <- proc.time()
  check_target(target_acceptance)
  
  type <- match.arg(type, c("full", "summary"))
  method <- match.arg(method, c("pm", "da", paste0("is", 1:3), "ekf"))
  simulation_method <- match.arg(simulation_method, c("psi", "bsf", "spdk"))
  if(simulation_method == "spdk") {
    stop("SPDK is (currently) not supported for non-linear non-Gaussian models.")
  }
  
  if (missing(S)) {
    S <- diag(0.1 * pmax(0.1, abs(object$theta)), length(object$theta))
  }
  
  out <-  switch(type,
    full = {
      if (method == "da"){
        out <- nonlinear_da_mcmc(t(object$y), object$Z, object$H, object$T,
          object$R, object$Z_gn, object$T_gn, object$a1, object$P1,
          object$theta, object$log_prior_pdf, object$known_params,
          object$known_tv_params, as.integer(object$time_varying),
          object$n_states, object$n_etas, seed,
          nsim_states, n_iter, n_burnin, n_thin, gamma, target_acceptance, S,
          end_adaptive_phase, n_threads,
          max_iter, conv_tol,
          pmatch(simulation_method, c("psi", "bsf", "spdk")), iekf_iter)
      } else {
        if (method == "pm") {
          out <- nonlinear_pm_mcmc(t(object$y), object$Z, object$H, object$T,
            object$R, object$Z_gn, object$T_gn, object$a1, object$P1,
            object$theta, object$log_prior_pdf, object$known_params,
            object$known_tv_params, as.integer(object$time_varying),
            object$n_states, object$n_etas, seed,
            nsim_states, n_iter, n_burnin, n_thin, gamma, target_acceptance, S,
            end_adaptive_phase, n_threads,
            max_iter, conv_tol,
            pmatch(simulation_method, c("psi", "bsf", "spdk")), iekf_iter)
        } else {
          if(method == "ekf") {
            out <- nonlinear_ekf_mcmc(t(object$y), object$Z, object$H, object$T,
              object$R, object$Z_gn, object$T_gn, object$a1, object$P1,
              object$theta, object$log_prior_pdf, object$known_params,
              object$known_tv_params, as.integer(object$time_varying),
              object$n_states, object$n_etas, seed,
              n_iter, n_burnin, n_thin, gamma, target_acceptance, S,
              end_adaptive_phase,  n_threads, iekf_iter, FALSE)
          } else {
            out <- nonlinear_is_mcmc(t(object$y), object$Z, object$H, object$T,
              object$R, object$Z_gn, object$T_gn, object$a1, object$P1,
              object$theta, object$log_prior_pdf, object$known_params,
              object$known_tv_params, as.integer(object$time_varying),
              object$n_states, object$n_etas, seed,
              nsim_states, n_iter, n_burnin, n_thin, gamma, target_acceptance, S,
              end_adaptive_phase, n_threads, pmatch(method, paste0("is", 1:3)),
              pmatch(simulation_method, c("psi", "bsf", "spdk")),
              max_iter, conv_tol, iekf_iter)
          }
        }
      }
      
      colnames(out$alpha) <- object$state_names
      out
    },
    summary = {
      if (method != "ekf") {
        stop("summary MCMC for non-linear models is only implemented for EKF approximation.")
      } else {
        out <- nonlinear_ekf_mcmc(t(object$y), object$Z, object$H, object$T,
          object$R, object$Z_gn, object$T_gn, object$a1, object$P1,
          object$theta, object$log_prior_pdf, object$known_params,
          object$known_tv_params, as.integer(object$time_varying),
          object$n_states, object$n_etas, seed,
          n_iter, n_burnin, n_thin, gamma, target_acceptance, S,
          end_adaptive_phase, n_threads, iekf_iter, TRUE)
        colnames(out$alpha) <- object$state_names
        out
      }
    })
  
  colnames(out$theta) <- rownames(out$S) <- colnames(out$S) <- names(object$theta)
  
  out$n_iter <- n_iter
  out$n_burnin <- n_burnin
  out$n_thin <- n_thin
  out$mcmc_type <- method
  out$call <- match.call()
  out$seed <- seed
  out$time <- proc.time() - a
  class(out) <- "mcmc_output"
  attr(out, "model_type") <- "nlg_ssm"
  out
}

#' @method run_mcmc sde_ssm
#' @rdname run_mcmc_ng
#' @param L_c,L_f Integer values defining the discretization levels for first and second stages. 
#' For PM methods, maximum of these is used.
#' @export
run_mcmc.sde_ssm <-  function(object, n_iter, nsim_states, type = "full",
  method = "da", L_c, L_f,
  n_burnin = floor(n_iter/2), n_thin = 1,
  gamma = 2/3, target_acceptance = 0.234, S, end_adaptive_phase = TRUE,
  n_threads = 1, seed = sample(.Machine$integer.max, size = 1), ...) {
  
  if(any(c(object$drift, object$diffusion, object$ddiffusion,
    object$prior_pdf, object$obs_pdf) %in% c("<pointer: (nil)>", "<pointer: 0x0>"))) {
    stop("NULL pointer detected, please recompile the pointer file and reconstruct the model.")
  }
  
  a <- proc.time()
  check_target(target_acceptance)
  if(nsim_states <= 0) stop("nsim_states should be positive integer.")
  
  type <- match.arg(type, c("full", "summary"))
  method <- match.arg(method, c("pm", "da", paste0("is", 1:3)))
  
  if (missing(S)) {
    S <- diag(0.1 * pmax(0.1, abs(object$theta)), length(object$theta))
  }
  
  out <-  switch(type,
    full = {
      if (method == "da"){
        if (L_f <= L_c) stop("L_f should be larger than L_c.")
        if(L_c < 1) stop("L_c should be at least 1")
        out <- sde_da_mcmc(object$y, object$x0, object$positive,
          object$drift, object$diffusion, object$ddiffusion,
          object$prior_pdf, object$obs_pdf, object$theta,
          nsim_states, L_c, L_f, seed,
          n_iter, n_burnin, n_thin, gamma, target_acceptance, S,
          end_adaptive_phase)
      } else {
        if(method == "pm") {
          if (missing(L_c)) L_c <- 0
          if (missing(L_f)) L_f <- 0
          L <- max(L_c, L_f)
          if(L <= 0) stop("L should be positive.")
          out <- sde_pm_mcmc(object$y, object$x0, object$positive,
            object$drift, object$diffusion, object$ddiffusion,
            object$prior_pdf, object$obs_pdf, object$theta,
            nsim_states, L, seed,
            n_iter, n_burnin, n_thin, gamma, target_acceptance, S,
            end_adaptive_phase)
        } else {
          if (L_f <= L_c) stop("L_f should be larger than L_c.")
          if(L_c < 1) stop("L_c should be at least 1")
          
          out <- sde_is_mcmc(object$y, object$x0, object$positive,
            object$drift, object$diffusion, object$ddiffusion,
            object$prior_pdf, object$obs_pdf, object$theta,
            nsim_states, L_c, L_f, seed,
            n_iter, n_burnin, n_thin, gamma, target_acceptance, S,
            end_adaptive_phase, pmatch(method, paste0("is", 1:3)), n_threads)
        }
      }
      colnames(out$alpha) <- object$state_names
      out
    },
    summary = {
      stop("summary MCMC not implemented for SDE models.")
    })
  
  colnames(out$theta) <- rownames(out$S) <- colnames(out$S) <- names(object$theta)
  
  out$n_iter <- n_iter
  out$n_burnin <- n_burnin
  out$n_thin <- n_thin
  out$mcmc_type <- method
  out$call <- match.call()
  out$seed <- seed
  out$time <- proc.time() - a
  class(out) <- "mcmc_output"
  attr(out, "model_type") <- "sde_ssm"
  out
}


#' @method run_mcmc lgg_ssm
#' @rdname run_mcmc_g
#' @inheritParams run_mcmc.gssm
#' @export
run_mcmc.lgg_ssm <- function(object, n_iter, sim_states = TRUE, type = "full",
  n_burnin = floor(n_iter/2), n_thin = 1, gamma = 2/3,
  target_acceptance = 0.234, S, end_adaptive_phase = TRUE,
  n_threads = 1, seed = sample(.Machine$integer.max, size = 1), ...) {
  
  if(any(c(object$Z, object$H, object$T,
    object$R, object$a1, object$P1,
    object$theta, object$obs_intercept, object$state_intercept,
    object$log_prior_pdf) %in% c("<pointer: (nil)>", "<pointer: 0x0>"))) {
    stop("NULL pointer detected, please recompile the pointer file and reconstruct the model.")
  }
  
  a <- proc.time()
  check_target(target_acceptance)
  
  type <- match.arg(type, c("full", "summary"))
  
  if (missing(S)) {
    S <- diag(0.1 * pmax(0.1, abs(object$theta)), length(object$theta))
  }
  
  out <- switch(type,
    full = {
      out <- general_gaussian_mcmc(t(object$y), object$Z, object$H, object$T,
        object$R, object$a1, object$P1,
        object$theta, object$obs_intercept, object$state_intercept,
        object$log_prior_pdf, object$known_params,
        object$known_tv_params, as.integer(object$time_varying), 
        object$n_states, object$n_etas, seed,
        n_iter, n_burnin, n_thin, gamma, target_acceptance, S,
        end_adaptive_phase, n_threads, sim_states)
      
      if (sim_states) {
        colnames(out$alpha) <- object$state_names
      }
      out
    },
    summary = {
      stop("Summary MCMC method not yet implemented.")
    })
  
  colnames(out$theta) <- rownames(out$S) <- colnames(out$S) <- names(object$theta)
  
  out$call <- match.call()
  out$seed <- seed
  out$n_iter <- n_iter
  out$n_burnin <- n_burnin
  out$n_thin <- n_thin
  out$mcmc_type <- "gaussian_mcmc"
  out$time <- proc.time() - a
  class(out) <- "mcmc_output"
  attr(out, "model_type") <- "lgg_ssm"
  out
}
