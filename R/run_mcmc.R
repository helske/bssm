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
#' @param type Type of output. Default is \code{"full"}, which returns
#' samples from the posterior \eqn{p(\alpha, \theta}. Option
#' \code{"parameters"} samples only parameters \eqn{\theta} (which includes the
#' regression coefficients \eqn{\beta}). This can be used for faster inference of
#' \eqn{\theta} only, or as an preliminary run for obtaining
#' initial values for \code{S}. Option \code{"summary"} does not simulate
#' states directly but computes the posterior means and variances of states using
#' fast Kalman smoothing. This is slightly faster, memory  efficient and
#' more accurate than calculations based on simulation smoother.
#' \eqn{\theta}. Optional for \code{bsm} objects.
#' @param n_burnin Length of the burn-in period which is disregarded from the
#' results. Defaults to \code{n_iter / 2}.
#' @param n_thin Thinning rate. All MCMC algoritms in \code{bssm} use the jump chain 
#' representation, and the thinning is applied to these blocks. 
#' This defaults to 1, but for IS-corrected method (\code{method="isc"}), larger 
#' value is often more effective.
#' @param gamma Tuning parameter for the adaptation of RAM algorithm. Must be
#' between 0 and 1 (not checked).
#' @param target_acceptance Target acceptance ratio for RAM. Defaults to 0.234.
#' @param S Initial value for the lower triangular matrix of RAM
#' algorithm, so that the covariance matrix of the Gaussian proposal
#' distribution is \eqn{SS'}.
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
  
  inits <- sapply(object$priors, "[[", "init")
  
  if (missing(S)) {
    S <- diag(0.1 * pmax(0.1, abs(inits)), length(inits))
  }
  priors <- combine_priors(object$priors)
  
  out <- switch(type,
    full = {
      out <- gaussian_mcmc(object, priors$prior_type, priors$params, sim_states, 
        n_iter, n_burnin, n_thin, gamma, target_acceptance, S, seed, 
        end_adaptive_phase, n_threads, model_type = 1L,
        object$Z_ind, object$H_ind, object$T_ind, object$R_ind)
      if (sim_states) {
        colnames(out$alpha) <- names(object$a1)
      }
      out
    },
    summary = {
      out <- gaussian_mcmc_summary(object, priors$prior_type, priors$params,
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
  out$isc <- FALSE
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
  
  if (missing(S)) {
    S <- diag(0.1 * pmax(0.1, abs(sapply(object$priors, "[[", "init"))), length(object$priors))
  }
  
  priors <- combine_priors(object$priors)
  
  out <- switch(type,
    full = {
      out <- gaussian_mcmc(object, priors$prior_type, priors$params, sim_states, 
        n_iter, n_burnin, n_thin, gamma, target_acceptance, S, seed, 
        end_adaptive_phase, n_threads, model_type = 2L, 0, 0, 0, 0)
      
      if (sim_states) {
        colnames(out$alpha) <- names(object$a1)
      }
      out
    },
    summary = {
      out <- gaussian_mcmc_summary(object, priors$prior_type, priors$params,
        n_iter, n_burnin, n_thin, gamma, target_acceptance, S, seed, 
        end_adaptive_phase, n_threads, model_type = 2L, 0, 0, 0, 0)
      colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <-
        names(object$a1)
      out$alphahat <- ts(out$alphahat, start = start(object$y),
        frequency = frequency(object$y))
      out
    })
  
  names_ind <- !object$fixed & c(TRUE, TRUE, object$slope, object$seasonal)
  colnames(out$theta) <- rownames(out$S) <- colnames(out$S) <-
    c(c("sd_y", "sd_level", "sd_slope", "sd_seasonal")[names_ind],
      colnames(object$xreg))
  
  out$call <- match.call()
  out$seed <- seed
  out$n_iter <- n_iter
  out$n_burnin <- n_burnin
  out$n_thin <- n_thin
  out$isc <- FALSE
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
#' @param type Either \code{"full"} (default), or \code{"summary"}. The
#' former produces samples of states whereas the latter gives the mean and
#' variance estimates of the states.
#' @param method Whether pseudo-marginal MCMC (\code{"pm"}) (default) or
#' importance sampling type correction (\code{"isc"}) is used.
#' @param simulation_method If \code{"spdk"}, non-sequential importance sampling based
#' on Gaussian approximation is used. If \code{"bootstrap"}, bootstrap filter
#' is used, and if \code{"psi"}, psi-auxiliary particle filter is used.
#' @param const_m For importance sampling correction method, should a constant number of 
#' samples be used for each block? Default is \code{TRUE}. See references for details.
#' @param delayed_acceptance For pseudo-marginal MCMC, should delayed acceptance based
#' on the Gaussian approximation be used?
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
  method = "pm", simulation_method = "psi", const_m = TRUE,
  delayed_acceptance = TRUE, n_burnin = floor(n_iter/2),
  n_thin = 1, gamma = 2/3, target_acceptance = 0.234, S, end_adaptive_phase = TRUE,
  local_approx  = TRUE, n_threads = 1,
  seed = sample(.Machine$integer.max, size = 1), ...) {
  
  a <- proc.time()
  check_target(target_acceptance)
  
  type <- match.arg(type, c("full", "summary"))
  method <- match.arg(method, c("pm", "isc"))
  simulation_method <- match.arg(simulation_method, c("psi", "bsf", "spdk"))
  
  if (nsim_states < 2) {
    #approximate inference
    method <- "pm"
    simulation_method <- "spdk"
  }
  
  inits <- sapply(object$priors, "[[", "init")
  
  if (missing(S)) {
    S <- diag(0.1 * pmax(0.1, abs(inits)), length(inits))
  }
  priors <- combine_priors(object$priors)
  
  object$distribution <- pmatch(object$distribution,
    c("poisson", "binomial", "negative binomial"))
  
  out <-  switch(type,
    full = {
      if (method == "pm"){
        if (delayed_acceptance) {
          out <- nongaussian_da_mcmc(object, priors$prior_types, priors$params, 
            nsim_states, n_iter, n_burnin, n_thin, gamma, target_acceptance, S,
            seed, end_adaptive_phase, n_threads, local_approx, object$initial_mode, 
            max_iter, conv_tol, pmatch(simulation_method, c("psi", "bsf", "spdk")), 
            model_type = 1L, object$Z_ind, object$T_ind, object$R_ind)
        } else {
          out <- nongaussian_pm_mcmc(object, priors$prior_types, priors$params, 
            nsim_states, n_iter, n_burnin, n_thin, gamma, target_acceptance, S,
            seed, end_adaptive_phase, n_threads, local_approx, object$initial_mode, 
            max_iter, conv_tol, pmatch(simulation_method, c("psi", "bsf", "spdk")), 
            model_type = 1L, object$Z_ind, object$T_ind, object$R_ind)
        }
      } else {
        out <- nongaussian_is_mcmc(object, priors$prior_types, priors$params, 
          nsim_states, n_iter, n_burnin, n_thin, gamma, target_acceptance, S,
          seed, end_adaptive_phase, n_threads, local_approx, object$initial_mode, 
          max_iter, conv_tol, pmatch(simulation_method, c("psi", "bsf", "spdk")), const_m, 
          model_type = 1L, object$Z_ind, object$T_ind, object$R_ind)
      }
      
      colnames(out$alpha) <- names(object$a1)
      out
    },
    summary = {
      stop("summary correction for general models is not yet implemented.")
    })
  
  out$n_iter <- n_iter
  out$n_burnin <- n_burnin
  out$n_thin <- n_thin
  out$isc <- method == "isc"
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
  method = "pm", simulation_method = "psi", const_m = TRUE,
  delayed_acceptance = TRUE, n_burnin = floor(n_iter/2), n_thin = 1,
  gamma = 2/3, target_acceptance = 0.234, S, end_adaptive_phase = TRUE,
  local_approx  = TRUE, n_threads = 1,
  seed = sample(.Machine$integer.max, size = 1), max_iter = 100, conv_tol = 1e-8, ...) {
  
  a <- proc.time()
  check_target(target_acceptance)
  
  type <- match.arg(type, c("full", "summary"))
  method <- match.arg(method, c("pm", "isc"))
  simulation_method <- match.arg(simulation_method, c("psi", "bsf", "spdk"))
  
  
  if (nsim_states < 2) {
    #approximate inference
    method <- "isc"
    nsim_states <- 0
  }
  
  if (missing(S)) {
    S <- diag(0.1 * pmax(0.1, abs(sapply(object$priors, "[[", "init"))), 
      length(object$priors))
  }
  
  priors <- combine_priors(object$priors)
  
  object$distribution <- pmatch(object$distribution, 
    c("poisson", "binomial", "negative binomial"))
  
  out <-  switch(type,
    full = {
      if (method == "pm"){
        if (delayed_acceptance) {
          out <- nongaussian_da_mcmc(object, priors$prior_types, priors$params, 
            nsim_states, n_iter, n_burnin, n_thin, gamma, target_acceptance, S,
            seed, end_adaptive_phase, n_threads, local_approx, object$initial_mode, 
            max_iter, conv_tol, pmatch(simulation_method, c("psi", "bsf", "spdk")), 
            model_type = 2L, 0, 0, 0)
        } else {
          out <- nongaussian_pm_mcmc(object, priors$prior_types, priors$params, 
            nsim_states, n_iter, n_burnin, n_thin, gamma, target_acceptance, S,
            seed, end_adaptive_phase, n_threads, local_approx, object$initial_mode, 
            max_iter, conv_tol, pmatch(simulation_method, c("psi", "bsf", "spdk")), 
            model_type = 2L, 0, 0, 0)
        }
      } else {
        out <- nongaussian_is_mcmc(object, priors$prior_types, priors$params, 
          nsim_states, n_iter, n_burnin, n_thin, gamma, target_acceptance, S,
          seed, end_adaptive_phase, n_threads, local_approx, object$initial_mode, 
          max_iter, conv_tol, pmatch(simulation_method, c("psi", "bsf", "spdk")), const_m, 
          model_type = 2L, 0, 0, 0)
      }
      
      colnames(out$alpha) <- names(object$a1)
      out
    },
    summary = {
      stop("summary not yet re-implemented.")
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
      #     pmatch(simulation_method, c("psi", "bsf", "spdk")), const_m, 
      #     sample(.Machine$integer.max, size = n_threads))
      # }
      colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <- names(object$a1)
      out$alphahat <- ts(out$alphahat, start = start(object$y), frequency = frequency(object$y))
      out$muhat <- ts(out$muhat, start = start(object$y), frequency = frequency(object$y))
      out
    })
  
  names_ind <-
    c(!object$fixed & c(TRUE, object$slope, object$seasonal), object$noise)
  colnames(out$theta) <- rownames(out$S) <- colnames(out$S) <-
    c(c("sd_level", "sd_slope", "sd_seasonal", "sd_noise")[names_ind],
      colnames(object$xreg), 
      if (object$distribution == "negative binomial") "nb_dispersion")
  out$n_iter <- n_iter
  out$n_burnin <- n_burnin
  out$n_thin <- n_thin
  out$isc <- method == "isc"
  out$call <- match.call()
  out$seed <- seed
  out$time <- proc.time() - a
  class(out) <- "mcmc_output"
  attr(out, "model_type") <- "ng_bsm"
  out
}

#' @method run_mcmc svm
#' @rdname run_mcmc_ng
#' @inheritParams run_mcmc.ngssm
#' @export
#'  
run_mcmc.svm <-  function(object, n_iter, nsim_states, type = "full",
  method = "pm", simulation_method = "psi", const_m = TRUE,
  delayed_acceptance = TRUE, n_burnin = floor(n_iter/2),
  n_thin = 1, gamma = 2/3, target_acceptance = 0.234, S, end_adaptive_phase = TRUE,
  local_approx  = TRUE, n_threads = 1,
  seed = sample(.Machine$integer.max, size = 1), max_iter = 100, conv_tol = 1e-8,...) {
  
  a <- proc.time()
  check_target(target_acceptance)
  type <- match.arg(type, c("full", "summary"))
  method <- match.arg(method, c("pm", "isc"))
  simulation_method <- match.arg(simulation_method, c("psi", "bsf", "spdk"))
  
  
  if (nsim_states < 2) {
    #approximate inference
    method <- "pm"
    simulation_method <- "spdk"
  }
  
  if (missing(S)) {
    inits <- abs(sapply(object$priors, "[[", "init"))
    S <- diag(0.1 * pmax(0.1, inits), length(inits))
  }
  
  priors <- combine_priors(object$priors)
  
  out <-  switch(type,
    full = {
      if (method == "pm"){
        if (delayed_acceptance) {
          out <- nongaussian_da_mcmc(object, priors$prior_types, priors$params, 
            nsim_states, n_iter, n_burnin, n_thin, gamma, target_acceptance, S,
            seed, end_adaptive_phase, n_threads, local_approx, object$initial_mode, 
            max_iter, conv_tol, pmatch(simulation_method, c("psi", "bsf", "spdk")), 
            model_type = 3L, 0, 0, 0)
        } else {
          out <- nongaussian_pm_mcmc(object, priors$prior_types, priors$params, 
            nsim_states, n_iter, n_burnin, n_thin, gamma, target_acceptance, S,
            seed, end_adaptive_phase, n_threads, local_approx, object$initial_mode, 
            max_iter, conv_tol, pmatch(simulation_method, c("psi", "bsf", "spdk")), 
            model_type = 3L, 0, 0, 0)
        }
      } else {
        out <- nongaussian_is_mcmc(object, priors$prior_types, priors$params, 
          nsim_states, n_iter, n_burnin, n_thin, gamma, target_acceptance, S,
          seed, end_adaptive_phase, n_threads, local_approx, object$initial_mode, 
          max_iter, conv_tol, pmatch(simulation_method, c("psi", "bsf", "spdk")), const_m, 
          model_type = 3L, 0, 0, 0)
      }
      
      colnames(out$alpha) <- names(object$a1)
      out
    },
    summary = {
      stop("summary for SV models not yet implemented.")
    })
  
  colnames(out$theta) <- rownames(out$S) <- colnames(out$S) <-
    c(names(object$priors), names(object$coefs))
  
  out$n_iter <- n_iter
  out$n_burnin <- n_burnin
  out$n_thin <- n_thin
  out$isc <- method == "isc"
  
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
  method = "pm", simulation_method = "psi", const_m = TRUE,
  delayed_acceptance = TRUE, n_burnin = floor(n_iter/2), n_thin = 1,
  gamma = 2/3, target_acceptance = 0.234, S, end_adaptive_phase = TRUE,
  n_threads = 1, seed = sample(.Machine$integer.max, size = 1), max_iter = 100, 
  conv_tol = 1e-8, iekf_iter = 0, ...) {
  
  a <- proc.time()
  check_target(target_acceptance)
  
  type <- match.arg(type, c("full", "summary"))
  method <- match.arg(method, c("pm", "isc", "ekf"))
  simulation_method <- match.arg(simulation_method, c("psi", "bsf", "spdk"))
  if(simulation_method == "spdk") {
    stop("SPDK is currently not supported for non-linear non-Gaussian models.")
  }
  if(method == "ekf") {
    nsim_states <- 1
  }
  if (nsim_states < 2 && method != "ekf") {
    #approximate inference
    method <- "ekf"
  }
  
  if (missing(S)) {
    S <- diag(0.1 * pmax(0.1, abs(object$theta)), length(object$theta))
  }

  out <-  switch(type,
    full = {
      if (method == "pm"){
        if (delayed_acceptance) {
          out <- nonlinear_da_mcmc(t(object$y), object$Z, object$H, object$T, 
            object$R, object$Z_gn, object$T_gn, object$a1, object$P1, 
            object$theta, object$log_prior_pdf, object$known_params, 
            object$known_tv_params, as.integer(object$time_varying), 
            as.integer(object$state_varying),
            object$n_states, object$n_etas, seed, 
            nsim_states, n_iter, n_burnin, n_thin, gamma, target_acceptance, S,
            end_adaptive_phase, n_threads,
            max_iter, conv_tol, 
            pmatch(simulation_method, c("psi", "bsf", "spdk")), iekf_iter)
        } else {
          out <- nonlinear_pm_mcmc(t(object$y), object$Z, object$H, object$T, 
            object$R, object$Z_gn, object$T_gn, object$a1, object$P1, 
            object$theta, object$log_prior_pdf, object$known_params, 
            object$known_tv_params, as.integer(object$time_varying), 
            as.integer(object$state_varying), object$n_states, object$n_etas, seed, 
            nsim_states, n_iter, n_burnin, n_thin, gamma, target_acceptance, S,
            end_adaptive_phase, n_threads,
            max_iter, conv_tol, 
            pmatch(simulation_method, c("psi", "bsf", "spdk")), iekf_iter)
        }
      } else {
        if(method == "ekf") {
          out <- nonlinear_ekf_mcmc(t(object$y), object$Z, object$H, object$T, 
            object$R, object$Z_gn, object$T_gn, object$a1, object$P1, 
            object$theta, object$log_prior_pdf, object$known_params, 
            object$known_tv_params, as.integer(object$time_varying), 
            as.integer(object$state_varying), object$n_states, object$n_etas, seed, 
            nsim_states, n_iter, n_burnin, n_thin, gamma, target_acceptance, S,
            end_adaptive_phase, max_iter, conv_tol, n_threads, iekf_iter)
        } else {
          out <- nonlinear_is_mcmc(t(object$y), object$Z, object$H, object$T, 
            object$R, object$Z_gn, object$T_gn, object$a1, object$P1, 
            object$theta, object$log_prior_pdf, object$known_params, 
            object$known_tv_params, as.integer(object$time_varying), 
            as.integer(object$state_varying), object$n_states, object$n_etas, seed, 
            nsim_states, n_iter, n_burnin, n_thin, gamma, target_acceptance, S,
            end_adaptive_phase, n_threads, const_m, 
            pmatch(simulation_method, c("psi", "bsf", "spdk")), 
            max_iter, conv_tol, iekf_iter)
        }
      }
      
      colnames(out$alpha) <- object$state_names
      out
    },
    summary = {
      stop("summary MCMC not implemented for non-linear models.")
      
    })
  
  colnames(out$theta) <- rownames(out$S) <- colnames(out$S) <- names(object$theta)
  
  out$n_iter <- n_iter
  out$n_burnin <- n_burnin
  out$n_thin <- n_thin
  out$isc <- method == "isc"
  out$call <- match.call()
  out$seed <- seed
  out$time <- proc.time() - a
  class(out) <- "mcmc_output"
  attr(out, "model_type") <- "nlg_ssm"
  out
}
