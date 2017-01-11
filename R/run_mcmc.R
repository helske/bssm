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
#' states directly computes the  posterior means and variances of states using
#' fast Kalman smoothing. This is slightly faster, memory  efficient and
#' more accurate than calculations based on simulation smoother.
#' \eqn{\theta}. Optional for \code{bsm} objects.
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
#' @param seed Seed for the random number generator.
#' @param ... Ignored.
#' @export
run_mcmc.gssm <- function(object, n_iter, sim_states = TRUE, type = "full", 
  n_burnin = floor(n_iter / 2), n_thin = 1, gamma = 2/3,
  target_acceptance = 0.234, S, end_adaptive_phase = TRUE,
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
      out <- gssm_run_mcmc(object, priors$prior_types, priors$params, n_iter,
        sim_states, n_burnin, n_thin, gamma, target_acceptance, S, 
        seed, end_adaptive_phase, object$Z_ind,
        object$H_ind, object$T_ind, object$R_ind)
      if (sim_states) {
        colnames(out$alpha) <- names(object$a1)
      }
      out
    },
    summary = {
      out <- gssm_run_mcmc_summary(object, priors$prior_types, priors$params, n_iter,
        n_burnin, n_thin, gamma, target_acceptance, S, seed, end_adaptive_phase, 
        object$Z_ind, object$H_ind, object$T_ind, object$R_ind)
      colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <-
        names(object$a1)
      out$alphahat <- ts(out$alphahat, start = start(object$y),
        frequency = frequency(object$y))
      out
    }
  )
  out$theta <- mcmc(out$theta, start = n_burnin + 1, thin = n_thin)
  out$call <- match.call()
  out$seed <- seed
  out$jump_chain <- FALSE
  out$time <- proc.time() - a
  class(out) <- "mcmc_output"
  out
}  

#' @method run_mcmc bsm
#' @rdname run_mcmc_g
#' @inheritParams run_mcmc.gssm
#' @export
run_mcmc.bsm <- function(object, n_iter, sim_states = TRUE, type = "full",
  n_burnin = floor(n_iter/2), n_thin = 1, gamma = 2/3,
  target_acceptance = 0.234, S, end_adaptive_phase = TRUE,
  seed = sample(.Machine$integer.max, size = 1), ...) {
  
  a <- proc.time()
  check_target(target_acceptance)
  
  type <- match.arg(type, c("full", "summary"))
  
  if (missing(S)) {
    S <- diag(0.1 * pmax(0.1, abs(sapply(object$priors, "[[", "init"))), length(object$priors))
  }
  
  priors <- combine_priors(object$priors)
  
  out <- switch(type,
    full = {
      out <- bsm_run_mcmc(object, priors$prior_type, priors$params, n_iter,
        sim_states, n_burnin, n_thin, gamma, target_acceptance, S, seed, 
        FALSE, end_adaptive_phase)
      
      if (sim_states) {
        colnames(out$alpha) <- names(object$a1)
      }
      out
    },
    summary = {
      out <- bsm_run_mcmc_summary(object, priors$prior_type, priors$params, 
        n_iter, n_burnin, n_thin, gamma, target_acceptance, S, seed,
        FALSE, end_adaptive_phase)
      
      colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <- names(object$a1)
      out$alphahat <- ts(out$alphahat, start = start(object$y),
        frequency = object$period)
      out
    })
  
  names_ind <- !object$fixed & c(TRUE, TRUE, object$slope, object$seasonal)
  colnames(out$theta) <- rownames(out$S) <- colnames(out$S) <-
    c(c("sd_y", "sd_level", "sd_slope", "sd_seasonal")[names_ind],
      colnames(object$xreg))
  out$theta <- mcmc(out$theta, start = n_burnin + 1, thin = n_thin)
  out$call <- match.call()
  out$seed <- seed
  out$jump_chain <- FALSE
  out$time <- proc.time() - a
  class(out) <- "mcmc_output"
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
#' @param adaptive_approx If \code{TRUE} (default), Gaussian approximation needed for 
#' importance sampling is performed at each iteration. If false, approximation is updated only 
#' once at the start of the MCMC.
#' @param n_threads Number of threads for state simulation.
#' @param seed Seed for the random number generator.
#' @param ... Ignored.
#' @export
run_mcmc.ngssm <- function(object, n_iter, nsim_states, type = "full",
  method = "pm", simulation_method = "spdk", const_m = TRUE,
  delayed_acceptance = TRUE, n_burnin = floor(n_iter/2),
  n_thin = 1, gamma = 2/3, target_acceptance = 0.234, S, end_adaptive_phase = TRUE,
  adaptive_approx  = TRUE, n_threads = 1,
  seed = sample(.Machine$integer.max, size = 1), ...) {
  
  set.seed(seed) #needed for reproducible parallel thread seeding
  
  a <- proc.time()
  check_target(target_acceptance)
  nb <- object$distribution == "negative binomial"
  
  type <- match.arg(type, c("full", "summary"))
  method <- match.arg(method, c("pm", "isc"))
  simulation_method <- match.arg(simulation_method, c("spdk", "bootstrap", "psi"))
  
  if (n_thin > 1 && method == "isc") {
    stop ("Thinning with block-IS algorithm is not supported.")
  }
  
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
        out <- ngssm_run_mcmc(object, priors$prior_types, priors$params, n_iter,
          nsim_states, n_burnin, n_thin, gamma, target_acceptance, S,
          object$init_signal, seed, end_adaptive_phase, adaptive_approx,
          delayed_acceptance, pmatch(simulation_method, c("spdk", "bootstrap", "psi")),
          object$Z_ind, object$T_ind, object$R_ind)
        
      } else {
        out <- ngssm_run_mcmc_is(object, priors$prior_types, priors$params, n_iter,
          nsim_states, n_burnin, n_thin, gamma, target_acceptance, S,
          object$init_signal, seed, n_threads, end_adaptive_phase, adaptive_approx,
          pmatch(simulation_method, c("spdk", "bootstrap", "psi")), const_m,
          object$Z_ind, object$T_ind, object$R_ind, 
          sample(.Machine$integer.max, size = n_threads))
      }
      
      colnames(out$alpha) <- names(object$a1)
      out
    },
    summary = {
      stop("summary correction for general models is not yet implemented.")
      
      # if (method == "pm"){
      #   out <- ngssm_run_mcmc_summary(object, priors$prior_types, priors$params, n_iter,
      #     nsim_states, n_burnin, n_thin, gamma, target_acceptance, S,
      #     init_signal, seed,  n_threads, end_adaptive_phase, adaptive_approx,
      #     delayed_acceptance, pmatch(simulation_method, c("spdk", "bootstrap", "psi")),
      #     Z_ind, T_ind, R_ind)
      # } else {
      #   if(correction_method == "PF") {
      #     stop("summary correction with particle filter is not yet implemented.")
      #   }
      #   out <- ngssm_run_mcmc_summary_is(object, priors$prior_types, priors$params, n_iter,
      #     nsim_states, n_burnin, n_thin, gamma, target_acceptance, S,
      #     init_signal, seed,  n_threads, end_adaptive_phase, adaptive_approx,
      #     pmatch(simulation_method, c("spdk", "bootstrap", "psi")), const_m, Z_ind, T_ind, R_ind)
      # }
      # colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <- names(object$a1)
      # out$alphahat <- ts(out$alphahat, start = start(object$y), frequency = frequency(object$y))
      # out$muhat <- ts(out$muhat, start = start(object$y), frequency = frequency(object$y))
      # out
    })
  
  if(method == "pm") {
    out$theta <- mcmc(out$theta, start = n_burnin + 1, thin = n_thin)
    out$jump_chain <- FALSE
  } else {
    out$jump_chain <- TRUE
    out$n_iter <- n_iter
    out$n_burnin <- n_burnin
  }
  out$call <- match.call()
  out$seed <- seed
  out$time <- proc.time() - a
  class(out) <- "mcmc_output"
  out
}


#' @method run_mcmc ng_bsm
#' @rdname run_mcmc_ng
#' @export
run_mcmc.ng_bsm <-  function(object, n_iter, nsim_states, type = "full",
  method = "pm", simulation_method = "spdk", const_m = TRUE,
  delayed_acceptance = TRUE, n_burnin = floor(n_iter/2),
  n_thin = 1, gamma = 2/3, target_acceptance = 0.234, S, end_adaptive_phase = TRUE,
  adaptive_approx  = TRUE, n_threads = 1,
  seed = sample(.Machine$integer.max, size = 1), noise_const...) {
  
  set.seed(seed) #needed for reproducible parallel thread seeding
  
  a <- proc.time()
  check_target(target_acceptance)
  nb <- object$distribution == "negative binomial"
  
  type <- match.arg(type, c("full", "summary"))
  method <- match.arg(method, c("pm", "isc"))
  simulation_method <- match.arg(simulation_method, c("spdk", "bootstrap", "psi"))
  
  if (n_thin > 1 && method == "isc") {
    stop ("Thinning with block-IS algorithm is not supported.")
  }
  
  if (nsim_states < 2) {
    #approximate inference
    method <- "pm"
    simulation_method <- "spdk"
  }
  
  if (missing(S)) {
    S <- diag(0.1 * pmax(0.1, abs(sapply(object$priors, "[[", "init"))), length(object$priors))
  }
  
  priors <- combine_priors(object$priors)
  
  object$distribution <- pmatch(object$distribution, c("poisson", "binomial", "negative binomial"))
  
  out <-  switch(type,
    full = {
      if (method == "pm"){
        out <- ng_bsm_run_mcmc(object, priors$prior_types, priors$params, n_iter,
          nsim_states, n_burnin, n_thin, gamma, target_acceptance, S,
          object$init_signal, seed, end_adaptive_phase, adaptive_approx,
          delayed_acceptance, pmatch(simulation_method, c("spdk", "bootstrap", "psi")))
        
      } else {
        out <- ng_bsm_run_mcmc_is(object, priors$prior_types, priors$params, n_iter,
          nsim_states, n_burnin, n_thin, gamma, target_acceptance, S,
          object$init_signal, seed, n_threads, end_adaptive_phase, adaptive_approx,
          pmatch(simulation_method, c("spdk", "bootstrap", "psi")), const_m, 
          sample(.Machine$integer.max, size = n_threads))
      }
      
      colnames(out$alpha) <- names(object$a1)
      out
    },
    summary = {
      if (method == "pm"){
        out <- ng_bsm_run_mcmc_summary(object, priors$prior_types, priors$params, n_iter,
          nsim_states, n_burnin, n_thin, gamma, target_acceptance, S,
          object$init_signal, seed,  n_threads, end_adaptive_phase, adaptive_approx,
          delayed_acceptance, pmatch(simulation_method, c("spdk", "bootstrap", "psi")))
      } else {
        
        out <- ng_bsm_run_mcmc_summary_is(object, priors$prior_types, priors$params, n_iter,
          nsim_states, n_burnin, n_thin, gamma, target_acceptance, S,
          object$init_signal, seed,  n_threads, end_adaptive_phase, adaptive_approx,
          pmatch(simulation_method, c("spdk", "bootstrap", "psi")), const_m, 
          sample(.Machine$integer.max, size = n_threads))
      }
      colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <- names(object$a1)
      out$alphahat <- ts(out$alphahat, start = start(object$y), frequency = frequency(object$y))
      out$muhat <- ts(out$muhat, start = start(object$y), frequency = frequency(object$y))
      out
    })
  
  names_ind <-
    c(!object$fixed & c(TRUE, object$slope, object$seasonal), object$noise)
  colnames(out$theta) <- rownames(out$S) <- colnames(out$S) <-
    c(c("sd_level", "sd_slope", "sd_seasonal", "sd_noise")[names_ind],
      colnames(object$xreg), if (nb) "nb_dispersion")
  if(method == "pm") {
    out$theta <- mcmc(out$theta, start = n_burnin + 1, thin = n_thin)
    out$jump_chain <- FALSE
  } else {
    out$jump_chain <- TRUE
    out$n_iter <- n_iter
    out$n_burnin <- n_burnin
  }
  out$call <- match.call()
  out$seed <- seed
  out$time <- proc.time() - a
  class(out) <- "mcmc_output"
  out
}




#' @method run_mcmc svm
#' @rdname run_mcmc_ng
#' @inheritParams run_mcmc.ngssm
#' @param gkl_prior If \code{gkl_prior} is \code{TRUE}, hard-coded priors based on 
#' Grothe, Kleppe, and Liesenfeld (2016) are used in SV model. Only applicable to model defined using
#' sigma in observation equation. See references for details.
#' @export
#' @references Oliver Grothe, Tore Selland Kleppe, Roman Liesenfeld (2016). 
#' "Bayesian Analysis in Non-linear Non-Gaussian State-Space Models using Particle Gibbs". 
#' ArXiv:1601.01125.
#'  
run_mcmc.svm <-  function(object, n_iter, nsim_states, type = "full",
  method = "pm", simulation_method = "spdk", const_m = TRUE,
  delayed_acceptance = TRUE, n_burnin = floor(n_iter/2),
  n_thin = 1, gamma = 2/3, target_acceptance = 0.234, S, end_adaptive_phase = TRUE,
  adaptive_approx  = TRUE, n_threads = 1,
  seed = sample(.Machine$integer.max, size = 1), gkl_prior = FALSE, ...) {
  
  set.seed(seed) #needed for reproducible parallel thread seeding
  
  a <- proc.time()
  check_target(target_acceptance)
  type <- match.arg(type, c("full", "summary"))
  method <- match.arg(method, c("pm", "isc"))
  simulation_method <- match.arg(simulation_method, c("spdk", "bootstrap", "psi"))
  
  if(gkl_prior && object$svm_type == 1) {
    stop("gkl prior is not applicable for the SV model parameterized with mu.")
  }
  if (n_thin > 1 && method == "isc") {
    stop ("Thinning with block-IS algorithm is not supported.")
  }
  
  if (nsim_states < 2) {
    #approximate inference
    method <- "pm"
    simulation_method <- "spdk"
  }
  
  if (missing(S)) {
    inits <- abs(sapply(object$priors, "[[", "init"))
    if(gkl_prior) {
      inits[1] <- 0.5 * (inits[1] + 1)
      inits[2] <- inits[2]^2
      inits[3] <- log(inits[3])
    }
    S <- diag(0.1 * pmax(0.1, inits), length(inits))
  }
  
  
  priors <- combine_priors(object$priors)
  
  out <-  switch(type,
    full = {
      if (method == "pm"){
        out <- svm_run_mcmc(object, priors$prior_types, priors$params, n_iter,
          nsim_states, n_burnin, n_thin, gamma, target_acceptance, S,
          object$init_signal, seed, end_adaptive_phase, adaptive_approx,
          delayed_acceptance, pmatch(simulation_method, c("spdk", "bootstrap", "psi")), gkl_prior)
        
      } else {
        out <- svm_run_mcmc_is(object, priors$prior_types, priors$params, n_iter,
          nsim_states, n_burnin, n_thin, gamma, target_acceptance, S,
          object$init_signal, seed, n_threads, end_adaptive_phase, adaptive_approx,
          pmatch(simulation_method, c("spdk", "bootstrap", "psi")), const_m, gkl_prior, 
          sample(.Machine$integer.max, size = n_threads))
      }
      
      colnames(out$alpha) <- names(object$a1)
      out
    },
    summary = {
      
      stop("summary for SV models not yet implemented.")
      # if (method == "pm"){
      #   out <- svm_run_mcmc_summary(object, priors$prior_types, priors$params, n_iter,
      #     nsim_states, n_burnin, n_thin, gamma, target_acceptance, S,
      #     object$init_signal, seed,  n_threads, end_adaptive_phase, adaptive_approx,
      #     delayed_acceptance, pmatch(simulation_method, c("spdk", "bootstrap", "psi")))
      # } else {
      #   if(correction_method == "PF") {
      #     stop("summary correction with particle filter is not yet implemented.")
      #   }
      #   out <- svm_run_mcmc_summary_is(object, priors$prior_types, priors$params, n_iter,
      #     nsim_states, n_burnin, n_thin, gamma, target_acceptance, S,
      #     object$init_signal, seed,  n_threads, end_adaptive_phase, adaptive_approx,
      #     pmatch(simulation_method, c("spdk", "bootstrap", "psi")), const_m)
      # }
      # colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <- names(object$a1)
      # out$alphahat <- ts(out$alphahat, start = start(object$y), frequency = frequency(object$y))
      # out$muhat <- ts(out$muhat, start = start(object$y), frequency = frequency(object$y))
      # out
    })
  
  if (gkl_prior) {
    out$theta[, 1] <- 2 * out$theta[, 1] - 1
    out$theta[, 2] <- sqrt(out$theta[, 2])
    out$theta[, 3] <- exp(out$theta[, 3])
  }
  colnames(out$theta) <- rownames(out$S) <- colnames(out$S) <-
    c(names(object$priors), names(object$coefs))
  if(method == "pm") {
    out$theta <- mcmc(out$theta, start = n_burnin + 1, thin = n_thin)
    out$jump_chain <- FALSE
  } else {
    out$jump_chain <- TRUE
    out$n_iter <- n_iter
    out$n_burnin <- n_burnin
  }
  out$call <- match.call()
  out$seed <- seed
  
  out$time <- proc.time() - a
  class(out) <- "mcmc_output"
  out
}

