#' Particle Smoothing
#'
#' Function \code{particle_smoother} performs filter-smoother or forward-backward smoother,
#' using a either bootstrap filtering or psi-auxiliary filter with stratification resampling.
#'
#' @param model Model.
#' @param nsim Number of samples.
#' @param method Choice of particle filter algorithm. 
#' For Gaussian and non-Gaussian models with linear dynamics,
#' options are \code{"bsf"} (bootstrap particle filter) 
#' and \code{"psi"} (\eqn{\psi}{psi}-APF, the default), and 
#' for non-linear models options \code{"ekf"} (extended Kalman particle filter) 
#' is also available.
#' @param max_iter Maximum number of iterations used in Gaussian approximation. Used \eqn{\psi}{psi}-APF.
#' @param conv_tol Tolerance parameter used in Gaussian approximation. Used \eqn{\psi}{psi}-APF.
#' @param iekf_iter If zero (default), first approximation for non-linear 
#' Gaussian models is obtained from extended Kalman filter. If 
#' \code{iekf_iter > 0}, iterated extended Kalman filter is used with 
#' \code{iekf_iter} iterations.
#' @param seed Seed for RNG.
#' @param ... Ignored.
#' @export
#' @rdname particle_smoother
particle_smoother <- function(model, nsim, ...) {
  UseMethod("particle_smoother", model)
}

#' @method particle_smoother gaussian
#' @export
particle_smoother.gaussian <- function(model, nsim,  method = "psi",
  seed = sample(.Machine$integer.max, size = 1), ...) {
  
  if(method == "psi") {
    out <- gaussian_psi_smoother(model, nsim, seed, model_type(model))
  } else {
    out <- bsf_smoother(model, nsim, seed, TRUE, model_type(model))
  }
  colnames(out$alphahat) <- colnames(out$Vt) <-
    colnames(out$Vt) <- names(model$a1)
  out$Vt <- out$Vt[, , -nrow(out$alphahat), drop = FALSE]
  out$alphahat <- ts(out$alphahat[-nrow(out$alphahat), , drop = FALSE], 
    start = start(model$y), frequency = frequency(model$y))
  rownames(out$alpha) <- names(model$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}
#' @rdname particle_smoother
#' @method particle_smoother nongaussian
#' @export
particle_smoother.nongaussian <- function(model, nsim, 
  method = "psi", 
  seed = sample(.Machine$integer.max, size = 1), 
  max_iter = 100, conv_tol = 1e-8, ...) {
  
  method <- match.arg(method, c("bsf", "psi"))
  
  model$max_iter <- max_iter
  model$conv_tol <- conv_tol
  model$distribution <- pmatch(model$distribution,
    c("svm", "poisson", "binomial", "negative binomial", "gamma", "gaussian"), 
    duplicates.ok = TRUE) - 1
  
  if(method == "psi") {
    out <- psi_smoother(model, nsim, seed, model_type(model))
  } else {
    out <- bsf_smoother(model, nsim, seed, FALSE, model_type(model))
  }
  colnames(out$alphahat) <- colnames(out$Vt) <-
    colnames(out$Vt) <- names(model$a1)
  out$Vt <- out$Vt[, , -nrow(out$alphahat), drop = FALSE]
  out$alphahat <- ts(out$alphahat[-nrow(out$alphahat), , drop = FALSE], 
    start = start(model$y), frequency = frequency(model$y))
  rownames(out$alpha) <- names(model$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}

#' @rdname particle_smoother
#' @method particle_smoother ssm_nlg
#' @export
particle_smoother.ssm_nlg <- function(model, nsim, 
  method = "psi", 
  seed = sample(.Machine$integer.max, size = 1),
  max_iter = 100, conv_tol = 1e-8, iekf_iter = 0, ...) {
  
  method <- match.arg(method, c("bsf", "psi", "ekf"))
  
  out <- switch(method,
    psi = psi_smoother_nlg(t(model$y), model$Z, model$H, model$T, 
      model$R, model$Z_gn, model$T_gn, model$a1, model$P1, 
      model$theta, model$log_prior_pdf, model$known_params, 
      model$known_tv_params, model$n_states, model$n_etas, 
      as.integer(model$time_varying), nsim, seed,
      max_iter, conv_tol, iekf_iter, default_update_fn, default_prior_fn),
    bsf = bsf_smoother_nlg(t(model$y), model$Z, model$H, model$T, 
      model$R, model$Z_gn, model$T_gn, model$a1, model$P1, 
      model$theta, model$log_prior_pdf, model$known_params, 
      model$known_tv_params, model$n_states, model$n_etas, 
      as.integer(model$time_varying), nsim, seed, default_update_fn, default_prior_fn),
    ekf = ekpf_smoother(t(model$y), model$Z, model$H, model$T, 
      model$R, model$Z_gn, model$T_gn, model$a1, model$P1, 
      model$theta, model$log_prior_pdf, model$known_params, 
      model$known_tv_params, model$n_states, model$n_etas, 
      as.integer(model$time_varying), nsim, 
      seed, default_update_fn, default_prior_fn)
  )
  colnames(out$alphahat) <- colnames(out$Vt) <-
    colnames(out$Vt) <- model$state_names
  out$Vt <- out$Vt[, , -nrow(out$alphahat), drop = FALSE]
  out$alphahat <- ts(out$alphahat[-nrow(out$alphahat), , drop = FALSE], 
    start = start(model$y), frequency = frequency(model$y))
  rownames(out$alpha) <- model$state_names
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}


#' @rdname particle_smoother
#' @method particle_smoother ssm_sde
#' @param L Integer defining the discretization level.
#' @export
particle_smoother.ssm_sde <- function(model, nsim, L, 
  seed = sample(.Machine$integer.max, size = 1), ...) {
  
  if(L < 1) stop("Discretization level L must be larger than 0.")
  out <-  bsf_smoother_sde(model$y, model$x0, model$positive, 
    model$drift, model$diffusion, model$ddiffusion, 
    model$prior_pdf, model$obs_pdf, model$theta, 
    nsim, round(L), seed)
  
  colnames(out$alphahat) <- colnames(out$Vt) <-
    colnames(out$Vt) <- model$state_names
  out$Vt <- out$Vt[, , -nrow(out$alphahat), drop = FALSE]
  out$alphahat <- ts(out$alphahat[-nrow(out$alphahat), , drop = FALSE], 
    start = start(model$y), frequency = frequency(model$y))
  rownames(out$alpha) <- model$state_names
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}

