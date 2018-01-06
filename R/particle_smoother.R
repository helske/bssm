#' Particle Smoothing
#'
#' Function \code{particle_smoother} performs filter-smoother or forward-backward smoother,
#' using a either bootstrap filtering or psi-auxiliary filter with stratification resampling.
#'
#' @param object Model.
#' @param nsim Number of samples.
#' @param filter_type Choice of particle filter algorithm. For Gaussian models, 
#' only option is \code{"bsf"} (bootstrap particle filter). 
#' In addition, for non-Gaussian or 
#' non-linear models, \code{"psi"} uses psi-particle filter, and 
#' for non-linear models options \code{"ekf"} (extended Kalman particle filter) 
#' is also available.
#' @param max_iter Maximum number of iterations used in Gaussian approximation. Used psi-PF.
#' @param conv_tol Tolerance parameter used in Gaussian approximation. Used psi-PF.
#' @param iekf_iter If zero (default), first approximation for non-linear 
#' Gaussian models is obtained from extended Kalman filter. If 
#' \code{iekf_iter > 0}, iterated extended Kalman filter is used with 
#' \code{iekf_iter} iterations.
#' @param seed Seed for RNG.
#' @param ... Ignored.
#' @export
#' @rdname particle_smoother
particle_smoother <- function(object, nsim, ...) {
  UseMethod("particle_smoother", object)
}
#' @method particle_smoother gssm
#' @rdname particle_smoother
#' @export
particle_smoother.gssm <- function(object, nsim,
  seed = sample(.Machine$integer.max, size = 1), ...) {
  
  out <- bsf_smoother(object, nsim, seed, TRUE, 1L)
  
  colnames(out$alphahat) <- colnames(out$Vt) <-
    colnames(out$Vt) <- names(object$a1)
  out$Vt <- out$Vt[, , -nrow(out$alphahat), drop = FALSE]
  out$alphahat <- ts(out$alphahat[-nrow(out$alphahat), , drop = FALSE], 
    start = start(object$y), frequency = frequency(object$y))
  rownames(out$alpha) <- names(object$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}

#' @method particle_smoother bsm
#' @export
particle_smoother.bsm <- function(object, nsim, 
  seed = sample(.Machine$integer.max, size = 1), ...) {
  
  out <- bsf_smoother(object, nsim, seed, TRUE, 2L)
  
  colnames(out$alphahat) <- colnames(out$Vt) <-
    colnames(out$Vt) <- names(object$a1)
  out$Vt <- out$Vt[, , -nrow(out$alphahat), drop = FALSE]
  out$alphahat <- ts(out$alphahat[-nrow(out$alphahat), , drop = FALSE], 
    start = start(object$y), frequency = frequency(object$y))
  rownames(out$alpha) <- names(object$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}
#' @rdname particle_smoother
#' @method particle_smoother ngssm
#' @export
particle_smoother.ngssm <- function(object, nsim, 
  filter_type = "bsf", 
  seed = sample(.Machine$integer.max, size = 1), 
  max_iter = 100, conv_tol = 1e-8, ...) {
  
  filter_type <- match.arg(filter_type, c("bsf", "psi"))
  
  object$distribution <- pmatch(object$distribution, c("poisson", "binomial", "negative binomial"))
  if(filter_type == "psi") {
    out <- psi_smoother(object, object$initial_mode, nsim, 
      seed, max_iter, conv_tol, 1L)
  } else {
    out <- bsf_smoother(object, nsim, seed, FALSE, 1L)
  }
  colnames(out$alphahat) <- colnames(out$Vt) <-
    colnames(out$Vt) <- names(object$a1)
  out$Vt <- out$Vt[, , -nrow(out$alphahat), drop = FALSE]
  out$alphahat <- ts(out$alphahat[-nrow(out$alphahat), , drop = FALSE], 
    start = start(object$y), frequency = frequency(object$y))
  rownames(out$alpha) <- names(object$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}
#' @method particle_smoother ng_bsm
#' @export
particle_smoother.ng_bsm <- function(object, nsim, filter_type = "psi", 
  seed = sample(.Machine$integer.max, size = 1), 
  max_iter = 100, conv_tol = 1e-8, ...) {
  
  filter_type <- match.arg(filter_type, c("psi", "bsf"))
  object$distribution <- pmatch(object$distribution, c("poisson", "binomial", "negative binomial"))
  if(filter_type == "psi") {
    out <- psi_smoother(object, object$initial_mode, nsim, 
      seed, max_iter, conv_tol, 2L)
  } else {
    out <- bsf_smoother(object, nsim, seed, FALSE, 2L)
  }
  colnames(out$alphahat) <- colnames(out$Vt) <-
    colnames(out$Vt) <- names(object$a1)
  out$Vt <- out$Vt[, , -nrow(out$alphahat), drop = FALSE]
  out$alphahat <- ts(out$alphahat[-nrow(out$alphahat), , drop = FALSE], 
    start = start(object$y), frequency = frequency(object$y))
  rownames(out$alpha) <- names(object$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}
#' @method particle_smoother svm
#' @export
particle_smoother.svm <- function(object, nsim,
  filter_type = "psi", 
  seed = sample(.Machine$integer.max, size = 1), 
  max_iter = 100, conv_tol = 1e-8, ...) {
  
  filter_type <- match.arg(filter_type, c("psi", "bsf"))
  if(filter_type == "psi") {
    out <- psi_smoother(object, object$initial_mode, nsim,
      seed, max_iter, conv_tol, 3L)
  } else {
    out <- bsf_smoother(object, nsim, seed, FALSE, 3L)
  }
  colnames(out$alphahat) <- colnames(out$Vt) <-
    colnames(out$Vt) <- names(object$a1)
  out$Vt <- out$Vt[, , -nrow(out$alphahat), drop = FALSE]
  out$alphahat <- ts(out$alphahat[-nrow(out$alphahat), , drop = FALSE], 
    start = start(object$y), frequency = frequency(object$y))
  rownames(out$alpha) <- names(object$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}
#' @rdname particle_smoother
#' @method particle_smoother nlg_ssm
#' @export
particle_smoother.nlg_ssm <- function(object, nsim, 
  filter_type = "psi", 
  seed = sample(.Machine$integer.max, size = 1),
  max_iter = 100, conv_tol = 1e-8, iekf_iter = 0, ...) {
  
  filter_type <- match.arg(filter_type, c("bsf", "psi", "ekf"))
  
  out <- switch(filter_type,
    psi = psi_smoother_nlg(t(object$y), object$Z, object$H, object$T, 
      object$R, object$Z_gn, object$T_gn, object$a1, object$P1, 
      object$theta, object$log_prior_pdf, object$known_params, 
      object$known_tv_params, object$n_states, object$n_etas, 
      as.integer(object$time_varying), nsim, seed,
      max_iter, conv_tol, iekf_iter),
    bsf = bsf_smoother_nlg(t(object$y), object$Z, object$H, object$T, 
      object$R, object$Z_gn, object$T_gn, object$a1, object$P1, 
      object$theta, object$log_prior_pdf, object$known_params, 
      object$known_tv_params, object$n_states, object$n_etas, 
      as.integer(object$time_varying), nsim, seed),
    ekf = ekpf_smoother(t(object$y), object$Z, object$H, object$T, 
      object$R, object$Z_gn, object$T_gn, object$a1, object$P1, 
      object$theta, object$log_prior_pdf, object$known_params, 
      object$known_tv_params, object$n_states, object$n_etas, 
      as.integer(object$time_varying), nsim, 
      seed)
  )
  colnames(out$alphahat) <- colnames(out$Vt) <-
    colnames(out$Vt) <- object$state_names
  out$Vt <- out$Vt[, , -nrow(out$alphahat), drop = FALSE]
  out$alphahat <- ts(out$alphahat[-nrow(out$alphahat), , drop = FALSE], 
    start = start(object$y), frequency = frequency(object$y))
  rownames(out$alpha) <- object$state_names
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}


#' @rdname particle_smoother
#' @method particle_smoother sde_ssm
#' @param L Integer defining the discretization level.
#' @export
particle_smoother.sde_ssm <- function(object, nsim, L, 
seed = sample(.Machine$integer.max, size = 1), ...) {
  
  if(L < 1) stop("Discretization level L must be larger than 0.")
  out <-  bsf_smoother_sde(object$y, object$x0, object$positive, 
    object$drift, object$diffusion, object$ddiffusion, 
    object$prior_pdf, object$obs_pdf, object$theta, 
    nsim, round(L), seed)
  
  colnames(out$alphahat) <- colnames(out$Vt) <-
    colnames(out$Vt) <- object$state_names
  out$Vt <- out$Vt[, , -nrow(out$alphahat), drop = FALSE]
  out$alphahat <- ts(out$alphahat[-nrow(out$alphahat), , drop = FALSE], 
    start = start(object$y), frequency = frequency(object$y))
  rownames(out$alpha) <- object$state_names
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}

