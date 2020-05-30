#' Particle Smoothing
#'
#' Function \code{particle_smoother} performs filter-smoother or forward-backward smoother,
#' using a either bootstrap filtering or psi-auxiliary filter with stratification resampling.
#'
#' @param object Model.
#' @param nsim Number of samples.
#' @param method Choice of particle filter algorithm. 
#' For Gaussian and non-Gaussian models with linear dynamics,
#' options are \code{"bsf"} (bootstrap particle filter) 
#' and \code{"psi"} (psi-APF, the default), and 
#' for non-linear models options \code{"ekf"} (extended Kalman particle filter) 
#' is also available.
#' @param max_iter Maximum number of iterations used in Gaussian approximation. Used psi-APF.
#' @param conv_tol Tolerance parameter used in Gaussian approximation. Used psi-APF.
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

#' @method particle_smoother gaussian
#' @export
particle_smoother.gaussian <- function(object, nsim,  method = "psi",
  seed = sample(.Machine$integer.max, size = 1), ...) {
  
  if(method == "psi") {
    out <- gaussian_psi_smoother(object, nsim, seed, model_type(model))
  } else {
    out <- bsf_smoother(object, nsim, seed, TRUE, model_type(model))
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
#' @method particle_smoother nongaussian
#' @export
particle_smoother.nongaussian <- function(object, nsim, 
  method = "psi", 
  seed = sample(.Machine$integer.max, size = 1), 
  max_iter = 100, conv_tol = 1e-8, ...) {
  
  method <- match.arg(method, c("bsf", "psi"))
  
  model$max_iter <- max_iter
  model$conv_tol <- conv_tol
  model$distribution <- pmatch(model$distribution,
    c("svm", "poisson", "binomial", "negative binomial")) - 1
  
  if(method == "psi") {
    out <- psi_smoother(object, nsim, seed, model_type(model))
  } else {
    out <- bsf_smoother(object, nsim, seed, FALSE, model_type(model))
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
  method = "psi", 
  seed = sample(.Machine$integer.max, size = 1),
  max_iter = 100, conv_tol = 1e-8, iekf_iter = 0, ...) {
  
  method <- match.arg(method, c("bsf", "psi", "ekf"))
  
  out <- switch(method,
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

