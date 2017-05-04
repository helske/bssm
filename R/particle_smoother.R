#' Particle Smoothing
#'
#' Function \code{particle_smoother} performs filter-smoother or forward-backward smoother,
#' using a either bootstrap filtering or psi-auxiliary filter with stratification resampling.
#'
#' @param object Model.
#' @param nsim Number of samples.
#' @param smoothing_method Either \code{"fs"} (filter-smoother), or \code{"fbs"} 
#' (forward-backward smoother).
#' @param filter_type For Gaussian models, \code{"bsf"} for bootstrap 
#' filter. Also for non-Gaussian or non-linear models, \code{"psi"} uses 
#' psi-particle filter, and for non-linear models options \code{"ekf"} 
#' (extended Kalman particle filter) and \code{"apf"} (auxiliary particle filter)
#' are available.
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
particle_smoother.gssm <- function(object, nsim, smoothing_method = "fs",
  filter_type = "bsf", seed = sample(.Machine$integer.max, size = 1), 
  optimal = TRUE, ...) {
  
  filter_type <- match.arg(filter_type, c("bsf", "apf"))
  if (smoothing_method == "fbs") {
    stop("FBS is not yet implemented.")
  }
  smoothing_method <- match.arg(smoothing_method, c("fs", "fbs"))
  if(filter_type == "bsf") {
    bsf_smoother(model, nsim, seed, TRUE, 1L)
  } else {
    out <- aux_smoother(object, nsim, seed, TRUE, 1L, optimal)
  }
  colnames(out$alphahat) <- colnames(out$Vt) <-
    colnames(out$Vt) <- names(object$a1)
  out$alphahat <- ts(out$alphahat, start = start(object$y), frequency = frequency(object$y))
  
  rownames(out$alpha) <- names(object$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}

#' @method particle_smoother bsm
#' @export
particle_smoother.bsm <- function(object, nsim, smoothing_method = "fs",
  filter_type = "bsf", seed = sample(.Machine$integer.max, size = 1), 
  optimal = TRUE, ...) {
  
  filter_type <- match.arg(filter_type, c("bsf", "apf"))
  if (smoothing_method == "fbs") {
    stop("FBS is not yet implemented.")
  }
  smoothing_method <- match.arg(smoothing_method, c("fs", "fbs"))
  if(filter_type == "bsf") {
    out <- bsf_smoother(object, nsim, seed, TRUE, 2L)
  } else {
    out <- aux_smoother(object, nsim, seed, TRUE, 2L, optimal)
  }
  colnames(out$alphahat) <- colnames(out$Vt) <-
    colnames(out$Vt) <- names(object$a1)
  out$alphahat <- ts(out$alphahat, start = start(object$y), frequency = frequency(object$y))
  
  rownames(out$alpha) <- names(object$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}
#' @rdname particle_smoother
#' @method particle_smoother ngssm
#' @export
particle_smoother.ngssm <- function(object, nsim, smoothing_method = "fs", 
  filter_type = "bsf", seed = sample(.Machine$integer.max, size = 1), 
  max_iter = 100, conv_tol = 1e-8, ...) {
  
  smoothing_method <- match.arg(smoothing_method, c("fs", "fbs"))
  filter_type <- match.arg(filter_type, c("bsf", "psi"))
  if (smoothing_method == "fbs" && filter_type == "psi") {
    stop("FBS with psi-filter is not yet implemented.")
  }
  object$distribution <- pmatch(object$distribution, c("poisson", "binomial", "negative binomial"))
  if(filter_type == "psi") {
    out <- psi_smoother(object, object$initial_mode, nsim, smoothing_method == "fs", 
      seed, max_iter, conv_tol, 1L)
  } else {
    out <- bsf_smoother(object, nsim, seed, FALSE, 1L)
  }
  colnames(out$alphahat) <- colnames(out$Vt) <-
    colnames(out$Vt) <- names(object$a1)
  out$alphahat <- ts(out$alphahat, start = start(object$y), frequency = frequency(object$y))
  
  rownames(out$alpha) <- names(object$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}
#' @method particle_smoother ng_bsm
#' @export
particle_smoother.ng_bsm <- function(object, nsim, filter_type = "psi", 
  smoothing_method = "fs", seed = sample(.Machine$integer.max, size = 1), 
  max_iter = 100, conv_tol = 1e-8, ...) {
  
  smoothing_method <- match.arg(smoothing_method, c("fs", "fbs"))
  filter_type <- match.arg(filter_type, c("psi", "bsf"))
  if (smoothing_method == "fbs" && filter_type == "psi") {
    stop("FBS is not yet implemented.")
  }
  object$distribution <- pmatch(object$distribution, c("poisson", "binomial", "negative binomial"))
  if(filter_type == "psi") {
    out <- psi_smoother(object, object$initial_mode, nsim, 
      smoothing_method == "fs", seed, max_iter, conv_tol, 2L)
  } else {
    out <- bsf_smoother(object, nsim, seed, FALSE, 2L)
  }
  colnames(out$alphahat) <- colnames(out$Vt) <-
    colnames(out$Vt) <- names(object$a1)
  out$alphahat <- ts(out$alphahat, start = start(object$y), frequency = frequency(object$y))
  
  rownames(out$alpha) <- names(object$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}
#' @method particle_smoother svm
#' @export
particle_smoother.svm <- function(object, nsim, smoothing_method = "fs", 
  filter_type = "psi", seed = sample(.Machine$integer.max, size = 1), 
  max_iter = 100, conv_tol = 1e-8, ...) {
  
  smoothing_method <- match.arg(smoothing_method, c("fs", "fbs"))
  filter_type <- match.arg(filter_type, c("psi", "bsf"))
  if (smoothing_method == "fbs" && filter_type == "psi") {
    stop("FBS with psi-filter is not yet implemented.")
  }
  if(filter_type == "psi") {
    out <- psi_smoother(object, object$initial_mode, nsim,
      smoothing_method == "fs", seed, max_iter, conv_tol, 3L)
  } else {
    out <- bsf_smoother(object, nsim, seed, FALSE, 3L)
  }
  colnames(out$alphahat) <- colnames(out$Vt) <-
    colnames(out$Vt) <- names(object$a1)
  out$alphahat <- ts(out$alphahat, start = start(object$y), frequency = frequency(object$y))
  
  rownames(out$alpha) <- names(object$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}

#' @method particle_smoother nlg_ssm
#' @export
particle_smoother.nlg_ssm <- function(object, nsim, smoothing_method = "fs", 
  filter_type = "psi", seed = sample(.Machine$integer.max, size = 1),
  max_iter = 100, conv_tol = 1e-8, iekf_iter = 0, ...) {
  
  smoothing_method <- match.arg(smoothing_method, c("fs", "fbs"))
  filter_type <- match.arg(filter_type, c("bsf", "psi", "apf", "ekf", "psi_df"))
  if (smoothing_method == "fbs") {
    stop("FBS is not yet implemented.")
  }
  out <- switch(filter_type,
    psi = psi_smoother_nlg(t(object$y), object$Z, object$H, object$T, 
      object$R, object$Z_gn, object$T_gn, object$a1, object$P1, 
      object$theta, object$log_prior_pdf, object$known_params, 
      object$known_tv_params, object$n_states, object$n_etas, 
      as.integer(object$time_varying), as.integer(object$state_varying), nsim, seed,
      max_iter, conv_tol, iekf_iter),
    bsf = bsf_smoother_nlg(t(object$y), object$Z, object$H, object$T, 
      object$R, object$Z_gn, object$T_gn, object$a1, object$P1, 
      object$theta, object$log_prior_pdf, object$known_params, 
      object$known_tv_params, object$n_states, object$n_etas, 
      as.integer(object$time_varying), as.integer(object$state_varying), nsim, seed),
    apf = aux_smoother_nlg(t(object$y), object$Z, object$H, object$T, 
      object$R, object$Z_gn, object$T_gn, object$a1, object$P1, 
      object$theta, object$log_prior_pdf, object$known_params, 
      object$known_tv_params, object$n_states, object$n_etas, 
      as.integer(object$time_varying), as.integer(object$state_varying), nsim, 
      seed),
    ekf = ekpf_smoother(t(object$y), object$Z, object$H, object$T, 
      object$R, object$Z_gn, object$T_gn, object$a1, object$P1, 
      object$theta, object$log_prior_pdf, object$known_params, 
      object$known_tv_params, object$n_states, object$n_etas, 
      as.integer(object$time_varying), as.integer(object$state_varying), nsim, 
      seed),
    psi_df = df_psi_smoother_nlg(t(object$y), object$Z, object$H, object$T, 
      object$R, object$Z_gn, object$T_gn, object$a1, object$P1, 
      object$theta, object$log_prior_pdf, object$known_params, 
      object$known_tv_params, object$n_states, object$n_etas, 
      as.integer(object$time_varying), as.integer(object$state_varying), nsim, seed,
      max_iter, conv_tol, iekf_iter)
  )
  colnames(out$alphahat) <- colnames(out$Vt) <-
    colnames(out$Vt) <- object$state_names
  out$alphahat <- ts(out$alphahat, start = start(object$y), frequency = frequency(object$y))
  
  rownames(out$alpha) <- object$state_names
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}
