#' Kalman Smoothing
#'
#' Methods for Kalman smoothing of the states. Function \code{fast_smoother}
#' computes only smoothed estimates of the states, and function
#' \code{smoother} computes also smoothed variances.
#' 
#' For non-Gaussian models, the smoothing is based on the approximate Gaussian model.
#'
#' @param object Model object.
#' @param ... Ignored.
#' @return Matrix containing the smoothed estimates of states, or a list
#' with the smoothed states and the variances.
#' @export
#' @rdname smoother
fast_smoother <- function(object, ...) {
  UseMethod("fast_smoother", object)
}
#' @method fast_smoother ssm_ulg
#' @export
fast_smoother.ssm_ulg <- function(object, ...) {
  
  out <- gaussian_fast_smoother(object, model_type = 1L)
  colnames(out) <- names(object$a1)
  ts(out[-nrow(out), , drop = FALSE], start = start(object$y), 
    frequency = frequency(object$y))
}
#' @method fast_smoother bsm_lg
#' @export
fast_smoother.bsm_lg <- function(object, ...) {
  
  out <- gaussian_fast_smoother(object, model_type = 2L)
  colnames(out) <- names(object$a1)
  ts(out[-nrow(out), , drop = FALSE], start = start(object$y), 
    frequency = frequency(object$y))
}
#' @method fast_smoother ar1_lg
#' @export
fast_smoother.ar1_lg <- function(object, ...) {
  
  out <- gaussian_fast_smoother(object, model_type = 3L)
  colnames(out) <- names(object$a1)
  ts(out[-nrow(out), , drop = FALSE], start = start(object$y), 
    frequency = frequency(object$y))
}

#' @method fast_smoother ssm_ung
#' @export
fast_smoother.ssm_ung <- function(object, ...) {
  fast_smoother(gaussian_approx(object))
}
#' @method fast_smoother bsm_ng
#' @export
fast_smoother.bsm_ng <- function(object, ...) {
  fast_smoother(gaussian_approx(object))
}
#' @method fast_smoother svm
#' @export
fast_smoother.svm <- function(object, ...) {
  fast_smoother(gaussian_approx(object))
}
#' @method fast_smoother ar1_ng
#' @export
fast_smoother.ar1_ng <- function(object, ...) {
  fast_smoother(gaussian_approx(object))
}
#' @export
#' @rdname smoother
smoother <- function(object, ...) {
  UseMethod("smoother", object)
}
#' @method smoother ssm_ulg
#' @export
smoother.ssm_ulg <- function(object, ...) {
  
  out <-  gaussian_smoother(object, model_type = 1L)
  colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <- names(object$a1)
  
  out$Vt <- out$Vt[, , -nrow(out$alphahat), drop = FALSE]
  out$alphahat <- ts(out$alphahat[-nrow(out$alphahat), , drop = FALSE], 
    start = start(object$y), frequency = frequency(object$y))
  out
}

#' @method smoother bsm_lg
#' @export
smoother.bsm_lg <- function(object, ...) {
  
  out <- gaussian_smoother(object, model_type = 2L)
  colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <- names(object$a1)
  out$Vt <- out$Vt[, , -nrow(out$alphahat), drop = FALSE]
  out$alphahat <- ts(out$alphahat[-nrow(out$alphahat), , drop = FALSE], 
    start = start(object$y), frequency = frequency(object$y))
  out
}
#' @method smoother ar1_lg
#' @export
smoother.ar1_lg <- function(object, ...) {
  
  out <- gaussian_smoother(object, model_type = 3L)
  colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <- names(object$a1)
  out$Vt <- out$Vt[, , -nrow(out$alphahat), drop = FALSE]
  out$alphahat <- ts(out$alphahat[-nrow(out$alphahat), , drop = FALSE], 
    start = start(object$y), frequency = frequency(object$y))
  out
}
#' @method smoother lgg_ssm
#' @export
smoother.lgg_ssm <- function(object, ...) {
  
  out <- general_gaussian_smoother(t(object$y), object$Z, object$H, object$T, 
    object$R, object$a1, object$P1, 
    object$theta, object$obs_intercept, object$state_intercept,
    object$log_prior_pdf, object$known_params, 
    object$known_tv_params, as.integer(object$time_varying), 
    object$n_states, object$n_etas)
  colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <- object$state_names
  out$Vt <- out$Vt[, , -nrow(out$alphahat), drop = FALSE]
  out$alphahat <- ts(out$alphahat[-nrow(out$alphahat), , drop = FALSE], 
    start = start(object$y), frequency = frequency(object$y))
  out
}
#' @method smoother ssm_ung
#' @export
smoother.ssm_ung <- function(object, ...) {
  smoother(gaussian_approx(object))
}

#' @method smoother bsm_ng
#' @export
smoother.bsm_ng <- function(object, ...) {
  smoother(gaussian_approx(object))
}

#' @method smoother svm
#' @export
smoother.svm <- function(object, ...) {
  smoother(gaussian_approx(object))
}

#' @method smoother ar1_ng
#' @export
smoother.ar1_ng <- function(object, ...) {
  smoother(gaussian_approx(object))
}
#' Extended Kalman Smoothing
#'
#' Function \code{ekf_smoother} runs the (iterated) extended Kalman smoother for 
#' the given non-linear Gaussian model of class \code{nlg_ssm}, 
#' and returns the smoothed estimates of the states and the corresponding variances.
#'
#' @param object Model object
#' @param iekf_iter If \code{iekf_iter > 0}, iterated extended Kalman filter is 
#' used with \code{iekf_iter} iterations.
#' @return List containing the log-likelihood,
#' smoothed state estimates \code{alphahat}, and the corresponding variances \code{Vt} and
#'  \code{Ptt}.
#' @export
#' @rdname ekf_smoother
#' @export
ekf_smoother <- function(object, iekf_iter = 0) {
  
  out <- ekf_smoother_nlg(t(object$y), object$Z, object$H, object$T, 
    object$R, object$Z_gn, object$T_gn, object$a1, object$P1, 
    object$theta, object$log_prior_pdf, object$known_params, 
    object$known_tv_params, object$n_states, object$n_etas, 
    as.integer(object$time_varying), iekf_iter)
  out$Vt <- out$Vt[, , -nrow(out$alphahat), drop = FALSE]
  out$alphahat <- ts(out$alphahat[-nrow(out$alphahat), , drop = FALSE], 
    start = start(object$y), frequency = frequency(object$y))
  out
}

ekf_fast_smoother <- function(object, iekf_iter = 0) {
  
  out <- ekf_fast_smoother_nlg(t(object$y), object$Z, object$H, object$T, 
    object$R, object$Z_gn, object$T_gn, object$a1, object$P1, 
    object$theta, object$log_prior_pdf, object$known_params, 
    object$known_tv_params, object$n_states, object$n_etas, 
    as.integer(object$time_varying), iekf_iter)
  ts(out[-nrow(out$alphahat), , drop = FALSE], start = start(object$y), 
    frequency = frequency(object$y))
}

