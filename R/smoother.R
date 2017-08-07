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
#' @method fast_smoother gssm
#' @export
fast_smoother.gssm <- function(object, ...) {
  
  out <- gaussian_fast_smoother(object, model_type = 1L)
  colnames(out) <- names(object$a1)
  ts(out, start = start(object$y), frequency = frequency(object$y))
}
#' @method fast_smoother bsm
#' @export
fast_smoother.bsm <- function(object, ...) {
  
  out <- gaussian_fast_smoother(object, model_type = 2L)
  colnames(out) <- names(object$a1)
  ts(out, start = start(object$y), frequency = frequency(object$y))
}
#' @method fast_smoother mv_gssm
#' @export
fast_smoother.mv_gssm <- function(object, ...) {
  
  out <- gaussian_fast_smoother(object, model_type = -1L)
  colnames(out) <- names(object$a1)
  ts(out, start = start(object$y), frequency = frequency(object$y))
}
#' @method fast_smoother ngssm
#' @export
fast_smoother.ngssm <- function(object, ...) {
  fast_smoother(gaussian_approx(object))
}
#' @method fast_smoother ng_bsm
#' @export
fast_smoother.ng_bsm <- function(object, ...) {
  fast_smoother(gaussian_approx(object))
}
#' @method fast_smoother svm
#' @export
fast_smoother.svm <- function(object, ...) {
  fast_smoother(gaussian_approx(object))
}
#' @export
#' @rdname smoother
smoother <- function(object, ...) {
  UseMethod("smoother", object)
}
#' @method smoother gssm
#' @export
smoother.gssm <- function(object, ...) {
  
  out <-  gaussian_smoother(object, model_type = 1L)
  colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <- names(object$a1)
  out$alphahat <- ts(out$alphahat, start = start(object$y), frequency = frequency(object$y))
  out
}
#' @method smoother bsm
#' @export
smoother.bsm <- function(object, ...) {
  
  out <- gaussian_smoother(object, model_type = 2L)
  colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <- names(object$a1)
  out$alphahat <- ts(out$alphahat, start = start(object$y), frequency = frequency(object$y))
  out
}
#' @method smoother lgg_ssm
#' @export
smoother.lgg_ssm <- function(object, ...) {
  
  out <- general_gaussian_smoother(t(object$y), object$Z, object$H, object$T, 
    object$R, object$a1, object$P1, 
    object$theta, object$obs_intercept, object$state_intercept,
    object$log_prior_pdf, object$known_params, 
    object$known_tv_params,
    object$n_states, object$n_etas)
  colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <- object$state_names
  out$alphahat <- ts(out$alphahat, start = start(object$y), frequency = frequency(object$y))
  out
}
#' @method smoother ngssm
#' @export
smoother.ngssm <- function(object, ...) {
  smoother(gaussian_approx(object))
}

#' @method smoother ng_bsm
#' @export
smoother.ng_bsm <- function(object, ...) {
  smoother(gaussian_approx(object))
}

#' @method smoother svm
#' @export
smoother.svm <- function(object, ...) {
  smoother(gaussian_approx(object))
}

#' Extended Kalman Smoothing
#'
#' Function \code{ekf_smoother} runs the (iterated) extended Kalman smoother for 
#' the given non-linear Gaussian model of class \code{nlg_ssm}, 
#' and returns the filtered estimates and one-step-ahead predictions of the 
#' states \eqn{\alpha_t} given the data up to time \eqn{t}.
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
  out$alphahat <- ts(out$alphahat, start = start(object$y), 
    frequency = frequency(object$y))
  out
}

ekf_fast_smoother <- function(object, iekf_iter = 0) {
  
  out <- ekf_fast_smoother_nlg(t(object$y), object$Z, object$H, object$T, 
    object$R, object$Z_gn, object$T_gn, object$a1, object$P1, 
    object$theta, object$log_prior_pdf, object$known_params, 
    object$known_tv_params, object$n_states, object$n_etas, 
    as.integer(object$time_varying), iekf_iter)
  ts(out, start = start(object$y), 
    frequency = frequency(object$y))
}

