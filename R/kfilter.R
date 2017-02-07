#' Kalman Filtering
#'
#' Function \code{kfilter} runs the Kalman filter for the given model, 
#' and returns the filtered estimates and one-step-ahead predictions of the 
#' states \eqn{\alpha_t} given the data up to time \eqn{t}.
#'
#' For non-Gaussian models, the Kalman filtering is based on the approximate Gaussian model.
#'
#' @param object Model object
#' @param ... Ignored.
#' @return List containing the log-likelihood (approximate in non-Gaussian case),
#' one-step-ahead predictions \code{at} and filtered
#' estimates \code{att} of states, and the corresponding variances \code{Pt} and
#'  \code{Ptt}.
#' @seealso \code{\link{particle_filter}}
#' @export
#' @rdname kfilter
kfilter <- function(object, ...) {
  UseMethod("kfilter", object)
}

#' @method kfilter gssm
#' @export
kfilter.gssm <- function(object, ...) {
  
  out <- gaussian_kfilter(object, model_type = 1L)
  colnames(out$at) <- colnames(out$att) <- colnames(out$Pt) <-
    colnames(out$Ptt) <- rownames(out$Pt) <- rownames(out$Ptt) <- names(object$a1)
  out$at <- ts(out$at, start = start(object$y), frequency = frequency(object$y))
  out$att <- ts(out$att, start = start(object$y), frequency = frequency(object$y))
  out
}
#' @method kfilter bsm
#' @export
kfilter.bsm <- function(object, ...) {
  
  out <- gaussian_kfilter(object, model_type = 2L)
  colnames(out$at) <- colnames(out$att) <- colnames(out$Pt) <-
    colnames(out$Ptt) <- rownames(out$Pt) <- rownames(out$Ptt) <- names(object$a1)
  out$at <- ts(out$at, start = start(object$y), frequency = frequency(object$y))
  out$att <- ts(out$att, start = start(object$y), frequency = frequency(object$y))
  out
}

#' @method kfilter ngssm
#' @export
kfilter.ngssm <- function(object, ...) {
  gaussian_kfilter(gaussian_approx(object))
}

#' @method kfilter ng_bsm
#' @export
kfilter.ng_bsm <- function(object, ...) {
  gaussian_kfilter(gaussian_approx(object))
}

#' @method kfilter svm
#' @export
kfilter.svm <- function(object, ...) {
  gaussian_kfilter(gaussian_approx(object))
}

#' @export
ekf <- function(object) {
  
  out <- ekf_nlg(t(object$y), object$Z, object$H, object$T, 
  object$R, object$Z_gn, object$T_gn, object$a1, object$P1, 
  object$theta, object$log_prior_pdf, object$known_params, 
  object$known_tv_params, object$n_states, object$n_etas, 
  as.integer(object$time_varying))
  
  out$at <- ts(out$at, start = start(object$y), frequency = frequency(object$y))
  out$att <- ts(out$att, start = start(object$y), frequency = frequency(object$y))
  out
}

#' @export
ekf_smoother <- function(object) {
  
 ekf_smoother_nlg(t(object$y), object$Z, object$H, object$T, 
    object$R, object$Z_gn, object$T_gn, object$a1, object$P1, 
    object$theta, object$log_prior_pdf, object$known_params, 
    object$known_tv_params, object$n_states, object$n_etas, 
    as.integer(object$time_varying))
  
}

#' @export
iekf_smoother <- function(object, max_iter = 100, conv_tol = 1e-8) {
  
  iekf_smoother_nlg(t(object$y), object$Z, object$H, object$T, 
    object$R, object$Z_gn, object$T_gn, object$a1, object$P1, 
    object$theta, object$log_prior_pdf, object$known_params, 
    object$known_tv_params, object$n_states, object$n_etas, 
    as.integer(object$time_varying), max_iter, conv_tol)
  
}