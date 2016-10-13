#' Gaussian approximation of non-Gaussian state space model
#'
#' Returns the approximating Gaussian model.
#' @param object model object.
#' @param max_iter Maximum number of iterations.
#' @param conv_tol Tolerance parameter. ***Document properly later!***
#' @param ... Ignored.
#' @export
#' @rdname gaussian_approx
gaussian_approx <- function(object, max_iter, conv_tol, ...) {
  UseMethod("gaussian_approx", object)
}
#' @method gaussian_approx ngssm
#' @export
gaussian_approx.ngssm<- function(object, max_iter = 100, conv_tol = 1e-8, ...) {
  
  object$distribution <- pmatch(object$distribution, c("poisson", "binomial", "negative binomial"))
  ngssm_approx_model(object, object$init_signal, max_iter, conv_tol)
}
#' @method gaussian_approx ng_bsm
#' @rdname gaussian_approx
#' @export
gaussian_approx.ng_bsm <- function(object, max_iter =  100, conv_tol = 1e-8, ...) {

  object$distribution <- pmatch(object$distribution, c("poisson", "binomial", "negative binomial"))
  ng_bsm_approx_model(object, object$init_signal, max_iter, conv_tol)
}
#' @method gaussian_approx svm
#' @export
gaussian_approx.svm <- function(object, max_iter = 100, conv_tol = 1e-8, ...) {
  
  svm_approx_model(object, object$init_signal, max_iter, conv_tol)
}