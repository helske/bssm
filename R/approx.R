#' Gaussian approximation of non-Gaussian state space model
#'
#' Returns the approximating Gaussian model.
#' @param object model object.
#' @param max_iter Maximum number of iterations.
#' @param conv_tol Tolerance parameter.
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
  out <- gaussian_approx_model(object, object$initial_mode, max_iter, conv_tol, model_type = 1L)
  gssm(y = out$y, Z = object$Z, H = out$H, T = object$T, R = object$R, a1 = object$a1, P1 = object$P1,
    xreg = if(ncol(object$xreg) > 0) object$xreg, beta = if(ncol(object$xreg) > 0) object$beta,
    obs_intercept = object$obs_intercept, state_intercept = object$state_intercept, 
    state_names = names(object$a1))
}
#' @method gaussian_approx ng_bsm
#' @rdname gaussian_approx
#' @export
gaussian_approx.ng_bsm <- function(object, max_iter = 100, conv_tol = 1e-8, ...) {

  object$distribution <- pmatch(object$distribution, c("poisson", "binomial", "negative binomial"))
  out <- gaussian_approx_model(object, object$initial_mode, max_iter, conv_tol, model_type = 2L)
  gssm(y = out$y, Z = object$Z, H = out$H, T = object$T, R = object$R, a1 = object$a1, P1 = object$P1,
    xreg = if(ncol(object$xreg) > 0) object$xreg, beta = if(ncol(object$xreg) > 0) object$beta,
    obs_intercept = object$obs_intercept, state_intercept = object$state_intercept, 
    state_names = names(object$a1))
}
#' @method gaussian_approx svm
#' @export
gaussian_approx.svm <- function(object, max_iter = 100, conv_tol = 1e-8, ...) {
  
  out <- gaussian_approx_model(object, object$initial_mode, max_iter, conv_tol, model_type = 3L)
  gssm(y = out$y, Z = object$Z, H = out$H, T = object$T, R = object$R, a1 = object$a1, P1 = object$P1,
    xreg = if(ncol(object$xreg) > 0) object$xreg, beta = if(ncol(object$xreg) > 0) object$beta,
    obs_intercept = object$obs_intercept, state_intercept = object$state_intercept, 
    state_names = names(object$a1))
}