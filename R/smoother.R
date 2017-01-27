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

#' @method smoother ngssm
#' @export
smoother.ngssm <- function(object, ...) {
  gaussian_smoother(gaussian_approx(object))
}

#' @method smoother ng_bsm
#' @export
smoother.ng_bsm <- function(object, ...) {
  gaussian_smoother(gaussian_approx(object))
}

#' @method smoother svm
#' @export
smoother.svm <- function(object, ...) {
  gaussian_smoother(gaussian_approx(object))
}
