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
  
  out <- gssm_fast_smoother(object$y)
  colnames(out) <- names(object$a1)
  ts(out, start = start(object$y), frequency = frequency(object$y))
}
#' @method fast_smoother bsm
#' @export
fast_smoother.bsm <- function(object, ...) {
  
  out <- bsm_fast_smoother(object)
  colnames(out) <- names(object$a1)
  ts(out, start = start(object$y), frequency = frequency(object$y))
}
#' @method fast_smoother ngssm
#' @export
fast_smoother.ngssm <- function(object, ...) {
  
  object$distribution <- pmatch(object$distribution,
    c("poisson", "binomial", "negative binomial"))
  
  out <- ngssm_fast_smoother(object, object$init_signal)
  colnames(out) <- names(object$a1)
  ts(out, start = start(object$y), frequency = frequency(object$y))
}
#' @method fast_smoother ng_bsm
#' @export
fast_smoother.ng_bsm <- function(object, ...) {
  
  object$distribution <- pmatch(object$distribution, c("poisson", "binomial", "negative binomial"))
  
  out <- ng_bsm_fast_smoother(object, object$init_signal)
  colnames(out) <- names(object$a1)
  ts(out, start = start(object$y), frequency = frequency(object$y))
}
#' @method fast_smoother svm
#' @export
fast_smoother.svm <- function(object, ...) {
  
  object$distribution <- 0
  object$phi <- rep(object$sigma, length(object$y))
  
  out <- svm_fast_smoother(object, object$init_signal)
  colnames(out) <- names(object$a1)
  ts(out, start = start(object$y), frequency = frequency(object$y))
}
#' @export
#' @rdname smoother
smoother <- function(object, ...) {
  UseMethod("smoother", object)
}
#' @method smoother gssm
#' @export
smoother.gssm <- function(object, ...) {
  
  out <- gssm_smoother(object)
  colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <- names(object$a1)
  out$alphahat <- ts(out$alphahat, start = start(object$y), frequency = frequency(object$y))
  out
}
#' @method smoother bsm
#' @export
smoother.bsm <- function(object, ...) {
  
  out <- bsm_smoother(object)
  colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <- names(object$a1)
  out$alphahat <- ts(out$alphahat, start = start(object$y), frequency = frequency(object$y))
  out
}
#' @method smoother ngssm
#' @export
smoother.ngssm <- function(object, ...) {
  
  object$distribution <- pmatch(object$distribution,
    c("poisson", "binomial", "negative binomial"))
  
  out <- ngssm_smoother(object, object$init_signal)
  colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <- names(object$a1)
  out$alphahat <- ts(out$alphahat, start = start(object$y), frequency = frequency(object$y))
  out
}
#' @method smoother ng_bsm
#' @export
smoother.ng_bsm <- function(object, ...) {
  
  object$distribution <- pmatch(object$distribution, c("poisson", "binomial", "negative binomial"))
  
  out <- ng_bsm_smoother(object, object$init_signal)
  colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <- names(object$a1)
  out$alphahat <- ts(out$alphahat, start = start(object$y), frequency = frequency(object$y))
  out
}
#' @method smoother svm
#' @export
smoother.svm <- function(object, ...) {
  
  object$distribution <- 0L
  object$phi <- rep(object$sigma, length(object$y))
  
  out <- svm_smoother(object, object$init_signal)
  colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <- names(object$a1)
  out$alphahat <- ts(out$alphahat, start = start(object$y), frequency = frequency(object$y))
  out
}