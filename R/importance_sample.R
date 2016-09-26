#' Importance Sampling from non-Gaussian State Space Model
#'
#' Returns \code{nsim} samples from the approximating Gaussian model with corresponding
#' (scaled) importance weights.
#' @param object of class \code{ng_bsm}, \code{svm} or \code{ngssm}.
#' @param nsim Number of samples.
#' @param seed Seed for the random number generator.
#' @param ... Ignored.
#' @export
#' @rdname importance_sample
importance_sample <- function(object, nsim, seed, ...) {
  UseMethod("importance_sample", object)
}
#' @method importance_sample ngssm
#' @rdname importance_sample
#' @export
importance_sample.ngssm <- function(object, nsim,  seed = sample(.Machine$integer.max, size = 1), ...) {

  object$distribution <- pmatch(object$distribution, c("poisson", "binomial", "negative binomial"))
  ngssm_importance_sample(object, object$init_signal, nsim, seed)
}
#' @method importance_sample ng_bsm
#' @rdname importance_sample
#' @export
importance_sample.ng_bsm <- function(object, nsim, seed = sample(.Machine$integer.max, size = 1), ...) {

  object$distribution <- pmatch(object$distribution, c("poisson", "binomial", "negative binomial"))
  ng_bsm_importance_sample(object, object$init_signal, nsim, seed)
}
#' @method importance_sample svm
#' @rdname importance_sample
#' @export
importance_sample.svm <- function(object, nsim, seed = sample(.Machine$integer.max, size = 1), ...) {

  object$distribution <- 0L
  object$phi <- rep(object$sigma, length(object$y))
  
  svm_importance_sample(object, object$init_signal, nsim, seed)
}