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
importance_sample.ngssm <- function(object, nsim, use_antithetic = TRUE, 
  max_iter = 100, conv_tol = 1e-8, seed = sample(.Machine$integer.max, size = 1), ...) {

  object$distribution <- pmatch(object$distribution, c("poisson", "binomial", "negative binomial"))
  out <- importance_sample_ung(object, nsim, use_antithetic, object$initial_mode, 
    max_iter, conv_tol, seed, 1L)
  rownames(out) <- names(object$a1)
  aperm(out, c(2, 1, 3))
}
#' @method importance_sample ng_bsm
#' @rdname importance_sample
#' @export
importance_sample.ng_bsm <- function(object, nsim, use_antithetic = TRUE, 
  max_iter = 100, conv_tol = 1e-8, seed = sample(.Machine$integer.max, size = 1), ...) {
  
  object$distribution <- pmatch(object$distribution, c("poisson", "binomial", "negative binomial"))
  out <- importance_sample_ung(object, nsim, use_antithetic, object$initial_mode, 
    max_iter, conv_tol, seed, 2L)
  rownames(out) <- names(object$a1)
  aperm(out, c(2, 1, 3))
}
#' @method importance_sample svm
#' @rdname importance_sample
#' @export
importance_sample.svm <- function(object, nsim, use_antithetic = TRUE, 
  max_iter = 100, conv_tol = 1e-8, seed = sample(.Machine$integer.max, size = 1), ...) {
  
  object$distribution <- pmatch(object$distribution, c("poisson", "binomial", "negative binomial"))
  out <- importance_sample_ung(object, nsim, use_antithetic, object$initial_mode, 
    max_iter, conv_tol, seed, 3L)
  rownames(out) <- names(object$a1)
  aperm(out, c(2, 1, 3))
}