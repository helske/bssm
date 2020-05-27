#' Importance Sampling from non-Gaussian State Space Model
#'
#' Returns \code{nsim} samples from the approximating Gaussian model with corresponding
#' (scaled) importance weights.
#' @param object of class \code{bsm_ng}, \code{svm} or \code{ssm_ung}.
#' @param nsim Number of samples.
#' @param use_antithetic Logical. If \code{TRUE} (default), use antithetic 
#' variable for location in simulation smoothing.
#' @param max_iter Maximum number of iterations used for the approximation.
#' @param conv_tol Convergence threshold for the approximation. Approximation is 
#' claimed to be converged when the mean squared difference of the modes is 
#' less than \code{conv_tol}.
#' @param seed Seed for the random number generator.
#' @param ... Ignored.
#' @export
#' @rdname importance_sample
importance_sample <- function(object, nsim, use_antithetic, 
  max_iter, conv_tol, seed, ...) {
  UseMethod("importance_sample", object)
}
#' @method importance_sample ssm_ung
#' @rdname importance_sample
#' @export
importance_sample.ssm_ung <- function(object, nsim, use_antithetic = TRUE, 
  max_iter = 100, conv_tol = 1e-8, seed = sample(.Machine$integer.max, size = 1), ...) {

  object$distribution <- pmatch(object$distribution, c("poisson", "binomial", "negative binomial"))
  out <- importance_sample_ung(object, nsim, use_antithetic, object$initial_mode, 
    max_iter, conv_tol, seed, 1L)
  rownames(out$alpha) <- names(object$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}
#' @method importance_sample bsm_ng
#' @rdname importance_sample
#' @export
importance_sample.bsm_ng <- function(object, nsim, use_antithetic = TRUE, 
  max_iter = 100, conv_tol = 1e-8, seed = sample(.Machine$integer.max, size = 1), ...) {
  
  object$distribution <- pmatch(object$distribution, c("poisson", "binomial", "negative binomial"))
  out <- importance_sample_ung(object, nsim, use_antithetic, object$initial_mode, 
    max_iter, conv_tol, seed, 2L)
  rownames(out$alpha) <- names(object$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}
#' @method importance_sample svm
#' @rdname importance_sample
#' @export
importance_sample.svm <- function(object, nsim, use_antithetic = TRUE, 
  max_iter = 100, conv_tol = 1e-8, seed = sample(.Machine$integer.max, size = 1), ...) {
  
  out <- importance_sample_ung(object, nsim, use_antithetic, object$initial_mode, 
    max_iter, conv_tol, seed, 3L)
  rownames(out$alpha) <- names(object$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}
#' @method importance_sample uar1_ng
#' @rdname importance_sample
#' @export
importance_sample.uar1_ng <- function(object, nsim, use_antithetic = TRUE, 
  max_iter = 100, conv_tol = 1e-8, seed = sample(.Machine$integer.max, size = 1), ...) {
  
  object$distribution <- pmatch(object$distribution, c("poisson", "binomial", "negative binomial"))
  out <- importance_sample_ung(object, nsim, use_antithetic, object$initial_mode, 
    max_iter, conv_tol, seed, 4L)
  rownames(out$alpha) <- names(object$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}
