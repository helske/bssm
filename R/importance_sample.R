#' Importance Sampling from non-Gaussian State Space Model
#'
#' Returns \code{nsim} samples from the approximating Gaussian model with corresponding
#' (scaled) importance weights.
#' @param model of class \code{bsm_ng}, \code{ar1_ng} \code{svm} or \code{ssm_ung}.
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
importance_sample <- function(model, nsim, use_antithetic, 
  max_iter, conv_tol, seed, ...) {
  UseMethod("importance_sample", model)
}
#' @method importance_sample nongaussian
#' @rdname importance_sample
#' @export
importance_sample.nongaussian <- function(model, nsim, use_antithetic = TRUE, 
  max_iter = 100, conv_tol = 1e-8, seed = sample(.Machine$integer.max, size = 1), ...) {

  if(inherits(model, "ssm_mng")) stop("Importance sampling is supported only for univariate models. ")
  model$max_iter <- max_iter
  model$conv_tol <- conv_tol
  model$distribution <- 
    pmatch(model$distribution,  
      c("svm", "poisson", "binomial", "negative binomial", "gamma", "gaussian")) - 1
  out <- importance_sample_ung(model, nsim, use_antithetic, seed, model_type(model))
  rownames(out$alpha) <- names(model$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}