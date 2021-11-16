#' Importance Sampling from non-Gaussian State Space Model
#'
#' Returns \code{nsim} samples from the approximating Gaussian model with 
#' corresponding (scaled) importance weights. 
#' Probably mostly useful for comparing KFAS and bssm packages.
#' 
#' 
#' @inheritParams gaussian_approx
#' @param model Model of class \code{bsm_ng}, \code{ar1_ng} \code{svm}, 
#' \code{ssm_ung}, or \code{ssm_mng}.
#' @param nsim Number of samples (positive integer).
#' @param use_antithetic Logical. If \code{TRUE} (default), use antithetic 
#' variable for location in simulation smoothing. Ignored for \code{ssm_mng} 
#' models.
#' @param seed Seed for the random number generator (positive integer).
#' @param ... Ignored.
#' @export
#' @rdname importance_sample
#' @examples 
#' data("sexratio", package = "KFAS")
#' model <- bsm_ng(sexratio[, "Male"], sd_level = 0.001, 
#'   u = sexratio[, "Total"],
#'   distribution = "binomial")
#' 
#' imp <- importance_sample(model, nsim = 1000)
#' 
#' est <- matrix(NA, 3, nrow(sexratio))
#' for(i in 1:ncol(est)) {
#'   est[, i] <- Hmisc::wtd.quantile(exp(imp$alpha[i, 1, ]), imp$weights, 
#'     prob = c(0.05,0.5,0.95), normwt=TRUE)
#' }
#' 
#' ts.plot(t(est),lty = c(2,1,2))
#' 
importance_sample <- function(model, nsim, use_antithetic, 
  max_iter, conv_tol, seed, ...) {
  UseMethod("importance_sample", model)
}
#' @method importance_sample nongaussian
#' @rdname importance_sample
#' @export
importance_sample.nongaussian <- function(model, nsim, use_antithetic = TRUE, 
  max_iter = 100, conv_tol = 1e-8, 
  seed = sample(.Machine$integer.max, size = 1), ...) {
  
  check_missingness(model)
  
  model$max_iter <- check_integer(max_iter, "max_iter", positive = FALSE)
  model$conv_tol <- check_positive_real(conv_tol, "conv_tol")
  nsim <- check_integer(nsim, "nsim")
  
  nsamples <- ifelse(!is.null(nrow(model$y)), nrow(model$y), length(model$y)) * 
    length(model$a1) * nsim
  if (nsim > 100 & nsamples > 1e12) {
    warning(paste("Trying to sample ", nsamples, 
      "values, you might run out of memory."))
  }
  seed <- check_integer(seed, "seed", FALSE, max = .Machine$integer.max)
  model$distribution <- pmatch(model$distribution,
    c("svm", "poisson", "binomial", "negative binomial", "gamma", "gaussian"), 
    duplicates.ok = TRUE) - 1
  out <- importance_sample_ng(model, nsim, use_antithetic, seed, 
    model_type(model))
  rownames(out$alpha) <- names(model$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}
