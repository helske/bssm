#' Log-likelihood of the State Space Model
#'
#' Computes the log-likelihood of the state space model of \code{bssm} package.
#' 
#' @param object Model object.
#' @param nsim_states Number of samples for importance sampling. If 0, approximate log-likelihood is returned.
#' See vignette for details.
#' @param seed Seed for the random number generator. Compared to other functions of the package, the
#' default seed is fixed (as 1) in order to work properly in numerical optimization algorithms.
#' @param ... Ignored.
#' @importFrom stats logLik
#' @method logLik gssm
#' @rdname logLik
#' @export
logLik.gssm <- function(object, ...) {
  gssm_loglik(object)
}
#' @method logLik bsm
#' @export
logLik.bsm <- function(object, ...) {
  bsm_loglik(object)
}
#' @method logLik ngssm
#' @rdname logLik
#' @export
logLik.ngssm <- function(object, nsim_states, seed = 1, ...) {

  object$distribution <- pmatch(object$distribution,
    c("poisson", "binomial", "negative binomial"))

  ngssm_loglik(object, object$init_signal, nsim_states, seed)
}
#' @method logLik ng_bsm
#' @export
logLik.ng_bsm <- function(object, nsim_states, seed = 1, ...) {
  
  object$distribution <- pmatch(object$distribution, c("poisson", "binomial", "negative binomial"))
  ng_bsm_loglik(object, object$init_signal, nsim_states, seed)
}
#' @method logLik svm
#' @export
logLik.svm <- function(object, nsim_states, seed = 1, ...) {
  
  object$distribution <- 0L
  object$phi <- rep(object$sigma, length(object$y))
  svm_loglik(object, object$init_signal, nsim_states, seed)
}