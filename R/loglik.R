#' Log-likelihood of the State Space Model
#'
#' Computes the log-likelihood of the state space model of \code{bssm} package.
#' 
#' @param object Model object.
#' @param nsim_states Number of samples for importance sampling. If 0, approximate log-likelihood is returned.
#' See vignette for details.
#' @param method Method for computing the log-likelihood of non-Gaussian/non-linear model. 
#' Method \code{"spdk"} uses the importance sampling approach by 
#' Shephard and Pitt (1997), and Durbin and Koopman (1997). \code{"psi"} (the default for linear-Gaussian signals) uses psi-auxiliary filter and 
#' \code{"bsf"} bootstrap filter (default for general non-linear Gaussian models).
#' @param seed Seed for the random number generator. Compared to other functions of the package, the
#' default seed is fixed (as 1) in order to work properly in numerical optimization algorithms.
#' @param max_iter Maximum number of iterations.
#' @param conv_tol Tolerance parameter.
#' @param ... Ignored.
#' @importFrom stats logLik
#' @method logLik gssm
#' @rdname logLik
#' @export
logLik.gssm <- function(object, ...) {
  gaussian_loglik(object, model_type = 1L)
}
#' @method logLik bsm
#' @export
logLik.bsm <- function(object, ...) {
  gaussian_loglik(object, model_type = 2L)
}
#' @method logLik mv_gssm
#' @export
logLik.mv_gssm <- function(object, ...) {
  gaussian_loglik(object, model_type = -1L)
}
#' @method logLik ngssm
#' @rdname logLik
#' @export
logLik.ngssm <- function(object, nsim_states, method = "psi", seed = 1, 
  max_iter = 100, conv_tol = 1e-8, ...) {
  
  method <- match.arg(method,  c("psi", "bsf", "spdk"))
  if (method == "bsf" & nsim_states == 0) stop("'nsim_state' must be positive for bootstrap filter.")
  object$distribution <- pmatch(object$distribution,
    c("poisson", "binomial", "negative binomial"))
  
  nongaussian_loglik(object, object$initial_mode, nsim_states, 
    pmatch(method,  c("psi", "bsf", "spdk")), seed, max_iter, conv_tol, model_type = 1L)
}
#' @method logLik ng_bsm
#' @export
logLik.ng_bsm <- function(object, nsim_states, method = "psi", seed = 1,
  max_iter = 100, conv_tol = 1e-8, ...) {
  
  method <- match.arg(method,  c("psi", "bsf", "spdk"))
  if (method == "bsf" & nsim_states == 0) stop("'nsim_state' must be positive for bootstrap filter.")
  object$distribution <- pmatch(object$distribution, c("poisson", "binomial", "negative binomial"))
  
  nongaussian_loglik(object, object$initial_mode, nsim_states, 
    pmatch(method,  c("psi", "bsf", "spdk")), seed, max_iter, conv_tol, model_type = 2L)
}
#' @method logLik svm
#' @export
logLik.svm <- function(object, nsim_states, method = "psi", seed = 1,
  max_iter = 100, conv_tol = 1e-8, ...) {
  
  method <- match.arg(method,  c("psi", "bsf", "spdk"))
  if (method == "bsf" & nsim_states == 0) stop("'nsim_states' must be positive for bootstrap filter.")
  nongaussian_loglik(object, object$initial_mode, nsim_states, 
    pmatch(method,  c("psi", "bsf", "spdk")), seed, max_iter, conv_tol, model_type = 3L)
}

#' @method logLik nlg_ssm
#' @export
logLik.nlg_ssm <- function(object, nsim_states, method = "bsf", seed = 1, 
  max_iter = 100, conv_tol = 1e-8, iekf_iter = 0, ...) {
  
  method <- match.arg(method,  c("psi", "bsf", "apf", "ekf", "psi_df"))
  if (method != "ekf" & nsim_states == 0) 
    stop("'nsim_states' must be positive for particle filter based log-likelihood estimation.")
  nonlinear_loglik(t(object$y), object$Z, object$H, object$T, 
    object$R, object$Z_gn, object$T_gn, object$a1, object$P1, 
    object$theta, object$log_prior_pdf, object$known_params, 
    object$known_tv_params, object$n_states, object$n_etas, 
    as.integer(object$time_varying), nsim_states, seed,
    max_iter, conv_tol, iekf_iter, pmatch(method, c("psi", "bsf", "apf", "ekf", "psi_df")))
}
