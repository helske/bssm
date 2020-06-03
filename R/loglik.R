#' Log-likelihood of the State Space Model
#'
#' Computes the log-likelihood of the state space model of \code{bssm} package.
#' 
#' @param object Model model.
#' @param nsim Number of samples for importance sampling. If 0, approximate log-likelihood is returned.
#' See vignette for details.
#' @param method Method for computing the log-likelihood of non-Gaussian/non-linear model. 
#' Method \code{"spdk"} uses the importance sampling approach by 
#' Shephard and Pitt (1997), and Durbin and Koopman (1997). 
#' \code{"psi"} (the default for linear-Gaussian signals) uses psi-auxiliary filter and 
#' \code{"bsf"} bootstrap filter (default for general non-linear Gaussian models).
#' @param seed Seed for the random number generator. Compared to other functions of the package, the
#' default seed is fixed (as 1) in order to work properly in numerical optimization algorithms.
#' @param max_iter Maximum number of iterations.
#' @param conv_tol Tolerance parameter.
#' @param ... Ignored.
#' @importFrom stats logLik
#' @method logLik gaussian
#' @rdname logLik
#' @export
#' @examples
#' model <- ssm_ulg(y = c(1,4,3), Z = 1, H = 1, T = 1, R = 1)
#' logLik(model)
logLik.gaussian <- function(object, ...) {
  gaussian_loglik(object, model_type(object))
}

#' @method logLik nongaussian
#' @rdname logLik
#' @export
#' @examples 
#' model <- ssm_ung(y = c(1,4,3), Z = 1, T = 1, R = 0.5, P1 = 2,
#'   distribution = "poisson")
#'   
#' model2 <- bsm_ng(y = c(1,4,3), sd_level = 0.5, P1 = 2,
#'   distribution = "poisson")
#' logLik(model, nsim = 0)
#' logLik(model2, nsim = 0)
#' logLik(model, nsim = 10)
#' logLik(model2, nsim = 10)
logLik.nongaussian <- function(object, nsim, method = "psi", seed = 1, 
  max_iter = 100, conv_tol = 1e-8, ...) {
  
  object$max_iter <- max_iter
  object$conv_tol <- conv_tol
  method <- pmatch(method, c("psi", "bsf", "spdk"))
  if (method == 2 & nsim == 0) stop("'nsim' must be positive for bootstrap filter.")
  
  object$distribution <- pmatch(object$distribution,
    c("svm", "poisson", "binomial", "negative binomial", "gamma", "gaussian"), 
    duplicates.ok = TRUE) - 1
  
  nongaussian_loglik(object, nsim, method, seed, model_type(object))
}

#' @method logLik ssm_nlg
#' @export
logLik.ssm_nlg <- function(object, nsim, method = "bsf", seed = 1, 
  max_iter = 100, conv_tol = 1e-8, iekf_iter = 0, ...) {
  
  method <- pmatch(method,  c("psi", "bsf", "ekf"))
  if (method != 3 & nsim == 0) 
    stop("'nsim' must be positive for particle filter based log-likelihood estimation.")
  nonlinear_loglik(t(object$y), object$Z, object$H, object$T, 
    object$R, object$Z_gn, object$T_gn, object$a1, object$P1, 
    object$theta, object$log_prior_pdf, object$known_params, 
    object$known_tv_params, object$n_states, object$n_etas, 
    as.integer(object$time_varying), nsim, seed,
    max_iter, conv_tol, iekf_iter, pmatch(method, c("psi", "bsf", "ekf")),
    default_update_fn, default_prior_fn)
}


#' @method logLik ssm_sde
#' @export
logLik.ssm_sde <- function(object, nsim, L, seed = 1, ...) {
  if(L <= 0) stop("Discretization level L must be larger than 0.")
  loglik_sde(object$y, object$x0, object$positive, 
    object$drift, object$diffusion, object$ddiffusion, 
    object$prior_pdf, object$obs_pdf, object$theta, 
    nsim, L, seed)
}


