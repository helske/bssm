#' Log-likelihood of a Gaussian State Space Model
#'
#' Computes the log-likelihood of a linear-Gaussian state space model of 
#' \code{bssm} package.
#' 
#' @param object Model model.
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

#' Log-likelihood of a Non-Gaussian State Space Model
#'
#' Computes the log-likelihood of a non-Gaussian state space model of 
#' \code{bssm} package.
#' 
#' @param object Model model.
#' @param particles Number of samples for particle filter or 
#' importance sampling. If 0, 
#' approximate log-likelihood based on the Gaussian approximation is returned.
#' @param method Sampling method, default is psi-auxiliary filter 
#' (\code{"psi"}). Other choices are \code{"bsf"} bootstrap particle filter, 
#' and \code{"spdk"}, which uses the importance sampling approach by 
#' Shephard and Pitt (1997) and Durbin and Koopman (1997). 
#' @param max_iter Maximum number of iterations for Gaussian approximation 
#' algorithm.
#' @param conv_tol Tolerance parameter for the approximation algorithm.
#' @param seed Seed for the random number generator.
#' @param ... Ignored.
#' @method logLik nongaussian
#' @export
#' @references
#' Durbin, J., & Koopman, S. (2002). A Simple and Efficient Simulation 
#' Smoother for State Space Time Series Analysis. Biometrika, 89(3), 603-615. 
#' 
#' Shephard, N., & Pitt, M. (1997). Likelihood Analysis of 
#' Non-Gaussian Measurement Time Series. Biometrika, 84(3), 653-667.
#' @examples 
#' model <- ssm_ung(y = c(1,4,3), Z = 1, T = 1, R = 0.5, P1 = 2,
#'   distribution = "poisson")
#'   
#' model2 <- bsm_ng(y = c(1,4,3), sd_level = 0.5, P1 = 2,
#'   distribution = "poisson")
#' logLik(model, particles = 0)
#' logLik(model2, particles = 0)
#' logLik(model, particles = 10, seed = 1)
#' logLik(model2, particles = 10, seed = 1)
logLik.nongaussian <- function(object, particles, method = "psi", 
  max_iter = 100, conv_tol = 1e-8, 
  seed = sample(.Machine$integer.max, size = 1),...) {
  
  object$max_iter <- max_iter
  object$conv_tol <- conv_tol
  
  if(missing(particles)) {
    nsim <- eval(match.call(expand.dots = TRUE)$nsim)
    if (!is.null(nsim)) {
      warning(paste("Argument `nsim` is deprecated. Use argument `particles`",
      "instead.", sep = " "))
      particles <- nsim
    }
  }
  
  method <- match.arg(method, c("psi", "bsf", "spdk"))
  method <- pmatch(method, c("psi", "bsf", "spdk"))
  if (method == 2 && particles == 0) 
    stop("'particles' must be positive for bootstrap filter.")
  
  object$distribution <- pmatch(object$distribution,
    c("svm", "poisson", "binomial", "negative binomial", "gamma", "gaussian"), 
    duplicates.ok = TRUE) - 1
  
  nongaussian_loglik(object, particles, method, seed, model_type(object))
}
#' Log-likelihood of a Non-linear State Space Model
#'
#' Computes the log-likelihood of a state space model of class 
#' \code{ssm_nlg} package.
#' 
#' @param object Model model.
#' @param particles Number of samples for particle filter. If 0, 
#' approximate log-likelihood is returned either based on the Gaussian 
#' approximation or EKF, depending on the \code{method} argument.
#' @param method Sampling method. Default is the bootstrap particle filter 
#' (\code{"bsf"}). Other choices are \code{"psi"} which uses 
#' psi-auxiliary filter (or approximating Gaussian model in the case of 
#' \code{particles = 0}), and \code{"ekf"} which uses EKF-based particle 
#' filter (or just EKF approximation in the case of \code{particles = 0}).
#' @param max_iter Maximum number of iterations for the gaussian approximation 
#' algorithm.
#' @param conv_tol Tolerance parameter for the approximation algorithm.
#' @param iekf_iter If \code{iekf_iter > 0}, iterated extended Kalman filter 
#' is used with
#' \code{iekf_iter} iterations in place of standard EKF. Defaults to zero.
#' @param seed Seed for the random number generator.
#' @param ... Ignored.
#' @method logLik ssm_nlg
#' @export
logLik.ssm_nlg <- function(object, particles, method = "bsf",
  max_iter = 100, conv_tol = 1e-8, iekf_iter = 0,
  seed = sample(.Machine$integer.max, size = 1), ...) {
  
  if(missing(particles)) {
    nsim <- eval(match.call(expand.dots = TRUE)$nsim)
    if (!is.null(nsim)) {
      warning(paste("Argument `nsim` is deprecated. Use argument `particles`", 
      "instead.", sep = " "))
      particles <- nsim
    }
  }
  
  method <- match.arg(method, c("psi", "bsf", "ekf"))
  if (method == "bsf" && particles == 0) 
    stop("'particles' must be positive for bootstrap particle filter.")
  method <- pmatch(method,  c("psi", "bsf", NA, "ekf"))
 
  nonlinear_loglik(t(object$y), object$Z, object$H, object$T, 
    object$R, object$Z_gn, object$T_gn, object$a1, object$P1, 
    object$theta, object$log_prior_pdf, object$known_params, 
    object$known_tv_params, object$n_states, object$n_etas, 
    as.integer(object$time_varying), particles, seed,
    max_iter, conv_tol, iekf_iter, method)
}
#' Log-likelihood of a State Space Model with SDE dynamics
#'
#' Computes the log-likelihood of a state space model of class 
#' \code{ssm_sde} package.
#' 
#' @param object Model model.
#' @param particles Number of samples for particle filter. 
#' @param L Integer  defining the discretization level defined as (2^L). 
#' @param seed Seed for the random number generator.
#' @param ... Ignored.
#' @method logLik ssm_sde
#' @export
logLik.ssm_sde <- function(object, particles, L,
  seed = sample(.Machine$integer.max, size = 1), ...) {
  if(L <= 0) stop("Discretization level L must be larger than 0.")
  if(missing(particles)) {
    nsim <- eval(match.call(expand.dots = TRUE)$nsim)
    if (!is.null(nsim)) {
      warning(paste("Argument `nsim` is deprecated. Use argument `particles`", 
      "instead.", sep = " "))
      particles <- nsim
    }
  }
  loglik_sde(object$y, object$x0, object$positive, 
    object$drift, object$diffusion, object$ddiffusion, 
    object$prior_pdf, object$obs_pdf, object$theta, 
    particles, L, seed)
}


