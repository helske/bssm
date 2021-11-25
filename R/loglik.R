#' Extract Log-likelihood of a State Space Model of class \code{bssm_model}
#'
#' Computes the log-likelihood of a state space model defined by \code{bssm} 
#' package.
#' 
#' @inheritParams particle_smoother
#' @param object Model of class \code{bssm_model}.
#' @param particles Number of samples for particle filter 
#' (non-negative integer). If 0, approximate log-likelihood is returned either 
#' based on the Gaussian approximation or EKF, depending on the \code{method} 
#' argument.
#' @param method Sampling method. For Gaussian and non-Gaussian models with 
#' linear dynamics,options are \code{"bsf"} (bootstrap particle filter, default 
#' for non-linear models) and \code{"psi"} (\eqn{\psi}-APF, the default for 
#' other models). For-nonlinear models option \code{"ekf"} 
#' uses EKF/IEKF-based particle filter (or just EKF/IEKF approximation in the 
#' case of \code{particles = 0}).
#' @importFrom stats logLik
#' @method logLik lineargaussian
#' @rdname logLik_bssm
#' @return A numeric value.
#' @seealso particle_smoother
#' @export
#' @references
#' Durbin, J., & Koopman, S. (2002). A Simple and Efficient Simulation 
#' Smoother for State Space Time Series Analysis. Biometrika, 89(3), 603-615. 
#' 
#' Shephard, N., & Pitt, M. (1997). Likelihood Analysis of 
#' Non-Gaussian Measurement Time Series. Biometrika, 84(3), 653-667.
#' 
#' Gordon, NJ, Salmond, DJ, Smith, AFM (1993). 
#' Novel approach to nonlinear/non-Gaussian Bayesian state estimation. 
#' IEE Proceedings-F, 140, 107-113.
#' 
#' Vihola, M, Helske, J, Franks, J. Importance sampling type estimators 
#' based on approximate marginal Markov chain Monte Carlo. 
#' Scand J Statist. 2020; 1-38. https://doi.org/10.1111/sjos.12492
#' 
#' Van Der Merwe, R, Doucet, A, De Freitas, N,  Wan, EA (2001). 
#' The unscented particle filter. 
#' In Advances in neural information processing systems, p 584-590.
#' 
#' Jazwinski, A 1970. Stochastic Processes and Filtering Theory. 
#' Academic Press.
#' 
#' Kitagawa, G (1996). Monte Carlo filter and smoother for non-Gaussian 
#' nonlinear state space models. 
#' Journal of Computational and Graphical Statistics, 5, 1-25.
#' @examples  
#' model <- ssm_ulg(y = c(1,4,3), Z = 1, H = 1, T = 1, R = 1)
#' logLik(model)
logLik.lineargaussian <- function(object, ...) {
  
     check_missingness(object)
  
  gaussian_loglik(object, model_type(object))
}

#' @method logLik nongaussian
#' @rdname logLik_bssm
#' @export
#' @examples
#' model <- ssm_ung(y = c(1,4,3), Z = 1, T = 1, R = 0.5, P1 = 2,
#'   distribution = "poisson")
#'   
#' model2 <- bsm_ng(y = c(1,4,3), sd_level = 0.5, P1 = 2,
#'   distribution = "poisson")
#'   
#' logLik(model, particles = 0)
#' logLik(model2, particles = 0)
#' logLik(model, particles = 10, seed = 1)
#' logLik(model2, particles = 10, seed = 1)
logLik.nongaussian <- function(object, particles, method = "psi", 
  max_iter = 100, conv_tol = 1e-8, 
  seed = sample(.Machine$integer.max, size = 1), ...) {
  
     check_missingness(object)
  
  object$max_iter <- check_intmax(max_iter, "max_iter", positive = FALSE)
  object$conv_tol <- check_positive_real(conv_tol, "conv_tol")
  
  if (missing(particles)) {
    nsim <- eval(match.call(expand.dots = TRUE)$nsim)
    if (!is.null(nsim)) {
      warning(paste("Argument `nsim` is deprecated. Use argument `particles`",
      "instead.", sep = " "))
      particles <- nsim
    }
  }
  
  method <- match.arg(tolower(method), c("psi", "bsf", "spdk"))
  method <- pmatch(method, c("psi", "bsf", "spdk"))
  if (method == 2 && particles == 0) 
    stop("'particles' must be positive for bootstrap filter.")
  
  object$distribution <- pmatch(object$distribution,
    c("svm", "poisson", "binomial", "negative binomial", "gamma", "gaussian"), 
    duplicates.ok = TRUE) - 1
  
  nongaussian_loglik(object, particles, method, seed, model_type(object))
}
#' @method logLik ssm_nlg
#' @rdname logLik_bssm
#' @export
logLik.ssm_nlg <- function(object, particles, method = "bsf",
  max_iter = 100, conv_tol = 1e-8, iekf_iter = 0,
  seed = sample(.Machine$integer.max, size = 1), ...) {
  
     check_missingness(object)
  
  if (missing(particles)) {
    nsim <- eval(match.call(expand.dots = TRUE)$nsim)
    if (!is.null(nsim)) {
      warning(paste("Argument `nsim` is deprecated. Use argument `particles`", 
      "instead.", sep = " "))
      particles <- nsim
    }
  }
  
  method <- match.arg(method, c("psi", "bsf", "ekf"))
  if (method == "bsf" && particles == 0) 
    stop("'particles' must be positive for bootstrap filter.")
  method <- pmatch(method,  c("psi", "bsf", NA, "ekf"))
 
  max_iter <- check_intmax(max_iter, "max_iter", positive = FALSE)
  conv_tol <- check_positive_real(conv_tol, "conv_tol")
  iekf_iter <- check_intmax(iekf_iter, "iekf_iter", positive = FALSE)
  seed <- check_intmax(seed, "seed", FALSE, max = .Machine$integer.max)
  
  nonlinear_loglik(t(object$y), object$Z, object$H, object$T, 
    object$R, object$Z_gn, object$T_gn, object$a1, object$P1, 
    object$theta, object$log_prior_pdf, object$known_params, 
    object$known_tv_params, object$n_states, object$n_etas, 
    as.integer(object$time_varying), particles, seed,
    max_iter, conv_tol, iekf_iter, method)
}
#' @param L Integer  defining the discretization level defined as (2^L). 
#' @method logLik ssm_sde
#' @rdname logLik_bssm
#' @export
logLik.ssm_sde <- function(object, particles, L,
  seed = sample(.Machine$integer.max, size = 1), ...) {
  
     check_missingness(object)
  
  if (L <= 0) stop("Discretization level L must be larger than 0.")
  if (missing(particles)) {
    nsim <- eval(match.call(expand.dots = TRUE)$nsim)
    if (!is.null(nsim)) {
      warning(paste("Argument `nsim` is deprecated. Use argument `particles`", 
      "instead.", sep = " "))
      particles <- nsim
    }
  }
  seed <- check_intmax(seed, "seed", FALSE, max = .Machine$integer.max)
  
  loglik_sde(object$y, object$x0, object$positive, 
    object$drift, object$diffusion, object$ddiffusion, 
    object$prior_pdf, object$obs_pdf, object$theta, 
    particles, L, seed)
}
