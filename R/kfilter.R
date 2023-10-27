#' Kalman Filtering
#'
#' Function \code{kfilter} runs the Kalman filter for the given model, 
#' and returns the filtered estimates and one-step-ahead predictions of the 
#' states \eqn{\alpha_t} given the data up to time \eqn{t}.
#'
#' For non-Gaussian models, the filtering is based on the approximate 
#' Gaussian model.
#'
#' @param model Model of class \code{lineargaussian}, \code{nongaussian} or 
#' \code{ssm_nlg}.
#' @param ... Ignored.
#' @return List containing the log-likelihood 
#' (approximate in non-Gaussian case), one-step-ahead predictions \code{at} 
#' and filtered estimates \code{att} of states, and the corresponding 
#' variances \code{Pt} and \code{Ptt} up to the time point n+1 where n is the 
#' length of the input time series.
#' @seealso \code{\link{bootstrap_filter}}
#' @export
#' @rdname kfilter
kfilter <- function(model, ...) {
  UseMethod("kfilter", model)
}
#' @method kfilter lineargaussian
#' @rdname kfilter
#' @export
#' @examples
#' x <- cumsum(rnorm(20))
#' y <- x + rnorm(20, sd = 0.1)
#' model <- bsm_lg(y, sd_level = 1, sd_y = 0.1)
#' ts.plot(cbind(y, x, kfilter(model)$att), col = 1:3)
kfilter.lineargaussian <- function(model, ...) {
  
  check_missingness(model)
  
  out <- gaussian_kfilter(model, model_type = model_type(model))
  colnames(out$at) <- colnames(out$att) <- colnames(out$Pt) <-
    colnames(out$Ptt) <- rownames(out$Pt) <- rownames(out$Ptt) <- 
    names(model$a1)
  out$at <- ts(out$at, start = start(model$y), frequency = frequency(model$y))
  out$att <- ts(out$att, start = start(model$y), frequency = frequency(model$y))
  out
}

#' @method kfilter nongaussian
#' @rdname kfilter
#' @export
kfilter.nongaussian <- function(model, ...) {
  kfilter(gaussian_approx(model))
}

#' (Iterated) Extended Kalman Filtering
#'
#' Function \code{ekf} runs the (iterated) extended Kalman filter for the given 
#' non-linear Gaussian model of class \code{ssm_nlg}, 
#' and returns the filtered estimates and one-step-ahead predictions of the 
#' states \eqn{\alpha_t} given the data up to time \eqn{t}.
#'
#' @param model Model of class \code{ssm_nlg}.
#' @param iekf_iter Non-negative integer. The default zero corresponds to 
#' normal EKF, whereas \code{iekf_iter > 0} corresponds to iterated EKF 
#' with \code{iekf_iter} iterations.
#' @return List containing the log-likelihood,
#' one-step-ahead predictions \code{at} and filtered
#' estimates \code{att} of states, and the corresponding variances \code{Pt} and
#'  \code{Ptt}.
#' @export
#' @rdname ekf
#' @export
#' @examples
#' \donttest{ # Takes a while on CRAN
#' set.seed(1)
#' mu <- -0.2
#' rho <- 0.7
#' sigma_y <- 0.1
#' sigma_x <- 1
#' x <- numeric(50)
#' x[1] <- rnorm(1, mu, sigma_x / sqrt(1 - rho^2))
#' for(i in 2:length(x)) {
#'   x[i] <- rnorm(1, mu * (1 - rho) + rho * x[i - 1], sigma_x)
#' }
#' y <- rnorm(50, exp(x), sigma_y)
#' 
#' pntrs <- cpp_example_model("nlg_ar_exp")
#' 
#' model_nlg <- ssm_nlg(y = y, a1 = pntrs$a1, P1 = pntrs$P1, 
#'   Z = pntrs$Z_fn, H = pntrs$H_fn, T = pntrs$T_fn, R = pntrs$R_fn, 
#'   Z_gn = pntrs$Z_gn, T_gn = pntrs$T_gn,
#'   theta = c(mu= mu, rho = rho, 
#'     log_sigma_x = log(sigma_x), log_sigma_y = log(sigma_y)), 
#'   log_prior_pdf = pntrs$log_prior_pdf,
#'   n_states = 1, n_etas = 1, state_names = "state")
#'
#' out_ekf <- ekf(model_nlg, iekf_iter = 0)
#' out_iekf <- ekf(model_nlg, iekf_iter = 5)
#' ts.plot(cbind(x, out_ekf$att, out_iekf$att), col = 1:3)
#' }
ekf <- function(model, iekf_iter = 0) {
  
  check_missingness(model)
  
  iekf_iter <- check_intmax(iekf_iter, "iekf_iter", positive = FALSE)
  
  out <- ekf_nlg(t(model$y), model$Z, model$H, model$T, 
    model$R, model$Z_gn, model$T_gn, model$a1, model$P1, 
    model$theta, model$log_prior_pdf, model$known_params, 
    model$known_tv_params, model$n_states, model$n_etas, 
    as.integer(model$time_varying), iekf_iter)
  
  colnames(out$at) <- colnames(out$att) <- colnames(out$Pt) <-
    colnames(out$Ptt) <- rownames(out$Pt) <- rownames(out$Ptt) <- 
    model$state_names
  
  out$at <- ts(out$at, start = start(model$y), frequency = frequency(model$y))
  out$att <- ts(out$att, start = start(model$y), frequency = frequency(model$y))
  out
}
#' Unscented Kalman Filtering
#'
#' Function \code{ukf} runs the unscented Kalman filter for the given 
#' non-linear Gaussian model of class \code{ssm_nlg}, 
#' and returns the filtered estimates and one-step-ahead predictions of the 
#' states \eqn{\alpha_t} given the data up to time \eqn{t}.
#'
#' @param model Model of class \code{ssm_nlg}.
#' @param alpha Positive tuning parameter of the UKF. Default is 0.001. Smaller 
#' the value, closer the sigma point are to the mean of the state. 
#' @param beta Non-negative tuning parameter of the UKF. The default value is 
#' 2, which is optimal for Gaussian states.
#' @param kappa Non-negative tuning parameter of the UKF, which also affects 
#' the spread of sigma points. Default value is 0.
#' @return List containing the log-likelihood,
#' one-step-ahead predictions \code{at} and filtered
#' estimates \code{att} of states, and the corresponding variances \code{Pt} and
#'  \code{Ptt}.
#' @export
#' @rdname ukf
#' @export
#' @examples
#' \donttest{ # Takes a while on CRAN
#' set.seed(1)
#' mu <- -0.2
#' rho <- 0.7
#' sigma_y <- 0.1
#' sigma_x <- 1
#' x <- numeric(50)
#' x[1] <- rnorm(1, mu, sigma_x / sqrt(1 - rho^2))
#' for(i in 2:length(x)) {
#'   x[i] <- rnorm(1, mu * (1 - rho) + rho * x[i - 1], sigma_x)
#' }
#' y <- rnorm(50, exp(x), sigma_y)
#' 
#' pntrs <- cpp_example_model("nlg_ar_exp")
#' 
#' model_nlg <- ssm_nlg(y = y, a1 = pntrs$a1, P1 = pntrs$P1, 
#'   Z = pntrs$Z_fn, H = pntrs$H_fn, T = pntrs$T_fn, R = pntrs$R_fn, 
#'   Z_gn = pntrs$Z_gn, T_gn = pntrs$T_gn,
#'   theta = c(mu= mu, rho = rho, 
#'     log_sigma_x = log(sigma_x), log_sigma_y = log(sigma_y)), 
#'   log_prior_pdf = pntrs$log_prior_pdf,
#'   n_states = 1, n_etas = 1, state_names = "state")
#'
#' out_iekf <- ekf(model_nlg, iekf_iter = 5)
#' out_ukf <- ukf(model_nlg, alpha = 0.01, beta = 2, kappa = 1)
#' ts.plot(cbind(x, out_iekf$att, out_ukf$att), col = 1:3)
#' }
ukf <- function(model, alpha = 0.001, beta = 2, kappa = 0) {
  
  check_missingness(model)
  
  if (alpha <= 0) stop("Parameter 'alpha' should be positive. ")
  if (beta < 0) stop("Parameter 'beta' should be non-negative. ")
  if (kappa < 0) stop("Parameter 'kappa' should be non-negative. ")
  
  out <- ukf_nlg(t(model$y), model$Z, model$H, model$T, 
    model$R, model$Z_gn, model$T_gn, model$a1, model$P1, 
    model$theta, model$log_prior_pdf, model$known_params, 
    model$known_tv_params, model$n_states, model$n_etas, 
    as.integer(model$time_varying),
    alpha, beta, kappa)
  
  colnames(out$at) <- colnames(out$att) <- colnames(out$Pt) <-
    colnames(out$Ptt) <- rownames(out$Pt) <- rownames(out$Ptt) <- 
    model$state_names
  
  out$at <- ts(out$at, start = start(model$y), frequency = frequency(model$y))
  out$att <- ts(out$att, start = start(model$y), frequency = frequency(model$y))
  out
}
