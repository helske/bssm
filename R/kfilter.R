#' Kalman Filtering
#'
#' Function \code{kfilter} runs the Kalman filter for the given model, 
#' and returns the filtered estimates and one-step-ahead predictions of the 
#' states \eqn{\alpha_t} given the data up to time \eqn{t}.
#'
#' For non-Gaussian models, the filtering is based on the approximate Gaussian model.
#'
#' @param model Model Model object.
#' @param ... Ignored.
#' @return List containing the log-likelihood (approximate in non-Gaussian case),
#' one-step-ahead predictions \code{at} and filtered
#' estimates \code{att} of states, and the corresponding variances \code{Pt} and
#'  \code{Ptt}.
#' @seealso \code{\link{bootstrap_filter}}
#' @export
#' @rdname kfilter
kfilter <- function(model, ...) {
  UseMethod("kfilter", model)
}
#' @method kfilter gaussian
#' @rdname kfilter
#' @export
#' @examples
#' x <- cumsum(rnorm(20))
#' y <- x + rnorm(20, sd = 0.1)
#' model <- bsm_lg(y, sd_level = 1, sd_y = 0.1)
#' ts.plot(cbind(y, x, kfilter(model)$att), col = 1:3)
kfilter.gaussian <- function(model, ...) {
  
  out <- gaussian_kfilter(model, model_type = model_type(model))
  colnames(out$at) <- colnames(out$att) <- colnames(out$Pt) <-
    colnames(out$Ptt) <- rownames(out$Pt) <- rownames(out$Ptt) <- names(model$a1)
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
#' @param model Model model
#' @param iekf_iter If \code{iekf_iter > 0}, iterated extended Kalman filter 
#' is used with \code{iekf_iter} iterations.
#' @return List containing the log-likelihood,
#' one-step-ahead predictions \code{at} and filtered
#' estimates \code{att} of states, and the corresponding variances \code{Pt} and
#'  \code{Ptt}.
#' @export
#' @rdname ekf
#' @export
ekf <- function(model, iekf_iter = 0) {
  
  out <- ekf_nlg(t(model$y), model$Z, model$H, model$T, 
    model$R, model$Z_gn, model$T_gn, model$a1, model$P1, 
    model$theta, model$log_prior_pdf, model$known_params, 
    model$known_tv_params, model$n_states, model$n_etas, 
    as.integer(model$time_varying), iekf_iter, 
    default_update_fn, default_prior_fn)
  
  colnames(out$at) <- colnames(out$att) <- colnames(out$Pt) <-
    colnames(out$Ptt) <- rownames(out$Pt) <- rownames(out$Ptt) <- model$state_names
  
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
#' @param model Model model
#' @param alpha,beta,kappa Tuning parameters for the UKF.
#' @return List containing the log-likelihood,
#' one-step-ahead predictions \code{at} and filtered
#' estimates \code{att} of states, and the corresponding variances \code{Pt} and
#'  \code{Ptt}.
#' @export
#' @rdname ukf
#' @export
ukf <- function(model, alpha = 1, beta = 0, kappa = 2) {
  
  out <- ukf_nlg(t(model$y), model$Z, model$H, model$T, 
    model$R, model$Z_gn, model$T_gn, model$a1, model$P1, 
    model$theta, model$log_prior_pdf, model$known_params, 
    model$known_tv_params, model$n_states, model$n_etas, 
    as.integer(model$time_varying),
    alpha, beta, kappa, default_update_fn, default_prior_fn)
  
  colnames(out$at) <- colnames(out$att) <- colnames(out$Pt) <-
    colnames(out$Ptt) <- rownames(out$Pt) <- rownames(out$Ptt) <- model$state_names
  
  out$at <- ts(out$at, start = start(model$y), frequency = frequency(model$y))
  out$att <- ts(out$att, start = start(model$y), frequency = frequency(model$y))
  out
}
