#' Kalman Smoothing
#'
#' Methods for Kalman smoothing of the states. Function \code{fast_smoother}
#' computes only smoothed estimates of the states, and function
#' \code{smoother} computes also smoothed variances.
#' 
#' For non-Gaussian models, the smoothing is based on the approximate Gaussian model.
#'
#' @param model Model model.
#' @param ... Ignored.
#' @return Matrix containing the smoothed estimates of states, or a list
#' with the smoothed states and the variances.
#' @export
#' @rdname smoother
fast_smoother <- function(model, ...) {
  UseMethod("fast_smoother", model)
}
#' @method fast_smoother gaussian
#' @export
fast_smoother.gaussian <- function(model, ...) {
  
  out <- gaussian_fast_smoother(model, model_type(model))
  colnames(out) <- names(model$a1)
  ts(out[-nrow(out), , drop = FALSE], start = start(model$y), 
    frequency = frequency(model$y))
}

#' @method fast_smoother nongaussian
#' @export
fast_smoother.nongaussian <- function(model, ...) {
  fast_smoother(gaussian_approx(model))
}
#' @export
#' @rdname smoother
smoother <- function(model, ...) {
  UseMethod("smoother", model)
}
#' @method smoother gaussian
#' @export
smoother.gaussian <- function(model, ...) {
  
  out <-  gaussian_smoother(model, model_type(model))
  colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <- names(model$a1)
  
  out$Vt <- out$Vt[, , -nrow(out$alphahat), drop = FALSE]
  out$alphahat <- ts(out$alphahat[-nrow(out$alphahat), , drop = FALSE], 
    start = start(model$y), frequency = frequency(model$y))
  out
}

#' @method smoother nongaussian
#' @export
smoother.nongaussian <- function(model, ...) {
  smoother(gaussian_approx(model))
}


#' Extended Kalman Smoothing
#'
#' Function \code{ekf_smoother} runs the (iterated) extended Kalman smoother for 
#' the given non-linear Gaussian model of class \code{ssm_nlg}, 
#' and returns the smoothed estimates of the states and the corresponding variances.
#'
#' @param model Model model
#' @param iekf_iter If \code{iekf_iter > 0}, iterated extended Kalman filter is 
#' used with \code{iekf_iter} iterations.
#' @return List containing the log-likelihood,
#' smoothed state estimates \code{alphahat}, and the corresponding variances \code{Vt} and
#'  \code{Ptt}.
#' @export
#' @rdname ekf_smoother
#' @export
ekf_smoother <- function(model, iekf_iter = 0) {
  
  out <- ekf_smoother_nlg(t(model$y), model$Z, model$H, model$T, 
    model$R, model$Z_gn, model$T_gn, model$a1, model$P1, 
    model$theta, model$log_prior_pdf, model$known_params, 
    model$known_tv_params, model$n_states, model$n_etas, 
    as.integer(model$time_varying), iekf_iter)
  colnames(out$alphahat) <- colnames(out$Vt) <-
    rownames(out$Vt) <- model$state_names
  
  out$Vt <- out$Vt[, , -nrow(out$alphahat), drop = FALSE]
  out$alphahat <- ts(out$alphahat[-nrow(out$alphahat), , drop = FALSE], 
    start = start(model$y), frequency = frequency(model$y))
  out
}

ekf_fast_smoother <- function(model, iekf_iter = 0) {
  
  out <- ekf_fast_smoother_nlg(t(model$y), model$Z, model$H, model$T, 
    model$R, model$Z_gn, model$T_gn, model$a1, model$P1, 
    model$theta, model$log_prior_pdf, model$known_params, 
    model$known_tv_params, model$n_states, model$n_etas, 
    as.integer(model$time_varying), iekf_iter)
  colnames(out$alphahat) <- colnames(out$Vt) <-
    rownames(out$Vt) <- model$state_names
  ts(out[-nrow(out$alphahat), , drop = FALSE], start = start(model$y), 
    frequency = frequency(model$y))
}

