#' Kalman Smoothing
#'
#' Methods for Kalman smoothing of the states. Function \code{fast_smoother}
#' computes only smoothed estimates of the states, and function
#' \code{smoother} computes also smoothed variances.
#' 
#' For non-Gaussian models, the smoothing is based on the approximate Gaussian 
#' model.
#'
#' @inheritParams gaussian_approx
#' @param ... Ignored.
#' @return Matrix containing the smoothed estimates of states, or a list
#' with the smoothed states and the variances.
#' @export
#' @rdname smoother
fast_smoother <- function(model, ...) {
  UseMethod("fast_smoother", model)
}
#' @method fast_smoother lineargaussian
#' @rdname smoother
#' @export
#' @examples
#' model <- bsm_lg(Nile, 
#'   sd_level = tnormal(120, 100, 20, min = 0),
#'   sd_y = tnormal(50, 50, 25, min = 0),
#'   a1 = 1000, P1 = 200)
#' ts.plot(cbind(Nile, fast_smoother(model)), col = 1:2)
fast_smoother.lineargaussian <- function(model, ...) {
  
  check_missingness(model)
  
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
#' @method smoother lineargaussian
#' @rdname smoother
#' @export
#' @examples
#' model <- bsm_lg(Nile, 
#'   sd_y = tnormal(120, 100, 20, min = 0),
#'   sd_level = tnormal(50, 50, 25, min = 0),
#'   a1 = 1000, P1 = 500^2)
#' 
#' out <- smoother(model)
#' ts.plot(cbind(Nile, out$alphahat), col = 1:2)
#' ts.plot(sqrt(out$Vt[1, 1, ]))
smoother.lineargaussian <- function(model, ...) {
  
  check_missingness(model)
  
  out <-  gaussian_smoother(model, model_type(model))
  colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <- 
    names(model$a1)
  
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
#' Function \code{ekf_smoother} runs the (iterated) extended Kalman smoother 
#' for the given non-linear Gaussian model of class \code{ssm_nlg}, 
#' and returns the smoothed estimates of the states and the corresponding 
#' variances. Function \code{ekf_fast_smoother} computes only smoothed 
#' estimates of the states.
#'
#' @inheritParams ekf
#' @return List containing the log-likelihood,
#' smoothed state estimates \code{alphahat}, and the corresponding variances 
#' \code{Vt} and \code{Ptt}.
#' @export
#' @rdname ekf_smoother
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
#' y <- rnorm(length(x), exp(x), sigma_y)
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
#' out_ekf <- ekf_smoother(model_nlg, iekf_iter = 0)
#' out_iekf <- ekf_smoother(model_nlg, iekf_iter = 1)
#' ts.plot(cbind(x, out_ekf$alphahat, out_iekf$alphahat), col = 1:3)
#' }
ekf_smoother <- function(model, iekf_iter = 0) {
  
  check_missingness(model)
  
  iekf_iter <- check_intmax(iekf_iter, "iekf_iter", positive = FALSE)
  
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
#' @rdname ekf_smoother
#' @export
ekf_fast_smoother <- function(model, iekf_iter = 0) {
  
  check_missingness(model)
  
  iekf_iter <- check_intmax(iekf_iter, "iekf_iter", positive = FALSE)
  
  out <- ekf_fast_smoother_nlg(t(model$y), model$Z, model$H, model$T, 
    model$R, model$Z_gn, model$T_gn, model$a1, model$P1, 
    model$theta, model$log_prior_pdf, model$known_params, 
    model$known_tv_params, model$n_states, model$n_etas, 
    as.integer(model$time_varying), iekf_iter)
  colnames(out$alphahat) <- model$state_names
  ts(out$alphahat[-nrow(out$alphahat),, drop = FALSE], start = start(model$y), 
    frequency = frequency(model$y))
}
