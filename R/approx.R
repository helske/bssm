#' Gaussian Approximation of Non-Gaussian/Non-linear State Space Model
#'
#' Returns the approximating Gaussian model. This function is rarely needed itself, 
#' and is mainly available for testing and debugging purposes.
#' 
#' @param model Model to be approximated.
#' @param max_iter Maximum number of iterations.
#' @param conv_tol Tolerance parameter.
#' @param iekf_iter For non-linear models, number of iterations in iterated EKF (defaults to 0).
#' @param ... Ignored.
#' @export
#' @rdname gaussian_approx
#' @examples 
#' data("poisson_series")
#' model <- bsm_ng(y = poisson_series, sd_slope = 0.01, sd_level = 0.1,
#'   distribution = "poisson")
#' out <- gaussian_approx(model)
gaussian_approx <- function(model, max_iter, conv_tol, ...) {
  UseMethod("gaussian_approx", model)
}
#' @rdname gaussian_approx
#' @method gaussian_approx nongaussian
#' @export
gaussian_approx.nongaussian <- function(model, max_iter = 100, conv_tol = 1e-8, ...) {
  
  model$max_iter <- max_iter
  model$conv_tol <- conv_tol
  model$distribution <- pmatch(model$distribution,
    c("svm", "poisson", "binomial", "negative binomial", "gamma", "gaussian"), 
    duplicates.ok = TRUE) - 1
  out <- gaussian_approx_model(model, model_type(model))
  
  if(ncol(out$y) == 1) {
    out$y <- ts(c(out$y), start = start(model$y), end = end(model$y), 
      frequency = frequency(model$y))
    D <- model$D
    if(length(model$beta) > 0) D <- as.numeric(D) + t(model$xreg %*% model$beta)
    approx_model <- ssm_ulg(y = out$y, Z = model$Z, H = out$H, T = model$T, 
      R = model$R, a1 = model$a1, P1 = model$P1, init_theta = model$theta,
      D = D, C = model$C, state_names = names(model$a1), update_fn = model$update_fn,
      prior_fn = model$prior_fn)
  } else {
    out$y <- ts(t(out$y), start = start(model$y), end = end(model$y), 
      frequency = frequency(model$y))
    approx_model <- ssm_mlg(y = out$y, Z = model$Z, H = out$H, T = model$T, 
      R = model$R, a1 = model$a1, P1 = model$P1, init_theta = model$theta,
      D = model$D, C = model$C, state_names = names(model$a1), 
      update_fn = model$update_fn, prior_fn = model$prior_fn)
  }
  approx_model
}
#' @rdname gaussian_approx
#' @method gaussian_approx ssm_nlg
#' @export
gaussian_approx.ssm_nlg <- function(model, max_iter = 100, 
  conv_tol = 1e-8, iekf_iter = 0, ...) {
  
  model$max_iter <- max_iter
  model$conv_tol <- conv_tol
  model$iekf_iter <- iekf_iter
  
  out <- gaussian_approx_model_nlg(t(model$y), model$Z, model$H, model$T, 
    model$R, model$Z_gn, model$T_gn, model$a1, model$P1, 
    model$theta, model$log_prior_pdf, model$known_params, 
    model$known_tv_params, model$n_states, model$n_etas,
    as.integer(model$time_varying),
    max_iter, conv_tol, iekf_iter, default_update_fn, default_prior_fn)
  
  out$y <- ts(t(out$y), start = start(model$y), end = end(model$y), frequency = frequency(model$y))
  ssm_mlg(y = out$y, Z = out$Z, H = out$H, T = out$T, 
    R = out$R, a1 = c(out$a1), P1 = out$P1,
    init_theta = model$theta, D = out$D, C = out$C)
}
