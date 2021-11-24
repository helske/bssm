#' Gaussian Approximation of Non-Gaussian/Non-linear State Space Model
#'
#' Returns the approximating Gaussian model which has the same conditional 
#' mode of p(alpha|y, theta) as the original model. 
#' This function is rarely needed itself, and is mainly available for 
#' testing and debugging purposes.
#' 
#' @param model Model to be approximated. Should be of class 
#' \code{bsm_ng}, \code{ar1_ng} \code{svm}, 
#' \code{ssm_ung}, or \code{ssm_mng}, or \code{ssm_nlg}, i.e. non-gaussian or 
#' non-linear \code{bssm_model}.
#' @param max_iter Maximum number of iterations as a positive integer. 
#' Default is 100 (although typically only few iterations are needed).
#' @param conv_tol Positive tolerance parameter. Default is 1e-8. Approximation 
#' is claimed to be converged when the mean squared difference of the modes of 
#' is less than \code{conv_tol}.
#' @param iekf_iter For non-linear models, non-negative number of iterations in 
#' iterated EKF (defaults to 0, i.e. normal EKF). Used only for models of class 
#' \code{ssm_nlg}.
#' @param ... Ignored.
#' @return Returns linear-Gaussian SSM of class \code{ssm_ulg} or 
#' \code{ssm_mlg} which has the same conditional mode of p(alpha|y, theta) as 
#'   the original model.
#' @references 
#' Koopman, SJ and Durbin J (2012). Time Series Analysis by State Space 
#' Methods. Second edition. Oxford: Oxford University Press.
#' 
#' Vihola, M, Helske, J, Franks, J. (2020). Importance sampling type estimators 
#' based on approximate marginal Markov chain Monte Carlo. 
#' Scand J Statist. 1-38. https://doi.org/10.1111/sjos.12492
#' @export
#' @rdname gaussian_approx
#' @examples 
#' data("poisson_series")
#' model <- bsm_ng(y = poisson_series, sd_slope = 0.01, sd_level = 0.1,
#'   distribution = "poisson")
#' out <- gaussian_approx(model)
#' for(i in 1:7)
#'  cat("Number of iterations used: ", i, ", y[1] = ",
#'    gaussian_approx(model, max_iter = i, conv_tol = 0)$y[1], "\n", sep ="")
#'    
gaussian_approx <- function(model, max_iter, conv_tol, ...) {
  UseMethod("gaussian_approx", model)
}
#' @rdname gaussian_approx
#' @method gaussian_approx nongaussian
#' @export
gaussian_approx.nongaussian <- function(model, max_iter = 100, 
  conv_tol = 1e-8, ...) {
  
  check_missingness(model)
  
  model$max_iter <- check_intmax(max_iter, "max_iter", positive = FALSE)
  model$conv_tol <- check_positive_real(conv_tol, "conv_tol")
  
  model$distribution <- pmatch(model$distribution,
    c("svm", "poisson", "binomial", "negative binomial", "gamma", "gaussian"), 
    duplicates.ok = TRUE) - 1
  out <- gaussian_approx_model(model, model_type(model))
  
  if (ncol(out$y) == 1L) {
    out$y <- ts(c(out$y), start = start(model$y), end = end(model$y), 
      frequency = frequency(model$y))
    D <- model$D
    if (length(model$beta) > 0) 
      D <- as.numeric(D) + t(model$xreg %*% model$beta)
    approx_model <- ssm_ulg(y = out$y, Z = model$Z, H = out$H, T = model$T, 
      R = model$R, a1 = model$a1, P1 = model$P1, init_theta = model$theta,
      D = D, C = model$C, state_names = names(model$a1), 
      update_fn = model$update_fn, prior_fn = model$prior_fn)
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
  
  check_missingness(model)
  
  model$max_iter <- check_intmax(max_iter, "max_iter", positive = FALSE)
  model$conv_tol <- check_positive_real(conv_tol, "conv_tol")
  model$iekf_iter <- check_intmax(iekf_iter, "iekf_iter", positive = FALSE)
  
  out <- gaussian_approx_model_nlg(t(model$y), model$Z, model$H, model$T, 
    model$R, model$Z_gn, model$T_gn, model$a1, model$P1, 
    model$theta, model$log_prior_pdf, model$known_params, 
    model$known_tv_params, model$n_states, model$n_etas,
    as.integer(model$time_varying),
    max_iter, conv_tol, iekf_iter)
  
  out$y <- ts(t(out$y), start = start(model$y), end = end(model$y), 
    frequency = frequency(model$y))
  ssm_mlg(y = out$y, Z = out$Z, H = out$H, T = out$T, 
    R = out$R, a1 = c(out$a1), P1 = out$P1,
    init_theta = model$theta, D = out$D, C = out$C)
}
