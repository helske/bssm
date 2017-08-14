#' Simulation Smoothing
#'
#' Function \code{sim_smoother} performs simulation smoothing i.e. simulates the states
#' from the conditional distribution \eqn{p(\alpha | y, \theta)}.
#' 
#' For non-Gaussian/non-linear models, the simulation is based on the approximating
#' Gaussian model.
#'
#' @param object Model object.
#' @param nsim Number of independent samples.
#' @param use_antithetic Use an antithetic variable for location. Default is \code{FALSE}.
#' @param seed Seed for the random number generator.
#' @param ... Ignored.
#' @return An array containing the generated samples.
#' @export
#' @rdname sim_smoother
#' @examples
#' model <- bsm(rep(NA, 50), sd_level = uniform(1,0,5), sd_y = uniform(1,0,5))
#' sim <- sim_smoother(model, 12)
#' ts.plot(sim[, 1, ])
sim_smoother <- function(object, nsim, seed, use_antithetic = FALSE, ...) {
  UseMethod("sim_smoother", object)
}
#' @method sim_smoother gssm
#' @export
sim_smoother.gssm <- function(object, nsim = 1, 
  seed = sample(.Machine$integer.max, size = 1), use_antithetic = FALSE, ...) {

  out <- gaussian_sim_smoother(object, nsim, use_antithetic, seed, model_type = 1L)
  rownames(out) <- names(object$a1)
  aperm(out, c(2, 1, 3))
}
#' @method sim_smoother bsm
#' @export
sim_smoother.bsm <- function(object, nsim = 1, 
  seed = sample(.Machine$integer.max, size = 1), use_antithetic = FALSE, ...) {

  out <- gaussian_sim_smoother(object, nsim, use_antithetic, seed, model_type = 2L)
  rownames(out) <- names(object$a1)
  aperm(out, c(2, 1, 3))
}
#' @method sim_smoother lgg_ssm
#' @export
sim_smoother.lgg_ssm <- function(object, nsim = 1, 
  seed = sample(.Machine$integer.max, size = 1), use_antithetic = FALSE, ...) {
  
  out <- general_gaussian_sim_smoother(t(object$y), object$Z, object$H, object$T, 
    object$R, object$a1, object$P1, 
    object$theta, object$obs_intercept, object$state_intercept,
    object$log_prior_pdf, object$known_params, 
    object$known_tv_params,
    object$n_states, object$n_etas, nsim, use_antithetic, seed)
  rownames(out) <- names(object$a1)
  aperm(out, c(2, 1, 3))
}

#' @method sim_smoother ngssm
#' @export
sim_smoother.ngssm <- function(object, nsim = 1,
  seed = sample(.Machine$integer.max, size = 1), use_antithetic = FALSE, ...) {
  sim_smoother(gaussian_approx(object), nsim = nsim, 
    use_antithetic = use_antithetic, seed = seed)
}
#' @method sim_smoother ng_bsm
#' @export
sim_smoother.ng_bsm <- function(object, nsim = 1,
  seed = sample(.Machine$integer.max, size = 1), use_antithetic = FALSE, ...) {
  sim_smoother(gaussian_approx(object), nsim = nsim, 
    use_antithetic = use_antithetic, seed = seed)
}
#' @method sim_smoother svm
#' @export
sim_smoother.svm <- function(object, nsim = 1, 
  seed = sample(.Machine$integer.max, size = 1), use_antithetic = FALSE, ...) {
  sim_smoother(gaussian_approx(object), nsim = nsim, 
    use_antithetic = use_antithetic, seed = seed)
}