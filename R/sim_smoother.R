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
#' @param use_antithetic Use an antithetic variable for location. Default is \code{FALSE}. Ignored for multivariate models.
#' @param seed Seed for the random number generator.
#' @param ... Ignored.
#' @return An array containing the generated samples.
#' @export
#' @rdname sim_smoother
#' @examples
#' model <- bsm_lg(rep(NA, 50), sd_level = uniform(1,0,5), sd_y = uniform(1,0,5))
#' sim <- sim_smoother(model, 12)
#' ts.plot(sim[, 1, ])
sim_smoother <- function(object, nsim, use_antithetic = FALSE, seed, ...) {
  UseMethod("sim_smoother", object)
}
#' @method sim_smoother gaussian
#' @rdname sim_smoother
#' @export
sim_smoother.gaussian <- function(object, nsim = 1, use_antithetic = FALSE,
  seed = sample(.Machine$integer.max, size = 1), ...) {

  out <- gaussian_sim_smoother(object, nsim, use_antithetic, seed, model_type(model))
  rownames(out) <- names(object$a1)
  aperm(out, c(2, 1, 3))[-(length(object$y) + 1), , , drop = FALSE]
}
#' @method sim_smoother nongaussian
#' @rdname sim_smoother
#' @export
sim_smoother.nongaussian <- function(object, nsim = 1,
  seed = sample(.Machine$integer.max, size = 1), use_antithetic = FALSE, ...) {
  sim_smoother(gaussian_approx(object), nsim = nsim, 
    use_antithetic = use_antithetic, seed = seed)
}

