#' Simulation Smoothing
#'
#' Function \code{sim_smoother} performs simulation smoothing i.e. simulates 
#' the states from the conditional distribution \eqn{p(\alpha | y, \theta)} 
#' for linear-Gaussian models.
#' 
#' For non-Gaussian/non-linear models, the simulation is based on the 
#' approximating Gaussian model.
#'
#' @param model Model object.
#' @param nsim Number of independent samples.
#' @param use_antithetic Use an antithetic variable for location. 
#' Default is \code{FALSE}. Ignored for multivariate models.
#' @param seed Seed for the random number generator.
#' @param ... Ignored.
#' @return An array containing the generated samples.
#' @export
#' @rdname sim_smoother
#' @examples
#' model <- bsm_lg(rep(NA, 50), sd_level = uniform(1,0,5), 
#'   sd_y = uniform(1,0,5))
#' sim <- sim_smoother(model, 12)
#' ts.plot(sim[, 1, ])
sim_smoother <- function(model, nsim, seed, use_antithetic = FALSE, ...) {
  UseMethod("sim_smoother", model)
}
#' @method sim_smoother gaussian
#' @rdname sim_smoother
#' @export
sim_smoother.gaussian <- function(model, nsim = 1, 
  seed = sample(.Machine$integer.max, size = 1), use_antithetic = FALSE, ...) {

  nsim <- check_integer(nsim, "nsim")  
  if (!test_flag(use_antithetic)) 
    stop("Argument 'use_antithetic' should be TRUE or FALSE. ")
  
  out <- gaussian_sim_smoother(model, nsim, use_antithetic, seed, 
    model_type(model))
  rownames(out) <- names(model$a1)
  aperm(out, c(2, 1, 3))[-(length(model$y) + 1), , , drop = FALSE]
}
#' @method sim_smoother nongaussian
#' @rdname sim_smoother
#' @export
sim_smoother.nongaussian <- function(model, nsim = 1,
  seed = sample(.Machine$integer.max, size = 1), use_antithetic = FALSE, ...) {
  
  nsim <- check_integer(nsim, "nsim")
  if (!test_flag(use_antithetic)) 
    stop("Argument 'use_antithetic' should be TRUE or FALSE. ")
  
  sim_smoother(gaussian_approx(model), nsim = nsim, 
    use_antithetic = use_antithetic, seed = seed)
}

