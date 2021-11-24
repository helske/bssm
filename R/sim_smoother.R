#' Simulation Smoothing
#'
#' Function \code{sim_smoother} performs simulation smoothing i.e. simulates 
#' the states from the conditional distribution \eqn{p(\alpha | y, \theta)} 
#' for linear-Gaussian models.
#' 
#' For non-Gaussian/non-linear models, the simulation is based on the 
#' approximating Gaussian model.
#'
#' @inheritParams importance_sample
#' @param model Model of class \code{bsm_lg}, \code{ar1_lg}
#' \code{ssm_ulg}, or \code{ssm_mlg}, or one of the non-gaussian models
#' \code{bsm_ng}, \code{ar1_ng} \code{svm}, 
#' \code{ssm_ung}, or \code{ssm_mng}.
#' @return An array containing the generated samples.
#' @export
#' @rdname sim_smoother
#' @examples
#' # only missing data, simulates from prior
#' model <- bsm_lg(rep(NA, 25), sd_level = 1, 
#'   sd_y = 1)
#' # use antithetic variable for location
#' sim <- sim_smoother(model, nsim = 4, use_antithetic = TRUE, seed = 1)
#' ts.plot(sim[, 1, ])
#' cor(sim[, 1, ])
sim_smoother <- function(model, nsim, seed, use_antithetic = TRUE, ...) {
  UseMethod("sim_smoother", model)
}
#' @method sim_smoother lineargaussian
#' @rdname sim_smoother
#' @export
sim_smoother.lineargaussian <- function(model, nsim = 1, 
  seed = sample(.Machine$integer.max, size = 1), use_antithetic = TRUE, ...) {
  
  check_missingness(model)
 
  nsim <- check_intmax(nsim, "nsim")  
  seed <- check_intmax(seed, "seed", FALSE, max = .Machine$integer.max) 
  if (!test_flag(use_antithetic)) 
    stop("Argument 'use_antithetic' should be TRUE or FALSE. ")
  nsamples <- ifelse(!is.null(nrow(model$y)), nrow(model$y), length(model$y)) * 
    length(model$a1) * nsim
  if (nsim > 100 & nsamples > 1e10) {
    warning(paste("Trying to sample ", nsamples, 
      "particles, you might run out of memory."))
  }
  out <- gaussian_sim_smoother(model, nsim, use_antithetic, seed, 
    model_type(model))
  rownames(out) <- names(model$a1)
  aperm(out, c(2, 1, 3))[-(length(model$y) + 1), , , drop = FALSE]
}
#' @method sim_smoother nongaussian
#' @rdname sim_smoother
#' @export
sim_smoother.nongaussian <- function(model, nsim = 1,
  seed = sample(.Machine$integer.max, size = 1), use_antithetic = TRUE, ...) {
  
  sim_smoother(gaussian_approx(model), nsim = nsim, 
    use_antithetic = use_antithetic, seed = seed)
}

