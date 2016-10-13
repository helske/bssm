#' Simulation Smoothing
#'
#' Function \code{sim_smoother} performs simulation smoothing i.e. simulates the states
#' from the conditional distribution \eqn{p(\alpha | y, \theta)}.
#'
#' For non-Gaussian models, the simulation smoothing is based on the approximate Gaussian model.
#'
#' @param object Model object.
#' @param nsim Number of independent samples.
#' @param seed Seed for the random number generator.
#' @param ... Ignored.
#' @return An array containing the generated samples.
#' @export
#' @rdname sim_smoother
#' @examples
#' model <- bsm(rep(NA, 50), sd_level = uniform(1,0,5), sd_y = uniform(1,0,5))
#' sim <- sim_smoother(model, 12)
#' ts.plot(sim[, 1, ])
sim_smoother <- function(object, nsim, seed, ...) {
  UseMethod("sim_smoother", object)
}
#' @method sim_smoother gssm
#' @export
sim_smoother.gssm <- function(object, nsim = 1, seed = sample(.Machine$integer.max, size = 1), ...) {

  out <- gssm_sim_smoother(object, nsim, seed)
  rownames(out) <- names(object$a1)
  aperm(out, c(2, 1, 3))
}
#' @method sim_smoother bsm
#' @export
sim_smoother.bsm <- function(object, nsim = 1, seed = sample(.Machine$integer.max, size = 1), ...) {

  out <- bsm_sim_smoother(object, nsim, seed)
  rownames(out) <- names(object$a1)
  aperm(out, c(2, 1, 3))
}
#' @method sim_smoother ngssm
#' @export
sim_smoother.ngssm <- function(object, nsim = 1, seed = sample(.Machine$integer.max, size = 1), ...) {

  object$distribution <- pmatch(object$distribution,
    c("poisson", "binomial", "negative binomial"))

  out <- ngssm_sim_smoother(object, nsim, object$init_signal, seed)
  rownames(out) <- names(object$a1)
  aperm(out, c(2, 1, 3))
}
#' @method sim_smoother ng_bsm
#' @export
sim_smoother.ng_bsm <- function(object, nsim = 1, seed = sample(.Machine$integer.max, size = 1), ...) {

  object$distribution <- pmatch(object$distribution, c("poisson", "binomial", "negative binomial"))
  out <- ng_bsm_sim_smoother(object, nsim, object$init_signal, seed)

  rownames(out) <- names(object$a1)
  aperm(out, c(2, 1, 3))
}
#' @method sim_smoother svm
#' @export
sim_smoother.svm <- function(object, nsim = 1, seed = sample(.Machine$integer.max, size = 1), ...) {

  out <- svm_sim_smoother(object, nsim, object$init_signal, seed)

  rownames(out) <- names(object$a1)
  aperm(out, c(2, 1, 3))
}