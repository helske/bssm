#' Coerce to gssm Object
#'
#' Converts Gaussian \code{SSModel} object of \code{KFAS} package to \code{gssm}
#' object.
#'
#' @param model Object of class \code{SSModel}.
#' @param kappa For \code{SSModel} object, a prior variance for initial state u
#' sed to replace exact diffuse elements of the original model.
#' @param ... Ignored.
#' @return Object of class \code{gssm}.
#' @export
#' @rdname as_gssm
as_gssm <- function(model, ...) {
  UseMethod("as_gssm", model)
}

#' Kalman Filtering
#'
#' Function \code{kfilter} runs the Kalman filter for the given model
#' (and it's parameters), and returns the filtered estimates  and
#' one-step-ahead predictions of the states \eqn{\alpha_t} given the
#' model parameters and data up to time \eqn{t}.
#'
#' For non-Gaussian models, the filtering is based on the approximate Gaussian model.
#'
#' @param object Model object
#' @param ... Ignored.
#' @return List containing the log-likelihood (approximate in non-Gaussian ce),
#' one-step-ahead predictions \code{at} and filtered
#' estimates \code{att} of states, and the corresponding variances \code{Pt} and
#'  \code{Ptt}.
#' @export
#' @rdname kfilter
kfilter <- function(object, ...) {
  UseMethod("kfilter", object)
}

#' Kalman Smoothing
#'
#' Methods for Kalman smoothing of the states. Function \code{fast_smoother}
#' computes only smoothed estimates of the states, and function
#' \code{smoother} computes also smoothed variances. Function
#' \code{sim_smoother} performs simulation smoothing i.e. simulates the states
#' from the conditional distribution \eqn{p(\alpha | y, \theta)}.
#'
#' @param object Object of class \code{gssm} or \code{bstsm}
#' @param nsim Number of samples. Simulation smoother uses one antithetic
#' variable, ideally making the first and second halves of the resulting array to be
#' negatively correlated (see the example).
#' @param seed Seed for Boost random number generator.
#' @param ... Ignored.
#' @return Matrix containing the smoothed estimates of states, list
#' with the smoothed states and the variances, or an array  containing the
#' generated samples.
#' @export
#' @rdname smoother
fast_smoother <- function(object, ...) {
  UseMethod("fast_smoother", object)
}

#' @export
#' @rdname smoother
smoother <- function(object, ...) {
  UseMethod("smoother", object)
}

#' @export
#' @rdname smoother
#' @examples
#' # need to give upper_prior as we have only NA's in y...
#' model <- bsm(rep(NA, 30), upper_prior = c(10, 10), sd_level = 1, sd_y = 1, slope = FALSE)
#' sim <- sim_smoother(model, 4)
#' ts.plot(sim[, 1, ])
#' cor(sim[, 1, ])
sim_smoother <- function(object, nsim, seed, ...) {
  UseMethod("sim_smoother", object)
}

#' Bayesian Inference of State Space Models using MCMC with RAM
#'
#' Adaptive Markov chain Monte Carlo simulation of state space models using
#' Robust Adaptive Metropolis algorithm by Vihola (2012).
#'
#' @param object Object of class \code{gssm}, \code{bstsm}, \code{ngssm}, or \code{ng_bsm}.
#' @param ... Arguments to be passed to methods.
#' See \code{\link{run_mcmc.gssm}} and \code{\link{run_mcmc.ngssm}} for details.
#' @export
#' @rdname run_mcmc
#' @references Vihola, Matti (2012). "Robust adaptive Metropolis algorithm with
#' coerced acceptance rate". Statistics and Computing, Volume 22, Issue 5,
#' pages 997--1008.
run_mcmc <- function(object, ...) {
  UseMethod("run_mcmc", object)
}

