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
#' Kalman filtering of linear Gaussian models.
#'
#' \code{kfilter} runs the Kalman filter for the given model
#' (and it's parameters), and returns the filtered estimates  and
#' one-step-ahead predictions of the states \eqn{\alpha_t} given the
#' model parameters and data up to time \eqn{t}.
#' @param object Object of class \code{gssm} or \code{bstsm}
#' @param ... Ignored.
#' @return List containing theone-step-ahead predictions \code{at} and filtered
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
#' @param nsim Number of samples.
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
sim_smoother <- function(object, nsim, ...) {
  UseMethod("sim_smoother", object)
}

#' Bayesian Inference of State Space Models using MCMC with RAM
#'
#' Adaptive Markov chain Monte Carlo simulation of state space models using
#' Robust Adaptive Metropolis algorithm by Vihola (2012).
#' Function
#'
#' @param object Object of class \code{gssm} or \code{bstsm}.
#' @param n_iter Number of MCMC iterations.
#' @param type Type of output. Default is \code{"full"}, which returns
#' samples from the posterior \eqn{p(\alpha, \theta}. Option
#' \code{"parameters"} samples only parameters \eqn{\theta} (which includes the
#' regression coefficients \eqn{\beta}). This can be used for faster inference of
#' \eqn{\theta} only, or as an preliminary run for obtaining
#' initial values for \code{S}. Option \code{"summary"} does not simulate
#' states directly computes the  posterior means and variances of states using
#' fast Kalman smoothing. This is slightly faster, memory  efficient and
#' more accurate than calculations based on simulation smoother.

#' @param lower_prior,upper_prior Bounds of the uniform prior for parameters
#' \eqn{\theta}. Optional for \code{bstsm} objects.
#' @param nsim_states Number of simulations of states per MCMC iteration. Only
#' used when \code{type = "full"}.
#' @param n_burnin Length of the burn-in period which is disregarded from the
#' results. Defaults to \code{n_iter / 10}.
#' @param n_thin Thinning rate. Defaults to 1. Increase for long time series in
#' order to save memory.
#' @param gamma Tuning parameter for the adaptation of RAM algorithm. Must be
#' between 0 and 1 (not checked).
#' @param target_acceptance Target acceptance ratio for RAM.
#' @param S Initial value for the lower triangular matrix of RAM
#' algorithm, so that the covariance matrix of the Gaussian proposal
#' distribution is \eqn{SS'}.
#' @param ... Ignored.
#' @export
#' @rdname run_mcmc
#' @references Vihola, Matti (2012). "Robust adaptive Metropolis algorithm with
#' coerced acceptance rate". Statistics and Computing, Volume 22, Issue 5,
#' pages 997--1008.
run_mcmc <- function(object, n_iter, type, lower_prior, upper_prior, nsim_states,
  n_burnin, n_thin, gamma, target_acceptance, S, ...) {
  UseMethod("run_mcmc", object)
}

