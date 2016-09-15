#' Kalman Filtering
#'
#' Function \code{kfilter} runs the Kalman filter for the given model
#' (and it's parameters, including regression coefficients), and returns the filtered estimates and
#' one-step-ahead predictions of the states \eqn{\alpha_t} given the
#' model parameters and data up to time \eqn{t}.
#'
#' For non-Gaussian models, the Kalman filtering is based on the approximate Gaussian model.
#'
#' @param object Model object
#' @param ... Ignored.
#' @return List containing the log-likelihood (approximate in non-Gaussian case),
#' one-step-ahead predictions \code{at} and filtered
#' estimates \code{att} of states, and the corresponding variances \code{Pt} and
#'  \code{Ptt}.
#' @seealso \code{\link{particle_filter}}
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
#' For non-Gaussian models, the smoothing is based on the approximate Gaussian model.
#'
#' @param object Model object.
#' @param nsim Number of independent samples.
#' @param seed Seed for the random number generator.
#' @param ... Ignored.
#' @return Matrix containing the smoothed estimates of states, list
#' with the smoothed states and the variances, or an array containing the
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
#' model <- bsm(rep(NA, 50), sd_level = uniform(1,0,5), sd_y = uniform(1,0,5), slope = FALSE)
#' sim <- sim_smoother(model, 12)
#' ts.plot(sim[, 1, ])
sim_smoother <- function(object, nsim, seed, ...) {
  UseMethod("sim_smoother", object)
}

#' Bayesian Inference of State Space Models using MCMC with RAM
#'
#' Adaptive Markov chain Monte Carlo simulation of state space models using
#' Robust Adaptive Metropolis algorithm by Vihola (2012).
#'
#' @param object Object of class \code{gssm}, \code{bsm}, \code{ngssm}, \code{svm},
#' or \code{ng_bsm}.
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

#' Importance Sampling from non-Gaussian State Space Model
#' 
#' Returns \code{nsim} samples from the approximating Gaussian model with corresponding 
#' (scaled) importance weights.
#' @param object of class \code{ng_bsm}, \code{svm} or \code{ngssm}.
#' @param nsim Number of samples.
#' @param seed Seed for the random number generator.
#' @param ... Ignored.
#' @export
#' @rdname importance_sample
importance_sample <- function(object, nsim, seed, ...) {
  UseMethod("importance_sample", object)
}


#' Gaussian approximation of non-Gaussian state space model
#' 
#' Returns the approximating Gaussian model.
#' @param object model object.
#' @param max_iter Maximum number of iterations.
#' @param conv_tol Tolerance parameter. Document properly later!
#' @param ... Ignored.
#' @export
#' @rdname gaussian_approx
gaussian_approx <- function(object, max_iter, conv_tol, ...) {
  UseMethod("gaussian_approx", object)
}

#' Particle Filtering
#' 
#' Function \code{particle_filter} performs a bootstrap filtering with stratification 
#' resampling.
#' 
#' @param object of class \code{bsm}, \code{ng_bsm} or \code{svm}.
#' @param nsim Number of samples.
#' @param seed Seed for RNG.
#' @param ... Ignored.
#' @return A list containing samples, weights from the last time point, and an 
#' estimate of log-likelihood.
#' @export
#' @rdname particle_filter
particle_filter <- function(object, nsim, seed, ...) {
  UseMethod("particle_filter", object)
}


#' Particle Smoothing
#' 
#' Function \code{particle_smoother} performs filter-smoother or forward-backward smoother, 
#' using a bootstrap filtering with stratification resampling.
#' 
#' @param object of class \code{bsm}, \code{ng_bsm} or \code{svm}.
#' @param nsim Number of samples.
#' @param method Either \code{"fs"} (filter-smoother), or \code{"fbs"} (forward-backward smoother).
#' @param seed Seed for RNG.
#' @param ... Ignored.
#' @export
#' @rdname particle_smoother
particle_smoother <- function(object, nsim, method, seed, ...) {
  UseMethod("particle_smoother", object)
}