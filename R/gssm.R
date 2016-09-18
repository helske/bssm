#'
#' Construct an object of class \code{gssm} by defining the corresponding terms
#' of the observation and state equation:
#'
#' \deqn{y_t = Z_t \alpha_t + H_t \epsilon_t, (\textrm{observation equation})}
#' \deqn{\alpha_{t+1} = T_t \alpha_t + R_t \eta_t, (\textrm{transition equation})}
#'
#' where \eqn{\epsilon_t \sim N(0, 1)}, \eqn{\eta_t \sim N(0, I_k)} and
#' \eqn{\alpha_1 \sim N(a_1, P_1)} independently of each other.
#'
#' @param y Observations as time series (or vector) of length \eqn{n}.
#' @param Z System matrix Z of the observation equation. Either a vector of
#' length m or a m x n array, or an object which can be coerced to such.
#' @param H Vector of standard deviations. Either a scalar or a vector of length
#' n, or an object which can be coerced to such.
#' @param T System matrix T of the state equation. Either a m x m matrix or a
#' m x m x n array, or object which can be coerced to such.
#' @param R Lower triangular matrix R the state equation. Either a m x k matrix or a
#' m x k x n array, or object which can be coerced to such.
#' @param a1 Prior mean for the initial state as a vector of length m.
#' @param P1 Prior covariance matrix for the initial state as m x m matrix.
#' @param xreg Matrix containing covariates.
#' @param beta Regression coefficients. Used as an initial
#' value in MCMC. Defaults to vector of zeros.
#' @param state_names Names for the states.
#' @return Object of class \code{gssm}.
#' @export
gssm <- function(y, Z, H, T, R, a1, P1, xreg = NULL, beta = NULL, state_names) {

  check_y(y)

  n <- length(y)

  if (length(Z) == 1) {
    dim(Z) <- c(1, 1)
    m <- 1
  } else {
    if (!(dim(Z)[2] %in% c(1, NA, n)))
      stop("Argument Z must be a vector of length m, or  (m x 1) or (m x n) matrix,
        where m is the number of states and n is the length of the series. ")
    m <- dim(Z)[1]
    dim(Z) <- c(m, (n - 1) * (max(dim(Z)[2], 0, na.rm = TRUE) > 1) + 1)
  }
  if (length(T) == 1 && m == 1) {
    dim(T) <- c(1, 1, 1)
  } else {
    if ((length(T) == 1) || any(dim(T)[1:2] != m) || !dim(T)[3] %in% c(1, NA, n))
      stop("Argument T must be a (m x m) matrix, (m x m x 1) or (m x m x n) array, where m is the number of states. ")
    dim(T) <- c(m, m, (n - 1) * (max(dim(T)[3], 0, na.rm = TRUE) > 1) + 1)
  }

  if (length(R) == m) {
    dim(R) <- c(m, 1, 1)
    k <- 1
  } else {
    if (!(dim(R)[1] == m) || dim(R)[2] > m || !dim(R)[3] %in% c(1, NA, n))
      stop("Argument R must be a (m x k) matrix, (m x k x 1) or (m x k x n) array, where k<=m is the number of disturbances eta, and m is the number of states. ")
    k <- dim(R)[2]
    dim(R) <- c(m, k, (n - 1) * (max(dim(R)[3], 0, na.rm = TRUE) > 1) + 1)
  }

  if (missing(a1)) {
    a1 <- rep(0, m)
  } else {
    if (length(a1) <= m) {
      a1 <- rep(a1, length.out = m)
    } else stop("Misspecified a1, argument a1 must be a vector of length m, where m is the number of state_names and 1<=t<=m.")
  }
  if (missing(P1)) {
    P1 <- matrix(0, m, m)
  } else {
    if (length(P1) == 1 && m == 1) {
      dim(P1) <- c(1, 1)
    } else {
      if (any(dim(P1)[1:2] != m))
        stop("Argument P1 must be (m x m) matrix, where m is the number of states. ")
    }
  }
  if (length(H)[3] %in% c(1, n))
    stop("Argument H must be a scalar or a vector of length n, where n is the length of the time series y.")

  if (missing(state_names)) {
    state_names <- paste("State", 1:m)
  }
  rownames(Z) <- colnames(T) <- rownames(T) <- rownames(R) <- names(a1) <-
    rownames(P1) <- colnames(P1) <- state_names

  if (is.null(xreg)) {

    xreg <- matrix(0, 0, 0)
    beta <- numeric(0)

  } else {

    if (is.null(dim(xreg)) && length(xreg) == n) {
      xreg <- matrix(xreg, n, 1)
    }

    check_xreg(xreg, n)

    if (is.null(colnames(xreg))) {
      colnames(xreg) <- paste0("coef_",1:ncol(xreg))
    }

    if (missing(beta)) {
      beta <- numeric(ncol(xreg))
    } else {
      check_beta(beta, ncol(xreg))
    }

    names(beta) <- colnames(xreg)
  }

  structure(list(y = as.ts(y), Z = Z, H = H, T = T, R = R, a1 = a1, P1 = P1,
    xreg = xreg, beta = beta), class = "gssm")
}

#' Log-likelihood of Exponential Family State Space Model
#'
#' Computes the log-likelihood of exponential family state space model.
#'
#' @param object Model object.
#' @param ... Ignored.
#' @importFrom stats logLik
#' @method logLik gssm
#' @rdname logLik
#' @export
logLik.gssm <- function(object, ...) {
  if (!is.null(dim(object$y)[2]) && dim(object$y)[2] > 1) {
    stop("not yet implemented for multivariate models.")
  }
  gssm_loglik(object)
}

#' @method kfilter gssm
#' @rdname kfilter
#' @export
kfilter.gssm <- function(object, ...) {

  if (!is.null(object$y) && ncol(object$y) > 1) {
    stop("not yet implemented for multivariate models.")
  }

  out <- gssm_filter(object)

  colnames(out$at) <- colnames(out$att) <- colnames(out$Pt) <-
    colnames(out$Ptt) <- rownames(out$Pt) <-
    rownames(out$Ptt) <- names(object$a1)
  out$at <- ts(out$at, start = start(object$y), frequency = frequency(object$y))
  out$att <- ts(out$att, start = start(object$y), frequency = frequency(object$y))
  out
}
#' @method fast_smoother gssm
#' @export
fast_smoother.gssm <- function(object, ...) {
  if (!is.null(object$y) && ncol(object$y) > 1) {
    stop("not yet implemented for multivariate models.")
  }
  out <- gssm_fast_smoother(object$y)

  colnames(out) <- names(object$a1)
  ts(out, start = start(object$y), frequency = frequency(object$y))
}
#' @method sim_smoother gssm
#' @export
sim_smoother.gssm <- function(object, nsim = 1,
  seed = sample(.Machine$integer.max, size = 1), ...) {

  if (!is.null(object$y) && ncol(object$y) > 1) {
    stop("not yet implemented for multivariate models.")
  }

  out <- gssm_sim_smoother(object, nsim, seed)

  rownames(out) <- names(object$a1)
  aperm(out, c(2, 1, 3))
}

#' @method smoother gssm
#' @export
smoother.gssm <- function(object, ...) {

  if (!is.null(object$y) && ncol(object$y) > 1) {
    stop("not yet implemented for multivariate models.")
  }

  out <- gssm_smoother(object)

  colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <- names(object$a1)
  out$alphahat <- ts(out$alphahat, start = start(object$y),
    frequency = frequency(object$y))
  out
}

#' Bayesian Inference of State Space Models using MCMC with RAM
#'
#' For general univariate Gaussian models, all \code{NA} values in
#' \code{Z}, \code{H}, \code{T}, and \code{R} are estimated without any constraints
#' (expect the bounds given by the uniform priors).
#'
#' Note that currently it is not possible to set some parameters equal to each
#' other, so for example stochastic trigonometric seasonal and cycle components
#' cannot be used (the corresponding errors are usually assumed i.i.d.).
#'
#' Note that the proposal for all parameters is multivariate Gaussian,
#' with uniform priors for each parameters. For \code{\link{bsm}} models,
#' generating proposals for standard deviations in log-scale is also possible
#' with argument \code{log_space = TRUE}.
#'
#' @method run_mcmc gssm
#' @rdname run_mcmc_g
#' @param object Model object.
#' @param n_iter Number of MCMC iterations.
#' @param priors Priors for the unknown parameters.
#' @param Z_est,H_est,T_est,R_est Arrays or matrices containing \code{NA}
#' values marking the unknown parameters which are to be estimated. Must be of
#' same dimension as the corresponding elements of the model object.
#' @param sim_states Simulate states of Gaussian state space models. Default is \code{TRUE}.
#' @param type Type of output. Default is \code{"full"}, which returns
#' samples from the posterior \eqn{p(\alpha, \theta}. Option
#' \code{"parameters"} samples only parameters \eqn{\theta} (which includes the
#' regression coefficients \eqn{\beta}). This can be used for faster inference of
#' \eqn{\theta} only, or as an preliminary run for obtaining
#' initial values for \code{S}. Option \code{"summary"} does not simulate
#' states directly computes the  posterior means and variances of states using
#' fast Kalman smoothing. This is slightly faster, memory  efficient and
#' more accurate than calculations based on simulation smoother.
#' \eqn{\theta}. Optional for \code{bsm} objects.
#' @param n_burnin Length of the burn-in period which is disregarded from the
#' results. Defaults to \code{n_iter / 2}.
#' @param n_thin Thinning rate. Defaults to 1. Increase for large models in
#' order to save memory.
#' @param gamma Tuning parameter for the adaptation of RAM algorithm. Must be
#' between 0 and 1 (not checked).
#' @param target_acceptance Target acceptance ratio for RAM. Defaults to 0.234.
#' @param S Initial value for the lower triangular matrix of RAM
#' algorithm, so that the covariance matrix of the Gaussian proposal
#' distribution is \eqn{SS'}.
#' @param end_adaptive_phase If \code{TRUE} (default), $S$ is held fixed after the burnin period.
#' @param seed Seed for the random number generator.
#' @param ... Ignored.
#' @export
run_mcmc.gssm <- function(object, n_iter, Z_est, H_est, T_est, R_est,
  sim_states = TRUE, type = "full", priors,
  n_burnin = floor(n_iter / 2), n_thin = 1, gamma = 2/3,
  target_acceptance = 0.234, S, end_adaptive_phase = TRUE,
  seed = sample(.Machine$integer.max, size = 1), ...) {

  type <- match.arg(type, c("full", "summary"))

  if (!is.null(object$y) && ncol(object$y) > 1) {
    stop("not yet implemented for multivariate models.")
  }

  if (missing(Z_est)) {
    Z_n <- 0
    Z_ind <- numeric(0)
  } else {
    Z_ind <- which(is.na(Z_est)) - 1
    Z_n <- length(Z_ind)
  }
  if (missing(H_est)) {
    H_n <- 0
    H_ind <- numeric(0)
  } else {
    H_ind <- which(is.na(H_est)) - 1
    H_n <- length(H_ind)
  }
  if (missing(T_est)) {
    T_n <- 0
    T_ind <- numeric(0)
  } else {
    T_ind <- which(is.na(T_est)) - 1
    T_n <- length(T_ind)
  }
  if (missing(R_est)) {
    R_n <- 0
    R_ind <- numeric(0)
  } else {
    R_ind <- which(is.na(R_est)) - 1
    R_n <- length(R_ind)
  }

  if (Z_n + H_n + T_n + R_n + length(object$coef) == 0) {
    stop("nothing to estimate. ")
  }

  if (missing(S)) {
    S <- diag(Z_n + H_n + T_n + R_n + length(object$coefs))
  }
priors <- combine_priors(priors)

  out <- switch(type,
    full = {
      out <- gssm_run_mcmc(object, priors$prior_types, priors$params, n_iter,
        nsim_states, n_burnin, n_thin, gamma, target_acceptance, S, Z_ind,
        H_ind, T_ind, R_ind, seed, end_adaptive_phase)
      out$alpha <- aperm(out$alpha, c(2, 1, 3))
      colnames(out$alpha) <- names(object$a1)
      out
    },
    summary = {
      out <- gssm_run_mcmc_summary(object, priors$prior_types, priors$params, n_iter,
        n_burnin, n_thin, gamma, target_acceptance, S, Z_ind, H_ind, T_ind,
        R_ind, seed, end_adaptive_phase)
      colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <-
        names(object$a1)
      out$alphahat <- ts(out$alphahat, start = start(object$y),
        frequency = frequency(object$y))
      out
    }
  )
  out$theta <- mcmc(out$theta, start = n_burnin + 1, thin = n_thin)
  out$call <- match.call()
  class(out) <- "mcmc_output"
  out
}

#' Predictions for Gaussian State Space Models
#'
#' Posterior intervals of future observations or their means
#' for Gaussian models. These are
#' computed using either the quantile method where the intervals are computed
#' as empirical quantiles the posterior sample, or parametric method by
#' Helske (2016).
#'
#' @param object Model object.#'
#' @param priors Priors for the unknown parameters.
#' @param n_ahead Number of steps ahead at which to predict.
#' @param interval Compute predictions on \code{"mean"} ("confidence interval") or
#' \code{"response"} ("prediction interval"). Defaults to \code{"response"}.
#' @param probs Desired quantiles. Defaults to \code{c(0.05, 0.95)}. Always includes median 0.5.
#' @param newdata Matrix containing the covariate values for the future time
#' points. Defaults to zero matrix of appropriate size.
#' @param method Either \code{"parametric"} (default) or \code{"quantile"}.
#' Only used in Gaussian case.
#' @param return_MCSE For method \code{"parametric"}, if \code{TRUE}, the Monte Carlo
#' standard errors are also returned.
#' @param ... Ignored.
#' @inheritParams run_mcmc.gssm
#' @return List containing the mean predictions, quantiles and Monte Carlo
#' standard errors .
#' @method predict gssm
#' @rdname predict
#' @export
predict.gssm <- function(object, n_iter, priors, newdata = NULL,
  n_ahead = 1, interval = "response",
  probs = c(0.05, 0.95), method = "quantile", return_MCSE = TRUE, nsim_states = 1,
  n_burnin = floor(n_iter / 2), n_thin = 1,
  gamma = 2/3, target_acceptance = 0.234, S, Z_est, H_est, T_est, R_est,
  seed = sample(.Machine$integer.max, size = 1), ...) {

  if (!is.null(object$y) && ncol(object$y) > 1) {
    stop("not yet implemented for multivariate models.")
  }

  interval <- pmatch(interval, c("mean", "response"))
  method <- match.arg(method, c("parametric", "quantile"))

  if (missing(Z_est)) {
    Z_n <- 0
    Z_ind <- numeric(0)
  } else {
    Z_ind <- which(is.na(Z_est)) - 1
    Z_n <- length(Z_ind)
  }
  if (missing(H_est)) {
    H_n <- 0
    H_ind <- numeric(0)
  } else {
    H_ind <- which(is.na(H_est)) - 1
    H_n <- length(H_ind)
  }
  if (missing(T_est)) {
    T_n <- 0
    T_ind <- numeric(0)
  } else {
    T_ind <- which(is.na(T_est)) - 1
    T_n <- length(T_ind)
  }
  if (missing(R_est)) {
    R_n <- 0
    R_ind <- numeric(0)
  } else {
    R_ind <- which(is.na(R_est)) - 1
    R_n <- length(R_ind)
  }

  if (missing(S)) {
    S <- diag(Z_n + H_n + T_n + R_n + length(object$coefs))
  }
  endtime <- end(object$y) + c(0, n_ahead)
  y_orig <- object$y
  object$y <- c(object$y, rep(NA, n_ahead))

  if (length(object$coefs) > 0) {
    if (is.null(newdata) || nrow(newdata) != n_ahead ||
        ncol(newdata) != length(object$coefs)) {
      stop("Model contains regression part but newdata is missing or its dimensions do not match with n_ahead and length of beta. ")
    }
    object$xreg <- rbind(object$xreg, newdata)
  }

  if (any(c(dim(object$Z)[3], dim(object$H)[3], dim(object$T)[3], dim(object$R)[3]) > 1)) {
    stop("Time-varying system matrices in prediction are not yet supported.")
  }
  probs <- sort(unique(c(probs, 0.5)))
  priors <- combine_priors(priors)
  if (method == "parametric") {

    out <- gssm_predict(object, priors$prior_types, priors$param, n_iter,
      n_burnin, n_thin, gamma, target_acceptance, S, n_ahead, interval,
      Z_ind, H_ind, T_ind, R_ind, probs, seed)

    if (return_MCSE) {
      ses <- matrix(0, n_ahead, length(probs))

      nsim <- nrow(out$y_mean)

      for (i in 1:n_ahead) {
        for (j in 1:length(probs)) {
          pnorms <- pnorm(q = out$intervals[i, j], out$y_mean[, i], out$y_sd[, i])
          eff_n <-  effectiveSize(pnorms)
          ses[i, j] <- sqrt((sum((probs[j] - pnorms) ^ 2) / nsim) / eff_n) /
            sum(dnorm(x = out$intervals[i, j], out$y_mean[, i], out$y_sd[, i]) / nsim)
        }
      }

      pred <- list(y = object$y, mean = ts(colMeans(out$y_mean), end = endtime, frequency = object$period),
        intervals = ts(out$intervals, end = endtime, frequency = object$period,
          names = paste0(100*probs, "%")),
        MCSE = ts(ses, end = endtime, frequency = object$period,
          names = paste0(100*probs, "%")))
    } else {
      pred <- list(y = object$y, mean = ts(colMeans(out$y_mean), end = endtime, frequency = object$period),
        intervals = ts(out$intervals, end = endtime, frequency = object$period,
          names = paste0(100 * probs, "%")))
    }
  } else {

    out <- gssm_predict2(object, priors$prior_types, priors$param, n_iter, nsim_states,
      n_burnin, n_thin, gamma, target_acceptance, S, n_ahead, interval,
      Z_ind, H_ind, T_ind, R_ind, seed)

    pred <- list(y = y_orig, mean = ts(rowMeans(out), end = endtime, frequency = object$period),
      intervals = ts(t(apply(out, 1, quantile, probs, type = 8)), end = endtime, frequency = object$period,
        names = paste0(100 * probs, "%")))

  }
  class(pred) <- "predict_bssm"
  pred
}

#' @method particle_filter gssm
#' @rdname particle_filter
#' @export
particle_filter.gssm <- function(object, nsim,
  seed = sample(.Machine$integer.max, size = 1), ...) {

  out <- gssm_particle_filter(object, nsim, seed)

  rownames(out$alpha) <- names(object$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}

#' @method particle_smoother gssm
#' @rdname particle_smoother
#' @export
particle_smoother.gssm <- function(object, nsim, method = "fs",
  seed = sample(.Machine$integer.max, size = 1), ...) {

  method <- match.arg(method, c("fs", "fbs"))
  out <- gssm_particle_smoother(object, nsim, seed, method == "fs")

  rownames(out$alpha) <- names(object$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}

#' @method particle_simulate gssm
#' @rdname particle_simulate
#' @export
particle_simulate.gssm <- function(object, nsim, nsim_store = 1,
  seed = sample(.Machine$integer.max, size = 1), ...) {

  out <- gssm_backward_simulate(object, nsim, seed, nsim_store)

  rownames(out$alpha) <- names(object$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}
