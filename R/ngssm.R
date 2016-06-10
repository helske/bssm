#' General Gaussian State Space Models
#'
#' Construct an object of class \code{gssm} by defining the corresponding terms
#' of the observation and state equation:
#'
#' \deqn{y_t = Z_t \alpha_t + H_t \epsilon_t, (\textrm{observation equation})}
#' \deqn{\alpha_{t+1} = T_t \alpha_t + R_t \eta_t, (\textrm{transition equation})}
#'
#' where \eqn{\epsilon_t \sim N(0, I_p)}, \eqn{\eta_t \sim N(0, I_k)} and
#' \eqn{\alpha_1 \sim N(a_1, P_1)} independently of each other.
#'
#' @param y Observations as time series (or vector) of length \eqn{n}.
#' @param Z System matrix Z of the observation equation. Either a p x m matrix
#' or a p x m x n array, or object which can be coerced to such.
#' @param T System matrix T of the state equation. Either a m x m matrix or a
#' m x m x n array, or object which can be coerced to such.
#' @param R Lower triangular matrix R the state equation. Either a m x k matrix or a
#' m x k x n array, or object which can be coerced to such.
#' @param a1 Prior mean for the initial state as a vector of length m.
#' @param P1 Prior covariance matrix for the initial state as m x m matrix.
#' @param distribution distribution of the observation. Possible choices are
#' \code{"poisson"}, \code{"binomial"}, and \code{"negative binomial"}.
#' @param phi Additional parameter vector relating to the non-Gaussian distribution.
#' For Poisson distribution, this corresponds to offset term. For binomial, this
#' is the number of trials.
#' @param xreg Matrix containing covariates.
#' @param beta Regression coefficients. Used as an initial
#' value in MCMC. Defaults to vector of zeros.
#' @param state_names Names for the states.
#' @return Object of class \code{bgssm}.
#' @export
ngssm <- function(y, Z, T, R, a1, P1,
  distribution, phi = 1, xreg = NULL, beta = NULL, state_names) {

  if (length(dim(y)) != 2) {
    p <- 1
    n <- length(y)
    dim(y) <- c(n, p)
  } else {
    n <- dim(y)[1]
    p <- dim(y)[2]
  }

  if (p > 1) {
    stop("multivariate models are not yet supported. ")
  }
  if ( p > 1 && is.null(colnames(y))) {
    colnames(y) <- paste0("Series ", 1:p)
  }
  class(y) <- if (p > 1) c("mts", "ts", "matrix") else "ts"
  if (is.null(tsp(y))) {
    tsp(y) <- c(1, n, 1)
  }
  if (length(Z) == 1 && p == 1) {
    dim(Z) <- c(1, 1, 1)
    m <- 1
  } else {
    if ((length(Z) == 1) || !(dim(Z)[1] == p) || !dim(Z)[3] %in% c(1, NA, n))
      stop("Argument Z must be a (p x m) matrix, (p x m x 1) or (p x m x n) array,
        where p is the number of time series, m is the number of states. ")
    m <- dim(Z)[2]
    dim(Z) <- c(p, m, (n - 1) * (max(dim(Z)[3], 0, na.rm = TRUE) > 1) + 1)
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

  if (!(length(phi) %in% c(1, n))) {
    stop("Argument phi must have length 1 or n. ")
  }

  if (length(phi) != n) {
    phi <- rep(phi, length.out = n)
  }

  if (missing(state_names)) {
    state_names <- paste("State", 1:m)
  }
  rownames(Z) <- colnames(T) <- rownames(T) <- rownames(R) <- names(a1) <-
    rownames(P1) <- colnames(P1) <- state_names


  if (is.null(xreg)) {
    xreg <- matrix(0,0,0)
    beta <- numeric(0)
  } else {
    if (is.null(dim(xreg))) {
      xreg <- matrix(xreg, n, 1)
    } else {
      if (nrow(xreg) != n)
        stop("Number of rows in xreg is not equal to the length of the series y.")
    }
    if (is.null(colnames(xreg))) {
      colnames(xreg) <- paste0("coef_",1:ncol(xreg))
    }
    names(beta) <- colnames(xreg)
  }
  structure(list(y = y, Z = Z, T = T, R = R, a1 = a1, P1 = P1, phi = phi,
    xreg = xreg, beta = beta, distribution =
      match.arg(distribution, c("poisson", "binomial", "negative binomial"))), class = "ngssm")
}

#' @method logLik ngssm
#' @rdname logLik
#' @export
logLik.ngssm <- function(object, ...) {
  if (!is.null(ncol(object$y)) && ncol(object$y) > 1) {
    stop("not yet implemented for multivariate models.")
  }

  nguvssm_loglik(object$y, object$Z, object$T, object$R, object$a1,
    object$P1, object$phi, object$xreg, object$beta,
    pmatch(object$distribution, c("poisson", "binomial")),
    initial_signal(object$y, object$phi, object$distribution))

}

#' @importFrom stats qlogis
initial_signal <- function(y, phi, distribution) {
  # time series division is much slower than matrix division
  y <- unclass(y)
  phi <- unclass(phi)
  if (distribution == "poisson") {
    y <- y/phi
    y[y < 0.1 | is.na(y)] <- 0.1
    y <- log(y)
  }
  if (distribution == "binomial") {
    y <- qlogis((ifelse(is.na(y), 0.5, y) + 0.5)/(phi + 1))
  }
  if (distribution == "gamma") {
    y[is.na(y) | y < 1] <- 1
    y <- log(y)
  }
  if (distribution == "negative binomial") {
    y[is.na(y) | y < 1/6] <- 1/6
    y <- log(y)
  }
  y
}
#' Bayesian Inference of State Space Models using MCMC with RAM
#'
#' Adaptive Markov chain Monte Carlo simulation of state space models using
#' Robust Adaptive Metropolis algorithm by Vihola (2012).
#'
#' @method run_mcmc ngssm
#' @rdname run_mcmc_ng
#' @param object Model object.
#' @param n_iter Number of MCMC iterations.
#' @param Z_est,T_est,R_est Matrices or arrays with same dimensions as the
#' corresponding system matrices in \code{object}, where \code{NA} values
#' identify the unknown parameters for estimation.
#' @param nsim_states Number of simulations of states per MCMC iteration. Only
#' used when \code{type = "full"}.
#' @param lower_prior,upper_prior Bounds of the uniform prior for parameters
#' \eqn{\theta}. Optional for \code{bstsm} objects.
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
#' @param seed Seed for Boost random number generator.
#' @param ... Ignored.
#' @export
run_mcmc.ngssm <- function(object, n_iter, Z_est, T_est, R_est, lower_prior, upper_prior,
  nsim_states = 1, n_burnin = floor(n_iter/2), n_thin = 1, gamma = 2/3,
  target_acceptance = 0.234, S, seed = sample(.Machine$integer.max, size = 1),
  ...) {

  if (!is.null(ncol(object$y)) && ncol(object$y) > 1) {
    stop("not yet implemented for multivariate models.")
  }

  if (missing(Z_est)) {
    Z_n <- 0
    Z_ind <- numeric(0)
  } else {
    Z_ind <- which(is.na(Z_est)) - 1
    Z_n <- length(Z_ind)
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
  nb <- object$distribution == "negative binomial"

  if (Z_n + T_n + R_n + (nb == 0)) {
    stop("nothing to estimate. ")
  }

  if (missing(S)) {
    S <- diag(Z_n  + T_n + R_n + nb)
  }
  if (length(lower_prior) == length(upper_prior) && nrow(S) == length(lower_prior)) {
    stop("Number of unknown parameters is not equal to the length of the prior vector.")
  }

  out <- nguvssm_mcmc_full(object$y, object$Z, object$T, object$R,
    object$a1, object$P1, object$phi, pmatch(object$distribution, c("poisson", "binomial", "negative binomial")),
    lower_prior, upper_prior, n_iter,
    nsim_states, n_burnin, n_thin, gamma, target_acceptance, S, Z_ind, T_ind,
    R_ind, object$xreg, object$beta, initial_signal(object$y, object$phi, object$distribution), seed)

  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  colnames(out$alpha) <- names(object$a1)
  out

  out$S <- matrix(out$S, length(lower_prior), length(lower_prior))
  if (nb) {
    out$theta[, ncol(out$theta)] <- exp(out$theta[, ncol(out$theta)])
  }
  out$theta <- mcmc(out$theta, start = n_burnin + 1, thin = n_thin)
  out
}

#' Predictions for Non-Gaussian State Space Models
#'
#' Posterior intervals of future observations or their means
#' (success probabilities in binomial case) for non-Gaussian models. These are
#' computed using either the quantile method where the intervals are computed
#' as empirical quantiles the posterior sample, or parametric method by
#' Helske (2016).
#' Note that for non-Gaussian models, the parametric method produces only approximate
#' intervals for the mean even when importance sampling is used, by assuming
#' that \eqn{p(\alpha | y)} is Gaussian.
#'
#' @return List containing the mean predictions, quantiles and Monte Carlo
#' standard errors.
#' @method predict ngssm
#' @rdname predict.ngssm
#' @inheritParams predict.gssm
#' @param newphi Vector of length \code{n_ahead} defining the future values of \eqn{\phi}.
#' Defaults to 1, expect for negative binomial distribution, where the initial
#' value is taken from \code{object$phi}.
#' @param Z_est,T_est,R_est Matrices or arrays with same dimensions as the
#' corresponding system matrices in \code{object}, where \code{NA} values
#' identify the unknown parameters for estimation.
#' @export
predict.ngssm <- function(object, n_iter, nsim_states, lower_prior, upper_prior,
  newdata = NULL,
  n_ahead = 1, interval = "mean",
  probs = c(0.05, 0.95),
  n_burnin = floor(n_iter / 2), n_thin = 1,
  gamma = 2/3, target_acceptance = 0.234, S,
  seed = sample(.Machine$integer.max, size = 1),  newphi = NULL, Z_est, T_est, R_est, ...) {

  if (!is.null(ncol(object$y)) && ncol(object$y) > 1) {
    stop("not yet implemented for multivariate models.")
  }

  interval <- pmatch(interval, c("mean", "response"))

  if (missing(Z_est)) {
    Z_n <- 0
    Z_ind <- numeric(0)
  } else {
    Z_ind <- which(is.na(Z_est)) - 1
    Z_n <- length(Z_ind)
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


  nb <- object$distribution == "negative binomial"

  if (Z_n + T_n + R_n + nb == 0) {
    stop("nothing to estimate. ")
  }

  if (missing(S)) {
    S <- diag(Z_n  + T_n + R_n + nb)
  }
  if (length(lower_prior) == length(upper_prior) && nrow(S) == length(lower_prior)) {
    stop("Number of unknown parameters is not equal to the length of the prior vector.")
  }

  endtime <- end(object$y) + c(0, n_ahead)
  y <- c(object$y, rep(NA, n_ahead))

  if (length(object$beta) > 0) {
    if (is.null(newdata) || nrow(newdata) != n_ahead ||
        ncol(newdata) != length(object$beta)) {
      stop("Model contains regression part but newdata is missing or its dimensions do not match with n_ahead and length of beta. ")
    }
    object$xreg <- rbind(object$xreg, newdata)
  }

  if (is.null(newphi) || length(newphi) != n_ahead) {
    stop("newphi is missing or its length is not equal to n_ahead. ")
  }


  if (any(c(dim(object$Z)[3], dim(object$T)[3], dim(object$R)[3]) > 1)) {
    stop("Time-varying system matrices in prediction are not yet supported.")
  }
  object$phi <- c(object$phi, newphi)
  probs <- sort(unique(c(probs, 0.5)))

  out <- nguvssm_predict2(y, object$Z, object$T, object$R,
    object$a1, object$P1, object$phi, pmatch(object$distribution, c("poisson", "binomial")),
    lower_prior, upper_prior, n_iter,
    nsim_states, n_burnin, n_thin, gamma, target_acceptance, S, n_ahead, interval,
    Z_ind, T_ind, R_ind, object$xreg, object$beta,
    initial_signal(y, object$phi, object$distribution), seed)

  pred <- list(y = object$y, mean = ts(rowMeans(out), end = endtime, frequency = object$period),
    intervals = ts(t(apply(out, 1, quantile, probs, type = 8)), end = endtime, frequency = object$period,
      names = paste0(100 * probs, "%")))


  class(pred) <- "predict_bssm"
  pred

}
