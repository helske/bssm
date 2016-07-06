#' Stochastic Volatility Model
#'
#' Constructs a simple stochastic volatility model with Gaussian errors and first order
#' autoregressive signal.
#'
#' @param y Vector or a \code{\link{ts}} object of observations.
#' @param mean Mean. Used as an initial value in MCMC.
#' @param phi AR coefficient. Used as an initial
#' value in MCMC.
#' @param sigma Sigma. Used as an initial value in MCMC.
#' @param sd_ar Standard error of AR-process. Used as an initial
#' value in MCMC.
#' @param lower_prior,upper_prior Prior bounds for uniform priors on mean, phi, and sds.
#' @return Object of class \code{svm}.
#' @export
svm <- function(y, mean, phi, sigma, sd_ar, lower_prior, upper_prior) {

  check_y(y)
  n <- length(y)

  if (missing(mean)) {
    mean <- 0
  }
  if (missing(phi)) {
    phi <- 0.5
  } else {
    if (abs(phi) >= 1) {
      stop("Argument phi must be between -1 and 1.")
    }
  }
  if (missing(sigma)) {
    sigma <- 1
  } else {
    if (sigma <= 0) {
      stop("Argument sigma must be positive.")
    }
  }
  if (missing(sd_ar)) {
    sd_ar <- 1
  } else {
    if (sd_ar <= 0) {
      stop("Argument sd_ar must be positive.")
    }
  }

  a1 <- 0
  P1 <- matrix(sd_ar^2 / (1 - phi^2))

  Z <- array(1, c(1,1,1))
  T <- array(phi, c(1,1,1))
  R <- array(sd_ar, c(1,1,1))

  init_signal <- (log(pmax(1e-4,y^2)) - log(sigma^2))

  if (missing(lower_prior)) {
    lower_prior <- c(-1e4, 1e-8, 1e-8, 1e-8)
  }

  if (missing(upper_prior)) {
    upper_prior <- c(1e4, 1 - 1e-8, 1e4, 1e4)
  }

  names(a1) <- rownames(P1) <- colnames(P1) <- rownames(Z) <-
    rownames(T) <- colnames(T) <- rownames(R) <- "signal"

  names(lower_prior) <- names(upper_prior) <-
    c("mean", "phi", "sigma", "sd_ar")

  structure(list(y = as.ts(y), Z = Z, T = T, R = R,
    a1 = a1, P1 = P1, mean = mean, sigma = sigma,
    lower_prior = lower_prior, upper_prior = upper_prior,
    init_signal = init_signal), class = "svm")
}

#' @method run_mcmc svm
#' @rdname run_mcmc_ng
#' @inheritParams run_mcmc.ngssm
#' @export
run_mcmc.svm <- function(object, n_iter, nsim_states = 1,
  lower_prior, upper_prior, n_burnin = floor(n_iter/2),
  n_thin = 1, gamma = 2/3, target_acceptance = 0.234, S,
  seed = sample(.Machine$integer.max, size = 1),
  method = "delayed acceptance",  n_threads = 1,
  seeds = sample(.Machine$integer.max, size = n_threads), ...) {

  method <- match.arg(method, c("standard", "DA", "IS1", "IS2"))

  if (missing(lower_prior)) {
    lower_prior <- object$lower_prior
  }
  if (missing(upper_prior)) {
    upper_prior <- object$upper_prior
  }

  if (missing(S)) {
    S <- diag(4)
  }
  if (nsim_states < 2) {
    #approximate inference
    method <- "standard"
    nsim_states <- 1
  }

  # this is stupid, correct!
  out <- switch(method,
    standard = {
      out <- svm_mcmc_full(object$y, object$Z, object$T, object$R,
        object$a1, object$P1, object$mean, object$sigma,
        lower_prior, upper_prior, n_iter,
        nsim_states, n_burnin, n_thin, gamma, target_acceptance, S,
        object$init_signal, 1, seed, n_threads, seeds)

      out$alpha <- aperm(out$alpha, c(2, 1, 3))
      colnames(out$alpha) <- names(object$a1)
      out
    },
    "DA" = {
      out <- svm_mcmc_full(object$y, object$Z, object$T, object$R,
        object$a1, object$P1,object$mean, object$sigma,
        lower_prior, upper_prior, n_iter,
        nsim_states, n_burnin, n_thin, gamma, target_acceptance, S,
        object$init_signal, 2, seed, n_threads, seeds)

      out$alpha <- aperm(out$alpha, c(2, 1, 3))
      colnames(out$alpha) <- names(object$a1)
      out
    }
  )
  out$S <- matrix(out$S, length(lower_prior), length(lower_prior))

  colnames(out$theta) <- rownames(out$S) <- colnames(out$S) <-
    c("mean", "phi", "sigma", "sd_ar")
  out$theta <- mcmc(out$theta, start = n_burnin + 1, thin = n_thin)
  out$call <- match.call()
  class(out) <- "mcmc_output"
  out
}
