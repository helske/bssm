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
svm <- function(y, ar, sd_ar, sigma, xreg = NULL, beta = NULL, lower_prior, upper_prior) {

  check_y(y)
  n <- length(y)


  if (missing(ar)) {
    ar <- 0.5
  } else {
    if (abs(ar) >= 1) {
      stop("Argument ar must be between -1 and 1.")
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

  a1 <- 0
  P1 <- matrix(sd_ar^2 / (1 - ar^2))

  Z <- matrix(1)
  T <- array(ar, c(1,1,1))
  R <- array(sd_ar, c(1,1,1))

  init_signal <- (log(pmax(1e-4,y^2)) - log(sigma^2))

  if (missing(lower_prior)) {
    lower_prior <- c(1e-8, 1e-8, 1e-8, rep(-1e4,length(beta)))
  }

  if (missing(upper_prior)) {
    upper_prior <- c(1 - 1e-8, 1e4, 1e4, rep(1e4,length(beta)))
  }

  names(a1) <- rownames(P1) <- colnames(P1) <- rownames(Z) <-
    rownames(T) <- colnames(T) <- rownames(R) <- "signal"

  names(lower_prior) <- names(upper_prior) <-
    c("ar", "sd_ar", "sigma", names(beta))

  structure(list(y = as.ts(y), Z = Z, T = T, R = R,
    a1 = a1, P1 = P1, sigma = sigma, xreg = xreg, beta = beta,
    lower_prior = lower_prior, upper_prior = upper_prior,
    init_signal = init_signal), class = "svm")
}

#' @method logLik svm
#' @rdname logLik
#' @inheritParams logLik.ngssm
#' @export
logLik.svm <- function(object, nsim_states,
  seed = 1, ...) {
  svm_loglik(object$y, object$Z, object$T, object$R,
    object$a1, object$P1, rep(object$sigma, length(object$y)),
    object$xreg, object$beta, object$init_signal, nsim_states, seed)
}

#' @method smoother svm
#' @export
smoother.svm <- function(object, ...) {

  out <- svm_smoother(object$y, object$Z, object$T, object$R,
    object$a1, object$P1, rep(object$sigma, length(object$y)),
    object$xreg, object$beta, object$init_signal)

  colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <- names(object$a1)
  out$alphahat <- ts(out$alphahat, start = start(object$y))
  out
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

  method <- match.arg(method, c("standard", "delayed acceptance",
    "IS correction", "block IS correction", "IS2"))

  if (missing(lower_prior)) {
    lower_prior <- object$lower_prior
  }
  if (missing(upper_prior)) {
    upper_prior <- object$upper_prior
  }

  if (missing(S)) {
    S <- diag(c(0.1,0.1,0.1, rep(1,length(object$beta))))
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
        object$a1, object$P1,
        rep(object$sigma, length(object$y)), object$xreg, object$beta,
        lower_prior, upper_prior, n_iter,
        nsim_states, n_burnin, n_thin, gamma, target_acceptance, S,
        object$init_signal, 1, seed, n_threads, seeds)

      out$alpha <- aperm(out$alpha, c(2, 1, 3))
      colnames(out$alpha) <- names(object$a1)
      out
    },
    "delayed acceptance" = {
      out <- svm_mcmc_full(object$y, object$Z, object$T, object$R,
        object$a1, object$P1,
        rep(object$sigma, length(object$y)), object$xreg, object$beta,
        lower_prior, upper_prior, n_iter,
        nsim_states, n_burnin, n_thin, gamma, target_acceptance, S,
        object$init_signal, 2, seed, n_threads, seeds)

      out$alpha <- aperm(out$alpha, c(2, 1, 3))
      colnames(out$alpha) <- names(object$a1)
      out
    },
    "IS correction" = {
      out <- svm_mcmc_full(object$y, object$Z, object$T, object$R,
        object$a1, object$P1, rep(object$sigma, length(object$y)), object$xreg, object$beta,
        lower_prior, upper_prior, n_iter,
        nsim_states, n_burnin, n_thin, gamma, target_acceptance, S,
        object$init_signal, 3, seed, n_threads, seeds)

      out$alpha <- aperm(out$alpha, c(2, 1, 3))
      colnames(out$alpha) <- names(object$a1)
      out
    },
    "block IS correction" = {
      out <- svm_mcmc_full(object$y, object$Z, object$T, object$R,
        object$a1, object$P1,rep(object$sigma, length(object$y)), object$xreg, object$beta,
        lower_prior, upper_prior, n_iter,
        nsim_states, n_burnin, n_thin, gamma, target_acceptance, S,
        object$init_signal, 4, seed, n_threads, seeds)

      out$alpha <- aperm(out$alpha, c(2, 1, 3))
      colnames(out$alpha) <- names(object$a1)
      out
    },
    "IS2" = {
      out <- svm_mcmc_full(object$y, object$Z, object$T, object$R,
        object$a1, object$P1,rep(object$sigma, length(object$y)), object$xreg, object$beta,
        lower_prior, upper_prior, n_iter,
        nsim_states, n_burnin, n_thin, gamma, target_acceptance, S,
        object$init_signal, 5, seed, n_threads, seeds)

      out$alpha <- aperm(out$alpha, c(2, 1, 3))
      colnames(out$alpha) <- names(object$a1)
      out
    }
  )
  out$S <- matrix(out$S, length(lower_prior), length(lower_prior))

  colnames(out$theta) <- rownames(out$S) <- colnames(out$S) <-
    c("ar", "sd_ar", "sigma", names(object$beta))
  out$theta <- mcmc(out$theta, start = n_burnin + 1, thin = n_thin)
  out$call <- match.call()
  class(out) <- "mcmc_output"
  out
}


#' @method importance_sample svm
#' @rdname importance_sample
#' @export
importance_sample.svm <- function(object, nsim,
  seed = sample(.Machine$integer.max, size = 1), ...) {
  
  svm_importance_sample(object$y, object$Z, object$T, object$R,
  object$a1, object$P1, rep(object$sigma, length(object$y)), object$xreg, object$beta,
  nsim, object$init_signal, seed)
}

#' @method gaussian_approx svm
#' @rdname gaussian_approx
#' @export
gaussian_approx.svm <- function(object, max_iter = 100, conv_tol = 1e-8, ...) {
  
 svm_approx_model(object$y, object$Z, object$T, object$R,
    object$a1, object$P1, rep(object$sigma, length(object$y)), object$xreg, object$beta,
    object$init_signal, max_iter, conv_tol)
}
