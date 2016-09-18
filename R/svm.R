#' Stochastic Volatility Model
#'
#' Constructs a simple stochastic volatility model with Gaussian errors and
#' first order autoregressive signal.
#'
#' @param y Vector or a \code{\link{ts}} object of observations.
#' @param ar prior for autoregressive coefficient.
#' @param sigma Prior for sigma parameter of observation equation.
#' @param sd_ar Prior for the standard deviation of noise of the AR-process.
#' @return Object of class \code{svm}.
#' @export
svm <- function(y, ar, sd_ar, sigma, beta, xreg = NULL) {

  check_y(y)
  n <- length(y)

  check_ar(ar$init)
  check_sd(sd_ar$init, "ar")
  check_sd(sigma$init, "sigma", FALSE)

  if (is.null(xreg)) {

    xreg <- matrix(0, 0, 0)
    coefs <- numeric(0)
    beta <- NULL

  } else {

    if (missing(beta)) {
      stop("No prior defined for beta. ")
    }
    if (is.null(dim(xreg)) && length(xreg) == n) {
      xreg <- matrix(xreg, n, 1)
    }

    check_xreg(xreg, n)
    check_beta(beta$init, ncol(xreg))
    coefs <- beta$init
    if (is.null(colnames(xreg))) {
      colnames(xreg) <- paste0("coef_",1:ncol(xreg))
    }
    names(coefs) <- colnames(xreg)

  }

  a1 <- 0
  P1 <- matrix(sd_ar$init^2 / (1 - ar$init^2))

  Z <- matrix(1)
  T <- array(ar$init, c(1, 1, 1))
  R <- array(sd_ar$init, c(1, 1, 1))

  init_signal <- log(pmax(1e-4, y^2)) - 2 * log(sigma$init)

  names(a1) <- rownames(P1) <- colnames(P1) <- rownames(Z) <-
    rownames(T) <- colnames(T) <- rownames(R) <- "signal"

  priors <- list(ar, sd_ar, sigma, beta)
  priors <- priors[!sapply(priors, is.null)]
  names(priors) <-
    c("ar", "sd_ar", "sigma", names(coefs))

  structure(list(y = as.ts(y), Z = Z, T = T, R = R,
    a1 = a1, P1 = P1, sigma = sigma$init, xreg = xreg, coefs = coefs,
    init_signal = init_signal, priors = priors), class = "svm")
}

#' @method logLik svm
#' @rdname logLik
#' @inheritParams logLik.ngssm
#' @export
logLik.svm <- function(object, nsim_states, seed = 1, ...) {
  object$distribution <- 0
  object$phi <- rep(object$sigma, length(object$y))
  svm_loglik(object, object$init_signal, nsim_states, seed)
}
#' @method kfilter svm
#' @rdname kfilter
#' @export
kfilter.svm <- function(object, ...) {

  object$distribution <- 0
  object$phi <- rep(object$sigma, length(object$y))
  out <- ngssm_filter(object, object$init_signal)

  colnames(out$at) <- colnames(out$att) <- colnames(out$Pt) <-
    colnames(out$Ptt) <- rownames(out$Pt) <-
    rownames(out$Ptt) <- names(object$a1)
  out$at <- ts(out$at, start = start(object$y)svm)
  out$att <- ts(out$att, start = start(object$y)svm)
  out
}
#' @method fast_smoother svm
#' @export
fast_smoother.svm <- function(object, ...) {

  object$distribution <- 0
  object$phi <- rep(object$sigma, length(object$y))
  out <- svm_fast_smoother(object, object$init_signal)
  colnames(out) <- names(object$a1)
  ts(out, start = start(object$y), frequency = frequency(object$y))

}
#' @method sim_smoother svm
#' @export
sim_smoother.svm <- function(object, nsim = 1,
  seed = sample(.Machine$integer.max, size = 1), ...) {

  object$distribution <- 0
  object$phi <- rep(object$sigma, length(object$y))
  out <- svm_sim_smoother(object, object$init_signal, nsim, seed)

  rownames(out) <- names(object$a1)
  aperm(out, c(2, 1, 3))
}

#' @method smoother svm
#' @export
smoother.svm <- function(object, ...) {
  object$distribution <- 0
  object$phi <- rep(object$sigma, length(object$y))
  out <- svm_smoother(object, object$init_signal)

  colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <- names(object$a1)
  out$alphahat <- ts(out$alphahat, start = start(object$y))
  out
}

#' @method run_mcmc svm
#' @rdname run_mcmc_ng
#' @inheritParams run_mcmc.ngssm
#' @export
run_mcmc.svm <- function(object, n_iter, nsim_states = 1, type = "full",
  method = "PM", simulation_method = "IS", correction_method = "IS2",
  delayed_acceptance = TRUE,
  n_burnin = floor(n_iter/2),
  n_thin = 1, gamma = 2/3, target_acceptance = 0.234, S, end_adaptive_phase = TRUE,
  adaptive_approx  = TRUE, n_threads = 1,
  seed = sample(.Machine$integer.max, size = 1), ...) {

  type <- match.arg(type, c("full", "summary"))
  method <- match.arg(method, c("PM", "IS"))
  simulation_method <- match.arg(simulation_method, c("IS", "PF"))
  correction_method <- match.arg(correction_method, c("IS1", "IS2", "PF"))

  if (n_thin > 1 && method %in% c("block IS correction", "IS2")) {
    stop ("Cannot use thinning with block-IS algorithm.")
  }

  if (missing(S)) {
    S <- diag(0.1 * pmax(0.1, abs(sapply(object$priors, "[[", "init"))), length(object$priors))
  }


  if (nsim_states < 2) {
    #approximate inference
    method <- "PM"
    simulation_method <- "IS"
  }
  priors <- combine_priors(object$priors)

  object$distribution <- 0
  object$phi <- rep(object$sigma, length(object$y))
  out <-  switch(type,
    full = {
      if (method == "PM") {
        out <- svm_run_mcmc(object,
          priors$prior_types, priors$params, n_iter,
          nsim_states, n_burnin, n_thin, gamma, target_acceptance, S,
          object$init_signal, seed,  n_threads, end_adaptive_phase, adaptive_approx,
          delayed_acceptance, simulation_method == "PF")

      } else {
        out <- svm_run_mcmc_is(object,
          priors$prior_types, priors$params, n_iter,
          nsim_states, n_burnin, n_thin, gamma, target_acceptance, S,
          object$init_signal, seed,  n_threads, end_adaptive_phase, adaptive_approx,
          pmatch(correction_method, c("IS1", "IS2", "PF")))
      }
      out$alpha <- aperm(out$alpha, c(2, 1, 3))
      colnames(out$alpha) <- names(object$a1)
      out
    },
    summary = {
      stop("summary for sv models not yet implemented.")
      # out <- svm_run_mcmc_summary(object,
      #   priors$prior_types, priors$params, n_iter,
      #   nsim_states, n_burnin, n_thin, gamma, target_acceptance, S,
      #   object$init_signal, seed,  n_threads, end_adaptive_phase, adaptive_approx,
      #   delayed_acceptance, simulation_method == "PF")
      #
      # colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <- names(object$a1)
      # out$alphahat <- ts(out$alphahat, start = start(object$y), frequency = frequency(object$y))
      # out$muhat <- ts(out$muhat, start = start(object$y), frequency = frequency(object$y))
      # out
    })
  out$S <- matrix(out$S, nrow(S), ncol(S))

  colnames(out$theta) <- rownames(out$S) <- colnames(out$S) <-
    c("ar", "sd_ar", "sigma", names(object$coefs))
  if(method == "PM") {
    out$theta <- mcmc(out$theta, start = n_burnin + 1, thin = n_thin)
  }
  out$call <- match.call()
  out$seed <- seed
  class(out) <- "mcmc_output"
  out
}


#' @method importance_sample svm
#' @rdname importance_sample
#' @export
importance_sample.svm <- function(object, nsim,
  seed = sample(.Machine$integer.max, size = 1), ...) {

  object$distribution <- 0
  object$phi <- rep(object$sigma, length(object$y))
  svm_importance_sample(object,
    nsim, object$init_signal, seed)
}

#' @method gaussian_approx svm
#' @rdname gaussian_approx
#' @export
gaussian_approx.svm <- function(object, max_iter = 100, conv_tol = 1e-8, ...) {

  object$distribution <- 0
  object$phi <- rep(object$sigma, length(object$y))
  svm_approx_model(object,
    object$init_signal, max_iter, conv_tol)
}


#' @method particle_filter svm
#' @rdname particle_filter
#' @export
particle_filter.svm <- function(object, nsim,
  seed = sample(.Machine$integer.max, size = 1), ...) {

  object$distribution <- 0
  object$phi <- rep(object$sigma, length(object$y))
  svm_particle_filter(object, nsim, seed)
}


#' @method particle_smoother svm
#' @rdname particle_smoother
#' @export
particle_smoother.svm <- function(object, nsim, method = "fs",
  seed = sample(.Machine$integer.max, size = 1), ...) {

  method <- match.arg(method, c("fs", "fbs"))
  object$distribution <- 0
  object$phi <- rep(object$sigma, length(object$y))
  out <- svm_particle_smoother(object, nsim, seed, method == "fs")

  rownames(out$alpha) <- names(object$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}

#' @method particle_simulate svm
#' @rdname particle_simulate
#' @export
particle_simulate.svm <- function(object, nsim, nsim_store = 1,
  seed = sample(.Machine$integer.max, size = 1), ...) {

  object$distribution <- 0
  object$phi <- rep(object$sigma, length(object$y))
  out <- svm_backward_simulate(object, nsim, seed, nsim_store)

  rownames(out$alpha) <- names(object$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}
