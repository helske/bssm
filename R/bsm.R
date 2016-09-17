#' Basic Structural (Time Series) Model
#'
#' Constructs a basic structural model with local level or local trend component
#' and seasonal component.
#'
#' @param y Vector or a \code{\link{ts}} object of observations.
#' @param sd_y Prior for the standard error of observation equation.
#' See \link[=uniform]{priors} for details.
#' @param sd_level  Prior for the standard error of the noise in level equation.
#' See\link[=uniform]{priors} for details. If missing, \code{sd_level} is fixed to zero.
#' @param sd_slope Prior for the standard error  of the noise in slope equation.
#' See\link[=uniform]{priors} for details. If missing, \code{sd_slope} is fixed to zero.
#' Ignored if \code{slope = FALSE}.
#' @param sd_seasonal Prior for the standard error of the noise in seasonal equation.
#' See\link[=uniform]{priors} for details. If missing, \code{sd_seasonal} is fixed to zero.
#' @param xreg Matrix containing covariates.
#' @param beta Prior for the regression coefficients.
#' @param period Length of the seasonal component i.e. the number of
#' observations per season. Default is \code{frequency(y)}.
#' @param slope Should the model contain the slope term. Default is \code{TRUE}.
#' If \code{FALSE}, local level component is used instead of local trend component.
#' @param seasonal Should the model contain seasonal term. Default is
#' \code{frequency(y) > 1}.
#' @param a1 Prior means for the initial states (level, slope, seasonals).
#' Defaults to vector of zeros.
#' @param P1 Prior covariance for the initial states (level, slope, seasonals).
#' Default is diagonal matrix with 1000 on the diagonal.
#' @return Object of class \code{bsm}.
#' @export
#' @examples
#'
#' prior <- uniform(0.1 * sd(log10(UKgas)), 0, 1)
#' model <- bsm(log10(UKgas), sd_y = prior,
#' sd_level =  prior, sd_slope =  prior, sd_seasonal =  prior)
#'
#' mcmc_out <- run_mcmc(model, n_iter = 5000)
#' summary(mcmc_out$theta)$stat
#' mcmc_out$theta[which.max(mcmc_out$posterior), ]
#' sqrt((fit <- StructTS(log10(UKgas), type = "BSM"))$coef)[c(4, 1:3)]
#'
bsm <- function(y, sd_y, sd_level, sd_slope, sd_seasonal,
  beta, xreg = NULL, slope = TRUE, seasonal = frequency(y) > 1,
  period = frequency(y), a1, P1) {

  check_y(y)
  n <- length(y)

  if (period == 1) {
    seasonal <- FALSE
  } else {
    if (missing(seasonal)) {
      seasonal <- TRUE
    }
  }

  #easier this way...
  notfixed <- c("level" = 1, "slope" = 1, "seasonal" = 1)

  npar_R <- sum(!c(missing(sd_level), missing(sd_slope),
      missing(sd_seasonal)) & c(TRUE, slope, seasonal))

  npar <- 1L + npar_R

  if(!missing(beta)) {
    npar <- npar + length(beta$init)
  }

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

  check_sd(sd_y$init, "y")

  if (missing(sd_level)) {
    notfixed[1] <- 0
    sd_level <- NULL
  } else {
    check_sd(sd_level$init, "level")
  }

  if (slope) {
    if (missing(sd_slope)) {
      notfixed[2] <- 0
      sd_slope <- NULL
    } else {
      check_sd(sd_slope$init, "slope")
    }
  } else sd_slope <- NULL

  if (seasonal) {
    if (missing(sd_seasonal)) {
      notfixed[3] <- 0
      sd_seasonal <- NULL
    } else {
      check_sd(sd_seasonal$init, "seasonal")
    }
    seasonal_names <- paste0("seasonal_", 1:(period - 1))
  } else {
    seasonal_names <- NULL
    sd_seasonal <- NULL
  }


  m <- as.integer(1L + slope + seasonal * (period - 1))

  if (missing(a1)) {
    a1 <- numeric(m)
  } else {
    if (length(a1) != m) {
      stop("Argument a1 must be a vector of length ", m)
    }
  }
  if (missing(P1)) {
    P1 <- diag(1e3, m)
  } else {
    if (is.null(dim(P1)) && length(P1) == 1L) {
      P1 <- matrix(P1)
    }
    if (!identical(dim(P1), c(m, m))) {
      stop("Argument P1 must be m x m matrix, where m = ", m)
    }
  }

  if (slope) {
    state_names <- c("level", "slope", seasonal_names)
  } else {
    state_names <- c("level", seasonal_names)
  }

  Z <- matrix(0, m, 1)
  Z[1, 1] <- 1
  if (seasonal) {
    Z[2 + slope, 1] <- 1
  }

  H <- matrix(sd_y$init)

  T <- matrix(0, m, m)
  T[1, 1] <- 1
  if (slope) {
    T[1:2, 2] <- 1
  }
  if (seasonal) {
    T[(2 + slope), (2 + slope):m] <- -1
    diag(T[(2 + slope + 1):m, (2 + slope):(m - 1)]) <- 1
  }

  R <- matrix(0, m, max(1, npar_R))

  if (notfixed[1]) {
    R[1, 1] <- sd_level$init
  }
  if (slope && notfixed[2]) {
    R[2, 1 + notfixed[1]] <- sd_slope$init
  }
  if (seasonal && notfixed[3]) {
    R[2 + slope, 1 + notfixed[1] + (slope && notfixed[2])] <- sd_seasonal$init
  }

  dim(H) <- 1
  dim(T) <- c(m, m, 1)
  dim(R) <- c(m, ncol(R), 1)

  names(a1) <- rownames(P1) <- colnames(P1) <- rownames(Z) <-
    rownames(T) <- colnames(T) <- rownames(R) <- state_names

  names_ind <- c(TRUE, notfixed & c(TRUE, slope, seasonal))

  priors <- list(sd_y, sd_level, sd_slope, sd_seasonal, beta)
  priors <- priors[!sapply(priors, is.null)]
  names(priors) <-
    c(c("sd_y", "sd_level", "sd_slope", "sd_seasonal")[names_ind], names(coefs))

  structure(list(y = as.ts(y), Z = Z, H = H, T = T, R = R,
    a1 = a1, P1 = P1, xreg = xreg, coefs = coefs,
    slope = slope, seasonal = seasonal, period = period,
    fixed = as.integer(!notfixed), priors = priors), class = "bsm")
}

#' @method logLik bsm
#' @rdname logLik
#' @export
logLik.bsm <- function(object, ...) {
  bsm_loglik(object)
}

#' @method kfilter bsm
#' @rdname kfilter
#' @export
kfilter.bsm <- function(object, ...) {

  out <- bsm_filter(object)

  colnames(out$at) <- colnames(out$att) <- colnames(out$Pt) <-
    colnames(out$Ptt) <- rownames(out$Pt) <-
    rownames(out$Ptt) <- names(object$a1)
  out$at <- ts(out$at, start = start(object$y), frequency = object$period)
  out$att <- ts(out$att, start = start(object$y), frequency = object$period)
  out
}
#' @method fast_smoother bsm
#' @export
fast_smoother.bsm <- function(object, ...) {

  out <- bsm_fast_smoother(object)

  colnames(out) <- names(object$a1)
  ts(out, start = start(object$y), frequency = object$period)
}
#' @method sim_smoother bsm
#' @export
sim_smoother.bsm <- function(object, nsim = 1, seed = sample(.Machine$integer.max, size = 1), ...) {

  out <- bsm_sim_smoother(object, nsim, seed)

  rownames(out) <- names(object$a1)
  aperm(out, c(2, 1, 3))
}

#' @method smoother bsm
#' @export
smoother.bsm <- function(object, ...) {

  out <- bsm_smoother(object)

  colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <- names(object$a1)
  out$alphahat <- ts(out$alphahat, start = start(object$y),
    frequency = object$period)
  out
}

#' @method run_mcmc bsm
#' @rdname run_mcmc_g
#' @param log_space Generate proposals for standard deviations in log-space. Default is \code{FALSE}.
#' @param n_threads Number of threads for state simulation.
#' @inheritParams run_mcmc.gssm
#' @export
run_mcmc.bsm <- function(object, n_iter, sim_states = TRUE, type = "full",
  n_burnin = floor(n_iter/2), n_thin = 1, gamma = 2/3,
  target_acceptance = 0.234, S, end_adaptive_phase = TRUE,
  log_space = FALSE, n_threads = 1,
  seed = sample(.Machine$integer.max, size = 1), ...) {

  type <- match.arg(type, c("full", "summary"))

  if(log_space) {
    stop("log_space = TRUE is under construction.")
  }

  if (missing(S)) {
    S <- diag(0.1 * pmax(0.1, abs(sapply(object$priors, "[[", "init"))), length(object$priors))
  }

  priors <- combine_priors(object$priors)

  out <- switch(type,
    full = {
      out <- bsm_run_mcmc(object,
        priors$prior_type, priors$params, n_iter,
        sim_states, n_burnin, n_thin, gamma, target_acceptance, S, seed, log_space, end_adaptive_phase)

      out$alpha <- aperm(out$alpha, c(2, 1, 3))
      colnames(out$alpha) <- names(object$a1)
      out
    },
    summary = {
      out <- bsm_run_mcmc_summary(object,
        priors$prior_type, priors$params, n_iter,
        n_burnin, n_thin, gamma, target_acceptance, S, seed,
        log_space, end_adaptive_phase)

      colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <- names(object$a1)
      out$alphahat <- ts(out$alphahat, start = start(object$y),
        frequency = object$period)
      out
    })
  # if (log_space && (ncol(out$theta) - length(object$coefs)) > 0) {
  #   out$theta[, 1:n_sd_par] <- exp(out$theta[, 1:n_sd_par])
  # }
  names_ind <- c(TRUE, !object$fixed & c(TRUE, object$slope, object$seasonal))
  colnames(out$theta) <- rownames(out$S) <- colnames(out$S) <-
    c(c("sd_y", "sd_level", "sd_slope", "sd_seasonal")[names_ind],
      colnames(object$xreg))
  out$theta <- mcmc(out$theta, start = n_burnin + 1, thin = n_thin)
  out$call <- match.call()
  class(out) <- "mcmc_output"
  out
}

#' @method predict bsm
#' @rdname predict
#' @export
#' @examples
#' require("graphics")
#' y <- log10(JohnsonJohnson)
#' prior <- uniform(0.01, 0, 1)
#' model <- bsm(y, sd_y = prior, sd_level = prior,
#'   sd_slope = prior, sd_seasonal = prior)
#'
#' pred1 <- predict(model, n_iter = 5000, n_ahead = 8)
#' pred2 <- predict(StructTS(y, type = "BSM"), n.ahead = 8)
#'
#' ts.plot(pred1$mean, pred1$intervals[,-2], pred2$pred +
#' cbind(0, -qnorm(0.95) * pred2$se, qnorm(0.95) * pred2$se),
#'   col = c(1, 1, 1, 2, 2, 2))
#'
predict.bsm <- function(object, n_iter, newdata = NULL,
  n_ahead = 1, interval = "response", probs = c(0.05, 0.95),
  method = "quantile", return_MCSE = TRUE, nsim_states = 1,
  n_burnin = floor(n_iter/2),
  n_thin = 1, gamma = 2/3, target_acceptance = 0.234, S,
  seed = sample(.Machine$integer.max, size = 1), ...) {

  interval <- pmatch(interval, c("mean", "response"))
  method <- match.arg(method, c("parametric", "quantile"))

  if (missing(S)) {
    S <- diag(0.1 * pmax(0.1, abs(sapply(object$priors, "[[", "init"))), length(object$priors))
  }

  priors <- combine_priors(object$priors)


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
  probs <- sort(unique(c(probs, 0.5)))
  if (method == "parametric") {
    out <- bsm_predict(object,
      priors$prior_types, priors$params, n_iter,
      n_burnin, n_thin, gamma, target_acceptance, S, n_ahead, interval,
      probs, seed, FALSE)

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

      pred <- list(y = y_orig, mean = ts(colMeans(out$y_mean), end = endtime, frequency = object$period),
        intervals = ts(out$intervals, end = endtime, frequency = object$period,
          names = paste0(100 * probs, "%")),
        MCSE = ts(ses, end = endtime, frequency = object$period,
          names = paste0(100 * probs, "%")))
    } else {
      pred <- list(y = y_orig, mean = ts(colMeans(out$y_mean), end = endtime, frequency = object$period),
        intervals = ts(out$intervals, end = endtime, frequency = object$period,
          names = paste0(100 * probs, "%")))
    }
  } else {
    out <- bsm_predict2(object, priors$prior_types, priors$params, n_iter, nsim_states,
      n_burnin, n_thin, gamma, target_acceptance, S, n_ahead, interval,
      seed, FALSE)

    pred <- list(y = y_orig, mean = ts(rowMeans(out), end = endtime, frequency = object$period),
      intervals = ts(t(apply(out, 1, quantile, probs, type = 8)), end = endtime, frequency = object$period,
        names = paste0(100 * probs, "%")))
  }
  class(pred) <- "predict_bssm"
  pred
}

#' @method particle_filter bsm
#' @rdname particle_filter
#' @export
particle_filter.bsm <- function(object, nsim,
  seed = sample(.Machine$integer.max, size = 1), ...) {

  out <- bsm_particle_filter(object, nsim, seed)

  rownames(out$alpha) <- names(object$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}

#' @method particle_smoother bsm
#' @rdname particle_smoother
#' @export
particle_smoother.bsm <- function(object, nsim, method = "fs",
  seed = sample(.Machine$integer.max, size = 1), ...) {

  method <- match.arg(method, c("fs", "fbs"))
  out <- bsm_particle_smoother(object, nsim, seed, method == "fs")

  rownames(out$alpha) <- names(object$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}

#' @method particle_simulate bsm
#' @rdname particle_simulate
#' @export
particle_simulate.bsm <- function(object, nsim, nsim_store = 1,
  seed = sample(.Machine$integer.max, size = 1), ...) {

  out <- bsm_backward_simulate(object, nsim, seed, nsim_store)

  rownames(out$alpha) <- names(object$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}
