#' Basic Structural (Time Series) Model
#'
#' Constructs a basic structural model with local level or local trend component
#' and seasonal component.
#'
#' @param y Vector or a \code{\link{ts}} object of observations.
#' @param sd_y Standard error of observation equation. Used as an initial
#' value in MCMC.
#' @param sd_level Standard error of level equation. Used as an initial
#' value in MCMC. If missing, \code{sd_level} is fixed to zero.
#' @param sd_slope Standard error of slope equation. Used as an initial
#' value in MCMC. If missing, \code{sd_slope} is fixed to zero.
#' Ignored if \code{slope = FALSE}.
#' @param sd_seasonal Standard error of seasonal equation. Used as an initial
#' value in MCMC. If missing, \code{sd_seasonal} is fixed to zero.
#' @param xreg Matrix containing covariates.
#' @param beta Regression coefficients. Used as an initial
#' value in MCMC. Defaults to vector of zeros.
#' @param period Length of the seasonal component i.e. the number of
#' observations per season. Default is \code{frequency(y)}.
#' @param slope Should the model contain the slope term. Default is \code{TRUE}.
#' If \code{FALSE}, local level component is used instead of local trend component.
#' @param seasonal Should the model contain seasonal term. Default is
#' \code{frequency(y) > 1}.
#' @param a1 Prior means for the initial states (level, slope, seasonals).
#' Defaults to vector of zeros.
#' @param P1 Prior covariance for the initial states (level, slope, seasonals).
#' Default is diagonal matrix with 1e5 on the diagonal.
#' @param lower_prior,upper_prior Lower and upper bounds for the uniform prior
#' on standard deviations (sd_y, sd_level, sd_slope, sd_seasonal) and regression
#' coefficients. Defaults to zero for lower bound and and \code{sd(y)} for
#' upper bound of standard deviations and (-Inf, Inf) for regression
#' coefficients.
#' @return Object of class \code{bstsm}.
#' @export
bsm <- function(y, sd_y = 1, sd_level, sd_slope, sd_seasonal, xreg = NULL, beta = NULL,
  period = frequency(y), slope = TRUE, seasonal = frequency(y) > 1, a1, P1,
  lower_prior, upper_prior) {


  if (!is.null(dim(y)[2]) && dim(y)[2] > 1) {
    stop("Argument y must a univariate time series. ")
  }

  if (length(sd_y) != 1) {
    stop("Argument sd_y must be of length one. ")
  }

  if (period == 1) {
    seasonal <- FALSE
  } else {
    if (missing(seasonal)) {
      seasonal <- TRUE
    }
  }

  fixed <- c("level" = NA, "slope" = NA, "seasonal" = NA)

  if (missing(sd_level)) {
    fixed[1] <- 0
    sd_level <- 0
  } else {
    if (length(sd_level) != 1) {
      stop("Argument sd_level must be of length one. ")
    }
  }

  if (slope) {
    if (missing(sd_slope)) {
      fixed[2] <- 0
      sd_slope <- 0
    } else {
      if (length(sd_slope) != 1) {
        stop("Argument sd_slope must be of length one. ")
      }
    }
  } else sd_slope <- 0

  if (seasonal) {
    if (missing(sd_seasonal)) {
      fixed[3] <- 0
      sd_seasonal <- 0
    } else {
      if (length(sd_seasonal) != 1) {
        stop("Argument sd_seasonal must be of length one. ")
      }
    }
    seasonal_names <- paste0("seasonal_", 1:(period - 1))
  } else {
    seasonal_names <- NULL
    sd_seasonal <- 0
  }

  n <- length(y)
  m <- as.integer(1L + slope + seasonal * (period - 1))

  if (missing(a1)) {
    a1 <- numeric(m)
  } else {
    if (length(a1) != m) {
      stop("Argument a1 must be a vector of length ", m)
    }
  }
  if (missing(P1)) {
    P1 <- diag(1e5, m)
  } else {
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

  H <- matrix(sd_y)

  T <- matrix(0, m, m)
  T[1, 1] <- 1
  if (slope) {
    T[1:2, 2] <- 1
  }
  if (seasonal) {
    T[(2 + slope), (2 + slope):m] <- -1
    diag(T[(2 + slope + 1):m, (2 + slope):(m - 1)]) <- 1
  }
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
    if (missing(beta)) {
      beta <- numeric(ncol(xreg))
    }
    names(beta) <- colnames(xreg)
  }

  npar_R <- sum(is.na(fixed) & c(TRUE, slope, seasonal))

  if (missing(lower_prior)) {
    lower_prior <- c(rep(0, 1 + npar_R), rep(-Inf, length(beta)))
  }
  if (missing(upper_prior)) {
    upper_prior <- c(rep(2 * sd(y, na.rm = TRUE), 1 + npar_R), rep(Inf, length(beta)))
    autoprior <- TRUE
  } else autoprior <- FALSE

  if (min(lower_prior[1:(1 + npar_R)], upper_prior[1:(1 + npar_R)]) < 0) {
    stop("Negative value in prior boundaries for standard deviations. ")
  }

  if (autoprior) {
    upper_prior[1] <- max(upper_prior[1], 2 * sd_y)
  } else {
    if (sd_y >  upper_prior[1]) {
      stop("Initial value for the sd_y is larger than the upper bound of
          the prior distribution. ")
    }
  }

  R <- matrix(0, m, max(1, npar_R))

  if (is.na(fixed[1])) {
    R[1, 1] <- sd_level
    if (autoprior) {
      upper_prior[2] <- max(upper_prior[2], 2 * sd_level)
    } else {
      if (sd_level > upper_prior[2]) {
        stop("Initial value for the sd_level is larger than the upper bound of
          the prior distribution. ")
      }
    }
  }
  if (slope && is.na(fixed[2])) {
    R[2, 1 + is.na(fixed[1])] <- sd_slope
    if (autoprior) {
      upper_prior[2 + is.na(fixed[1])] <-
        max(upper_prior[2 + is.na(fixed[1])], 2 * sd_slope)
    } else {
      if (sd_slope > upper_prior[2 + is.na(fixed[1])]) {
        stop("Initial value for the sd_slope is larger than the upper bound of
          the prior distribution. ")
      }
    }
  }
  if (seasonal && is.na(fixed[3])) {
    R[2 + slope, 1 + is.na(fixed[1]) + (slope && is.na(fixed[2]))] <- sd_seasonal
    if (autoprior) {
      upper_prior[2 + is.na(fixed[1]) + (slope && is.na(fixed[2]))] <-
        max(upper_prior[2 + is.na(fixed[1]) + (slope && is.na(fixed[2]))] ,
          2 * sd_seasonal)
    } else {
      if (sd_seasonal >  upper_prior[2 + is.na(fixed[1]) + (slope && is.na(fixed[2]))]) {
        stop("Initial value for the sd_seasonal is larger than the upper bound of
          the prior distribution. ")
      }
    }
  }

  dim(H) <- 1
  dim(T) <- c(m, m, 1)
  dim(R) <- c(m, ncol(R), 1)

  names(a1) <- rownames(P1) <- colnames(P1) <- rownames(Z) <-
    rownames(T) <- colnames(T) <- rownames(R) <- state_names

  names_ind <- c(TRUE, is.na(fixed) & c(TRUE, slope, seasonal))
  names(lower_prior) <- names(upper_prior) <-
    c(c("sd_y", "sd_level", "sd_slope", "sd_seasonal")[names_ind], names(beta))

  structure(list(y = as.ts(y), Z = Z, H = H, T = T, R = R,
    a1 = a1, P1 = P1, xreg = xreg, beta = beta,
    slope = slope, seasonal = seasonal, period = period, fixed = !is.na(fixed),
    lower_prior = lower_prior, upper_prior = upper_prior), class = "bstsm")
}

#' @method logLik bstsm
#' @rdname logLik
#' @export
logLik.bstsm <- function(object, ...) {
  bstsm_loglik(object$y, object$Z, object$H, object$T, object$R, object$a1,
    object$P1, object$slope, object$seasonal, object$fixed, object$xreg, object$beta)
}

#' @method kfilter bstsm
#' @rdname kfilter
#' @export
kfilter.bstsm <- function(object, ...) {

  out <- bstsm_filter(object$y, object$Z, object$H, object$T, object$R,
    object$a1, object$P1, object$slope, object$seasonal, object$fixed,
    object$xreg, object$beta)

  colnames(out$at) <- colnames(out$att) <- colnames(out$Pt) <-
    colnames(out$Ptt) <- rownames(out$Pt) <-
    rownames(out$Ptt) <- names(object$a1)
  out$at <- ts(out$at, start = start(object$y), frequency = object$period)
  out$att <- ts(out$att, start = start(object$y), frequency = object$period)
  out
}
#' @method fast_smoother bstsm
#' @rdname smoother
#' @export
fast_smoother.bstsm <- function(object, ...) {

  out <- bstsm_fast_smoother(object$y, object$Z, object$H, object$T,
    object$R, object$a1, object$P1, object$slope, object$seasonal, object$fixed,
    object$xreg, object$beta)

  colnames(out) <- names(object$a1)
  ts(out, start = start(object$y), frequency = object$period)
}
#' @method sim_smoother bstsm
#' @rdname smoother
#' @export
sim_smoother.bstsm <- function(object, nsim = 1, seed = sample(.Machine$integer.max, size = 1), ...) {

  out <- bstsm_sim_smoother(object$y, object$Z, object$H, object$T, object$R,
    object$a1, object$P1, nsim, object$slope, object$seasonal, object$fixed,
    object$xreg, object$beta, seed)

  rownames(out) <- names(object$a1)
  aperm(out, c(2, 1, 3))
}

#' @method smoother bstsm
#' @rdname smoother
#' @export
smoother.bstsm <- function(object, ...) {

  out <- bstsm_smoother(object$y, object$Z, object$H, object$T, object$R,
    object$a1, object$P1, object$slope, object$seasonal, object$fixed,
    object$xreg, object$beta)

  colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <- names(object$a1)
  out$alphahat <- ts(out$alphahat, start = start(object$y),
    frequency = object$period)
  out
}

#' @method run_mcmc bstsm
#' @rdname run_mcmc
#' @export
#' @examples
#' init_sd <- 0.1 * sd(log10(UKgas))
#' model <- bsm(log10(UKgas), sd_y = init_sd, sd_level = init_sd,
#'   sd_slope = init_sd, sd_seasonal = init_sd)
#' set.seed(123)
#' mcmc_out <- run_mcmc(model, n_iter = 5000)
#' names(mcmc_out)
#' mcmc_out$acceptance_rate
#' plot(mcmc_out$theta)
#' summary(mcmc_out$theta)
#' ts.plot(log10(UKgas), rowMeans(mcmc_out$alpha[, "level", ]), col = 1:2)
#' pred <- predict(model, n_iter = 5000, n_ahead = 8, S = mcmc_out$S)
#' ts.plot(pred$y, pred$mean, pred$interval, col = c(1, 2, 2, 2),
#'   lty = c(1, 1, 2, 2))
run_mcmc.bstsm <- function(object, n_iter, type = "full", lower_prior, upper_prior,
  nsim_states = 1, n_burnin = floor(n_iter/2), n_thin = 1, gamma = 2/3,
  target_acceptance = 0.234, S, seed = sample(.Machine$integer.max, size = 1),
  log_space = FALSE, ...) {

  type <- match.arg(type, c("full", "parameters", "summary"))

  if (missing(lower_prior)) {
    lower_prior <- object$lower_prior
  }
  if (missing(upper_prior)) {
    upper_prior <- object$upper_prior
  }
  n_sd_par <- length(lower_prior) - ncol(object$xreg)

  if (log_space && n_sd_par > 0) {
    lower_prior[1:(length(lower_prior) - ncol(object$xreg))] <-
      log(lower_prior[1:(length(lower_prior) - ncol(object$xreg))])
    upper_prior[1:(length(lower_prior) - ncol(object$xreg))] <-
      log(upper_prior[1:(length(lower_prior) - ncol(object$xreg))])
  }


  if (missing(S)) {
    sd_init <- sd(object$y, na.rm = TRUE)
    if (log_space) {
      sd_init <- abs(log(sd_init))
    }
    S <- diag(pmin(c(rep(0.1 * sd_init, length.out = n_sd_par), pmax(1, object$beta)),
      abs(upper_prior - lower_prior)), length(lower_prior))
  }

  out <- switch(type,
    full = {
      out <- bstsm_mcmc_full(object$y, object$Z, object$H, object$T, object$R,
        object$a1, object$P1, lower_prior, upper_prior, n_iter,
        nsim_states, n_burnin, n_thin, gamma, target_acceptance, S, object$slope,
        object$seasonal, object$fixed, object$xreg, object$beta, seed, log_space)

      out$alpha <- aperm(out$alpha, c(2, 1, 3))
      colnames(out$alpha) <- names(object$a1)
      out
    },
    parameters = {
      bstsm_mcmc_param(object$y, object$Z, object$H, object$T, object$R,
        object$a1, object$P1, lower_prior, upper_prior, n_iter,
        n_burnin, n_thin, gamma, target_acceptance, S, object$slope,
        object$seasonal, object$fixed, object$xreg, object$beta, seed, log_space)

    },
    summary = {
      out <- bstsm_mcmc_summary(object$y, object$Z, object$H, object$T, object$R,
        object$a1, object$P1, lower_prior, upper_prior, n_iter,
        n_burnin, n_thin, gamma, target_acceptance, S, object$slope,
        object$seasonal, object$fixed, object$xreg, object$beta, seed, log_space)

      colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <- names(object$a1)
      out$alphahat <- ts(out$alphahat, start = start(object$y),
        frequency = object$period)
      out
    })
  if (log_space && n_sd_par > 0) {
    out$theta[, 1:n_sd_par] <- exp(out$theta[, 1:n_sd_par])
  }
  out$S <- matrix(out$S, length(lower_prior), length(lower_prior))
  names_ind <- c(TRUE, !object$fixed & c(TRUE, object$slope, object$seasonal))
  colnames(out$theta) <- rownames(out$S) <- colnames(out$S) <-
    c(c("sd_y", "sd_level", "sd_slope", "sd_seasonal")[names_ind],
      colnames(object$xreg))
  out$theta <- mcmc(out$theta, start = n_burnin + 1, thin = n_thin)
  out
}

#' @method predict bstsm
#' @rdname predict
#' @export
predict.bstsm <- function(object, n_iter, lower_prior, upper_prior, newdata = NULL,
  n_ahead = 1, interval = "response", probs = c(0.05, 0.95),
  method = "parametric", return_MCSE = TRUE, nsim_states = 1, n_burnin = floor(n_iter/2),
  n_thin = 1, gamma = 2/3, target_acceptance = 0.234, S,
  seed = sample(.Machine$integer.max, size = 1), log_space = FALSE, ...) {

  interval <- pmatch(interval, c("mean", "response"))
  method <- match.arg(method, c("parametric", "quantile"))

  if (missing(lower_prior)) {
    lower_prior <- object$lower_prior
  }
  if (missing(upper_prior)) {
    upper_prior <- object$upper_prior
  }
  n_sd_par <- length(lower_prior) - ncol(object$xreg)

  if (log_space && n_sd_par > 0) {
    lower_prior[1:(length(lower_prior) - ncol(object$xreg))] <-
      log(lower_prior[1:(length(lower_prior) - ncol(object$xreg))])
    upper_prior[1:(length(lower_prior) - ncol(object$xreg))] <-
      log(upper_prior[1:(length(lower_prior) - ncol(object$xreg))])
  }

  if (missing(S)) {
    sd_init <- sd(object$y, na.rm = TRUE)
    if (log_space) {
      sd_init <- abs(log(sd_init))
    }
    S <- diag(pmin(c(rep(0.1 * sd_init, length.out = n_sd_par), pmax(1, object$beta)),
      abs(upper_prior - lower_prior)), length(lower_prior))
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
  probs <- sort(unique(c(probs, 0.5)))
  if (method == "parametric") {
    out <- bstsm_predict(y, object$Z, object$H, object$T, object$R,
      object$a1, object$P1, lower_prior, upper_prior, n_iter,
      n_burnin, n_thin, gamma, target_acceptance, S, n_ahead, interval,
      object$slope, object$seasonal, object$fixed, object$xreg, object$beta,
      probs, seed, log_space)

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
          names = paste0(100 * probs, "%")),
        MCSE = ts(ses, end = endtime, frequency = object$period,
          names = paste0(100 * probs, "%")))
    } else {
      pred <- list(y = object$y, mean = ts(colMeans(out$y_mean), end = endtime, frequency = object$period),
        intervals = ts(out$intervals, end = endtime, frequency = object$period,
          names = paste0(100 * probs, "%")))
    }
  } else {
    out <- bstsm_predict2(y, object$Z, object$H, object$T, object$R,
      object$a1, object$P1, lower_prior, upper_prior, n_iter, nsim_states,
      n_burnin, n_thin, gamma, target_acceptance, S, n_ahead, interval,
      object$slope, object$seasonal, object$fixed, object$xreg, object$beta, seed, log_space)

    pred <- list(y = object$y, mean = ts(rowMeans(out), end = endtime, frequency = object$period),
      intervals = ts(t(apply(out, 1, quantile, probs, type = 9)), end = endtime, frequency = object$period,
        names = paste0(100 * probs, "%")))
  }
  class(pred) <- "predict_bssm"
  pred
}
