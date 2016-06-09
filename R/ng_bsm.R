#' Non-Gaussian Basic Structural (Time Series) Model
#'
#' Constructs a non-Gaussian basic structural model with local level or
#' local trend component, a seasonal component, and regression component
#' (or subset of these components).
#'
#' @param y Vector or a \code{\link{ts}} object of observations.
#' @param sd_level Standard error of level equation. Used as an initial
#' value in MCMC. If missing, \code{sd_level} is fixed to zero.
#' @param sd_slope Standard error of slope equation. Used as an initial
#' value in MCMC. If missing, \code{sd_slope} is fixed to zero.
#' Ignored if \code{slope = FALSE}.
#' @param sd_seasonal Standard error of seasonal equation. Used as an initial
#' value in MCMC. If missing, \code{sd_seasonal} is fixed to zero.
#' @param sd_noise Standard error of additional noise term. Used as an initial
#' value in MCMC. If missing, additional noise term is omitted from the model.
#' @param distribution distribution of the observation. Possible choices are
#' \code{"poisson"} and \code{"binomial"}.
#' @param phi Additional parameter vector relating to the non-Gaussian distribution.
#' For Poisson distribution, this corresponds to offset term. For binomial, this
#' is the number of trials.
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
#' on standard deviations (sd_level, sd_slope, sd_seasonal) and regression
#' coefficients. Defaults to zero for lower bound and and
#' \code{sd(init_signal)} for upper bound of standard deviations and
#' (-1e4, 1e4) for regression coefficients.
#' @return Object of class \code{ng_bstsm}.
#' @export
#' @examples
#' model <- ng_bsm(Seatbelts[, "VanKilled"], distribution = "poisson",
#'   sd_level = 0.01, sd_seasonal = 0.01, slope = FALSE,
#'   xreg = Seatbelts[, "law"])
#' \dontrun{
#' set.seed(123)
#' mcmc_out <- run_mcmc(model, n_iter = 5000)
#' mcmc_out$acceptance_rate
#' plot(mcmc_out$theta)
#' summary(mcmc_out$theta)
#'
#' require("ggplot2")
#' ggplot(as.data.frame(mcmc_out$theta[,1:2]), aes(x = sd_level, y = sd_seasonal)) +
#'   geom_point() + stat_density2d(aes(fill = ..level.., alpha = ..level..),
#'   geom = "polygon") + scale_fill_continuous(low = "green",high = "blue") +
#'   guides(alpha = "none")
#'
#' pred <- predict(model, n_iter = 5000, nsim_states = 25, n_ahead = 36,
#'   probs = seq(0.05, 0.95, by = 0.05), newdata = matrix(1, 36, 1),
#'   newphi = rep(1, 36))
#' autoplot(pred)
#' }
ng_bsm <- function(y, sd_level, sd_slope, sd_seasonal, sd_noise,
  distribution, phi = 1, xreg = NULL, beta = NULL,
  period = frequency(y), slope = TRUE, seasonal = frequency(y) > 1, a1, P1,
  lower_prior, upper_prior) {

  if (!is.null(dim(y)[2]) && dim(y)[2] > 1) {
    stop("Argument y must a univariate time series. ")
  }

  if (period == 1) {
    seasonal <- FALSE
  } else {
    if (missing(seasonal)) {
      seasonal <- TRUE
    }
  }

  fixed <- c("level" = NA, "slope" = NA, "seasonal" = NA)

  if (missing(sd_noise)) {
    noise <- FALSE
  } else {
    if (length(sd_noise) != 1) {
      stop("Argument sd_noise must be of length one. ")
    }
    if (sd_noise < 0) {
      stop("Argument sd_noise must be non-negative. ")
    }
    noise <- TRUE
  }


  if (missing(sd_level)) {
    fixed[1] <- 0
    sd_level <- 0
  } else {
    if (length(sd_level) != 1) {
      stop("Argument sd_level must be of length one. ")
    }
    if (sd_level < 0) {
      stop("Argument sd_level must be non-negative. ")
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
      if (sd_slope < 0) {
        stop("Argument sd_slope must be non-negative. ")
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
      if (sd_seasonal < 0) {
        stop("Argument sd_seasonal must be non-negative. ")
      }
    }
    seasonal_names <- paste0("seasonal_", 1:(period - 1))
  } else {
    seasonal_names <- NULL
    sd_seasonal <- 0
  }

  n <- length(y)
  m <- as.integer(1L + slope + seasonal * (period - 1) + noise)

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


  Z <- matrix(0, m,1)
  Z[1, 1] <- 1

  if (seasonal) {
    Z[2 + slope,1] <- 1
  }

  T <- matrix(0, m, m)
  T[1, 1] <- 1

  if (slope) {
    T[1:2, 2] <- 1
  }

  if (seasonal) {
    T[(2 + slope), (2 + slope):(m - noise)] <- -1
    diag(T[(2 + slope + 1):(m - noise), (2 + slope):(m - 1 - noise)]) <- 1
  }

  if (is.null(xreg)) {
    xreg <- matrix(0, 0, 0)
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

  npar_R <- sum(is.na(fixed) & c(TRUE, slope, seasonal)) + noise

  if (!(length(phi) %in% c(1, n))) {
    stop("Argument phi must have length 1 or n. ")
  }

  if (length(phi) != n) {
    phi <- rep(phi, length.out = n)
  }

  distribution <- match.arg(distribution, c("poisson", "binomial",
    "negative binomial"))
  nb <- distribution == "negative binomial"
  init_signal <- initial_signal(y, phi, distribution)

  if (missing(lower_prior)) {
    lower_prior <- c(rep(0, npar_R), rep(-1e4, length(beta) + nb))
  }

  if (missing(upper_prior)) {
    autoprior <- TRUE
    sds <- sd(init_signal)
    if (distribution == "poisson") {
      sds <- 2 * sd(log(phi) + init_signal)
    } else {
      sds <- 2 * sd(init_signal)
    }
    upper_prior <- c(rep(sds, npar_R), rep(1e4, length(beta) + nb))
  } else autoprior <- FALSE

  if (min(lower_prior[1:(npar_R)], upper_prior[1:(npar_R)]) < 0) {
    stop("Negative value in prior boundaries for standard deviations. ")
  }
  # if (nb && min(lower_prior[length(lower_prior)],
  #   upper_prior[length(lower_prior)]) <= 0) {
  #   stop("Non-positive value in prior boundaries for dispersion parameter of negative binomial distribution. ")
  # }
  R <- matrix(0, m, max(1, npar_R))

  #level
  if (is.na(fixed[1])) {
    R[1, 1] <- sd_level
    if (autoprior) {
      upper_prior[1] <- max(upper_prior[1], 2 * sd_level)
    } else {
      if (sd_level > upper_prior[1]) {
        stop("Initial value for the sd_level is larger than the upper bound of
          the prior distribution. ")
      }
    }
  }

  #slope
  if (slope && is.na(fixed[2])) {
    R[2, 1 + is.na(fixed[1])] <- sd_slope
    if (autoprior) {
      upper_prior[1 + is.na(fixed[1])] <-
        max(upper_prior[1 + is.na(fixed[1])], 2 * sd_slope)
    } else {
      if (sd_slope > upper_prior[1 + is.na(fixed[1])]) {
        stop("Initial value for the sd_slope is larger than the upper bound of
          the prior distribution. ")
      }
    }
  }

  #seasonal
  if (seasonal && is.na(fixed[3])) {
    R[2 + slope, 1 + is.na(fixed[1]) + (slope && is.na(fixed[2]))] <- sd_seasonal
    if (autoprior) {
      upper_prior[1 + is.na(fixed[1]) + (slope && is.na(fixed[2]))] <-
        max(upper_prior[1 + is.na(fixed[1]) + (slope && is.na(fixed[2]))] ,
          2 * sd_seasonal)
    } else {
      if (sd_seasonal >  upper_prior[1 + is.na(fixed[1]) + (slope && is.na(fixed[2]))]) {
        stop("Initial value for the sd_seasonal is larger than the upper bound of
          the prior distribution. ")
      }
    }
  }
  #additional noise term
  if (noise) {
    P1[m, m] <- sd_noise^2
    Z[m] <- 1
    state_names <- c(state_names, "noise")

    R[m, 1 + is.na(fixed[1]) + (slope && is.na(fixed[2])) + (seasonal && is.na(fixed[3]))] <- sd_noise
    if (autoprior) {
      upper_prior[1 + is.na(fixed[1]) + (slope && is.na(fixed[2])) +
          (seasonal && is.na(fixed[3]))] <- max(upper_prior[1 + is.na(fixed[1]) +
              (slope && is.na(fixed[2])) + (seasonal && is.na(fixed[3]))], 2 * sd_noise)
    } else {
      if (sd_noise >  upper_prior[1 + is.na(fixed[1]) + (slope && is.na(fixed[2])) +
          (seasonal && is.na(fixed[3]))]) {
        stop("Initial value for the sd_noise is larger than the upper bound of
          the prior distribution. ")
      }
    }
  }
  dim(T) <- c(m, m, 1)
  dim(R) <- c(m, ncol(R), 1)

  names(a1) <- rownames(P1) <- colnames(P1) <- rownames(Z) <-
    rownames(T) <- colnames(T) <- rownames(R) <- state_names

  names_ind <- c(is.na(fixed) & c(TRUE, slope, seasonal), noise)

  names(lower_prior) <- names(upper_prior) <-
    c(c("sd_level", "sd_slope", "sd_seasonal", "sd_noise")[names_ind], names(beta), if(nb) "nb_dispersion")

  structure(list(y = as.ts(y), Z = Z, T = T, R = R,
    a1 = a1, P1 = P1, phi = phi, xreg = xreg, beta = beta,
    slope = slope, seasonal = seasonal, noise = noise, period = period, fixed = !is.na(fixed),
    lower_prior = lower_prior, upper_prior = upper_prior,
    distribution = distribution, init_signal = init_signal), class = "ng_bstsm")
}

#' @method logLik ng_bstsm
#' @rdname logLik
#' @export
logLik.ng_bstsm <- function(object, ...) {
  ng_bstsm_loglik(object$y, object$Z, object$T, object$R, object$a1,
    object$P1, object$phi, object$slope, object$seasonal, object$noise, object$fixed,
    object$xreg, object$beta,
    pmatch(object$distribution, c("poisson", "binomial", "negative binomial")),
    object$init_signal)
}

#' @method run_mcmc ng_bstsm
#' @rdname run_mcmc_ng
#' @param log_space Generate proposals for standard deviations in log-space. Default is \code{FALSE}.
#' @param n_store Number of samples to store from the simulation smoother per iteration.
#' Default is 1.
#' @param method Use \code{"standard"} MCMC or \code{"delayed acceptance"} approach.
#' @export
run_mcmc.ng_bstsm <- function(object, n_iter, nsim_states = 1,
  lower_prior, upper_prior, n_burnin = floor(n_iter/2),
  n_thin = 1, gamma = 2/3, target_acceptance = 0.234, S,
  seed = sample(.Machine$integer.max, size = 1), log_space = FALSE,
  n_store = 1, method = "delayed acceptance",  ...) {

  method <- match.arg(method, c("standard", "delayed acceptance"))

  if (missing(lower_prior)) {
    lower_prior <- object$lower_prior
  }
  if (missing(upper_prior)) {
    upper_prior <- object$upper_prior
  }
  nb <- object$distribution == "negative binomial"
  n_sd_par <- length(lower_prior) - ncol(object$xreg) - nb

  if (log_space && n_sd_par > 0) {
    lower_prior[1:(length(lower_prior) - ncol(object$xreg) - nb)] <-
      log(lower_prior[1:(length(lower_prior) - ncol(object$xreg) - nb)])
    upper_prior[1:(length(lower_prior) - ncol(object$xreg) - nb)] <-
      log(upper_prior[1:(length(lower_prior) - ncol(object$xreg) - nb)])
  }

  if (missing(S)) {
    sd_init <- sd(object$init_signal, na.rm = TRUE)
    if (log_space) {
      sd_init <- abs(log(sd_init))
    }
    S <- diag(pmin(c(rep(0.1 * sd_init, length.out = n_sd_par),
      pmax(1, abs(object$beta)), if(nb) 1),
      abs(upper_prior - lower_prior)), length(lower_prior))
  }

  out <- switch(method,
    standard = {
      out <- ng_bstsm_mcmc_full(object$y, object$Z, object$T, object$R,
        object$a1, object$P1, object$phi,
        pmatch(object$distribution, c("poisson", "binomial", "negative binomial")),
        lower_prior, upper_prior, n_iter,
        nsim_states, n_burnin, n_thin, gamma, target_acceptance, S, object$slope,
        object$seasonal, object$noise, object$fixed, object$xreg, object$beta,
        object$init_signal, 1, seed, log_space)

      out$alpha <- aperm(out$alpha, c(2, 1, 3))
      colnames(out$alpha) <- names(object$a1)
      out
    },
    "delayed acceptance" = {
      out <- ng_bstsm_mcmc_full(object$y, object$Z, object$T, object$R,
        object$a1, object$P1, object$phi,
        pmatch(object$distribution, c("poisson", "binomial", "negative binomial")),
        lower_prior, upper_prior, n_iter,
        nsim_states, n_burnin, n_thin, gamma, target_acceptance, S, object$slope,
        object$seasonal, object$noise, object$fixed, object$xreg, object$beta,
        object$init_signal, 2, seed, log_space)

      out$alpha <- aperm(out$alpha, c(2, 1, 3))
      colnames(out$alpha) <- names(object$a1)
      out
    })
  if (log_space && n_sd_par > 0) {
    out$theta[, 1:n_sd_par] <- exp(out$theta[, 1:n_sd_par])
  }
  if (nb) {
    out$theta[, ncol(out$theta)] <- exp(out$theta[, ncol(out$theta)])
  }
  out$S <- matrix(out$S, length(lower_prior), length(lower_prior))
  names_ind <-
    c(!object$fixed & c(TRUE, object$slope, object$seasonal), object$noise)
  colnames(out$theta) <- rownames(out$S) <- colnames(out$S) <-
    c(c("sd_level", "sd_slope", "sd_seasonal", "sd_noise")[names_ind],
      colnames(object$xreg), if (nb) "nb_dispersion")
  out$theta <- mcmc(out$theta, start = n_burnin + 1, thin = n_thin)
  out
}

#' @method predict ng_bstsm
#' @rdname predict.ngssm
#' @export
predict.ng_bstsm <- function(object, n_iter, nsim_states, lower_prior, upper_prior,
  newdata = NULL, n_ahead = 1, interval = "mean", probs = c(0.05, 0.95),
  n_burnin = floor(n_iter/2), n_thin = 1,
  gamma = 2/3, target_acceptance = 0.234, S,
  seed = sample(.Machine$integer.max, size = 1), newphi = NULL, log_space = FALSE, ...) {

  interval <- pmatch(interval, c("mean", "response"))

  if (missing(lower_prior)) {
    lower_prior <- object$lower_prior
  }
  if (missing(upper_prior)) {
    upper_prior <- object$upper_prior
  }
  nb <- object$distribution == "negative binomial"
  n_sd_par <- length(lower_prior) - ncol(object$xreg) - nb

  if (log_space && n_sd_par > 0) {
    lower_prior[1:(length(lower_prior) - ncol(object$xreg) - nb)] <-
      log(lower_prior[1:(length(lower_prior) - ncol(object$xreg) - nb)])
    upper_prior[1:(length(lower_prior) - ncol(object$xreg) - nb)] <-
      log(upper_prior[1:(length(lower_prior) - ncol(object$xreg) - nb)])
  }

  if (missing(S)) {
    sd_init <- sd(object$init_signal, na.rm = TRUE)
    if (log_space) {
      sd_init <- abs(log(sd_init))
    }
    S <- diag(pmin(c(rep(0.1 * sd_init, length.out = n_sd_par),
      pmax(1, abs(object$beta)), if(nb) 1),
      abs(upper_prior - lower_prior)), length(lower_prior))
  }


  endtime <- end(object$y) + c(0, n_ahead)
  y <- c(object$y, rep(NA, n_ahead))

  if (length(object$beta) > 0) {
    if (!is.null(newdata) && (nrow(newdata) != n_ahead ||
        ncol(newdata) != length(object$beta))) {
      stop("Model contains regression part but dimensions of newdata does not match with n_ahead and length of beta. ")
    }
    if (is.null(newdata)) {
      newdata <- matrix(0, n_ahead, length(object$beta))
    }
    object$xreg <- rbind(object$xreg, newdata)
  }
  if (nb) {
    phi <- c(object$phi, rep(object$phi[length(object$phi)], n_ahead))
  } else {
    if (is.null(newphi)) {
      phi <- c(object$phi, rep(1, n_ahead))
    } else {
      if (length(newphi) != n_ahead) {
        stop("Length of newphi is not equal to n_ahead. ")
      } else {
        phi <- c(object$phi, newphi)
      }
    }
  }
  probs <- sort(unique(c(probs, 0.5)))
  out <- ng_bstsm_predict2(y, object$Z, object$T, object$R,
    object$a1, object$P1, phi,
    pmatch(object$distribution, c("poisson", "binomial", "negative binomial")),
    lower_prior, upper_prior, n_iter,
    nsim_states, n_burnin, n_thin, gamma, target_acceptance, S, n_ahead, interval,
    object$slope, object$seasonal, object$noise, object$fixed, object$xreg, object$beta,
    c(object$init_signal, rep(log(0.1), n_ahead)), seed, log_space)

  if (interval == 1 && (object$distribution != "negative binomial")) {
    object$y <- object$y / object$phi
  }
  pred <- list(y = object$y, mean = ts(rowMeans(out), end = endtime, frequency = object$period),
    intervals = ts(t(apply(out, 1, quantile, probs, type = 8)), end = endtime, frequency = object$period,
      names = paste0(100 * probs, "%")))


  class(pred) <- "predict_bssm"
  pred

}
