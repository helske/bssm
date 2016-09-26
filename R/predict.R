

#' Predictions for Gaussian State Space Models
#'
#' Posterior intervals of future observations or their means
#' (success probabilities in binomial case) for Gaussian models. These are
#' computed using either the quantile method where the intervals are computed
#' as empirical quantiles the posterior sample, or parametric method by
#' Helske (2016).
#'
#' @param object Model object.
#' @param priors Priors for the unknown parameters.
#' @param n_ahead Number of steps ahead at which to predict.
#' @param interval Compute predictions on \code{"mean"} ("confidence interval") or
#' \code{"response"} ("prediction interval"). Defaults to \code{"response"}.
#' @param probs Desired quantiles. Defaults to \code{c(0.05, 0.95)}. Always includes median 0.5.
#' @param newdata Matrix containing the covariate values for the future time
#' points. Defaults to zero matrix of appropriate size.
#' @param method Either \code{"parametric"} (default) or \code{"quantile"}.
#' Only used for linear-Gaussian case.
#' @param return_MCSE For method \code{"parametric"}, if \code{TRUE}, the Monte Carlo
#' standard errors are also returned.
#' @param nsim_states Number of samples used in importance sampling.
#' @param newphi Vector of length \code{n_ahead} defining the future values of \eqn{\phi}.
#' Defaults to 1, expect for negative binomial distribution, where the initial
#' value is taken from \code{object$phi}.
#' @param ... Ignored.
#' @inheritParams run_mcmc.gssm
#' @return List containing the mean predictions, quantiles and Monte Carlo
#' standard errors .
#' @method predict gssm
#' @rdname predict
#' @export
predict.gssm <- function(object, n_iter, priors, newdata = NULL,
  n_ahead = 1, interval = "response",
  probs = c(0.05, 0.95), method = "quantile", return_MCSE = TRUE,
  n_burnin = floor(n_iter / 2), n_thin = 1,
  gamma = 2/3, target_acceptance = 0.234, S,
  seed = sample(.Machine$integer.max, size = 1), ...) {

  interval <- pmatch(interval, c("mean", "response"))
  method <- match.arg(method, c("parametric", "quantile"))

  Z_ind <- which(is.na(object$Z)) - 1L
  Z_n <- length(Z_ind)
  H_ind <- which(is.na(object$H)) - 1L
  H_n <- length(H_ind)
  T_ind <- which(is.na(object$T)) - 1L
  T_n <- length(T_ind)
  R_ind <- which(is.na(object$R)) - 1L
  R_n <- length(R_ind)
  
  if ((Z_n + H_n + T_n + R_n + length(object$coef)) == 0) {
    stop("nothing to estimate. ")
  }
  inits <- sapply(priors, "[[", "init")
  if(length(inits) != (Z_n + H_n + T_n + R_n + length(object$coef))) {
    stop("Number of unknown parameters is not equal to the number of priors.")
  }
  if(Z_n > 0) {
    object$Z[is.na(object$Z)] <- inits[1:Z_n]
  }
  if(H_n > 0) {
    object$H[is.na(object$H)] <- inits[Z_n + 1:H_n]
  }
  if(T_n > 0) {
    object$T[is.na(object$T)] <- inits[Z_n + H_n + 1:T_n]
  }
  if(R_n > 0) {
    object$R[is.na(object$R)] <- inits[Z_n + H_n + T_n + 1:R_n]
  }
  
  if (missing(S)) {
    S <- diag(0.1 * pmax(0.1, abs(inits)), length(inits))
  }
  priors <- combine_priors(priors)
  
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

      pred <- list(y = object$y, mean = ts(colMeans(out$y_mean), end = endtime, frequency = frequency(object$y)),
        intervals = ts(out$intervals, end = endtime, frequency = frequency(object$y),
          names = paste0(100*probs, "%")),
        MCSE = ts(ses, end = endtime, frequency = frequency(object$y),
          names = paste0(100*probs, "%")))
    } else {
      pred <- list(y = object$y, mean = ts(colMeans(out$y_mean), end = endtime, frequency = frequency(object$y)),
        intervals = ts(out$intervals, end = endtime, frequency = frequency(object$y),
          names = paste0(100 * probs, "%")))
    }
  } else {

    out <- gssm_predict2(object, priors$prior_types, priors$param, n_iter,
      n_burnin, n_thin, gamma, target_acceptance, S, n_ahead, interval,
      Z_ind, H_ind, T_ind, R_ind, seed)

    pred <- list(y = y_orig, mean = ts(rowMeans(out), end = endtime, frequency = frequency(object$y)),
      intervals = ts(t(apply(out, 1, quantile, probs, type = 8)), end = endtime, frequency = frequency(object$y),
        names = paste0(100 * probs, "%")))

  }
  class(pred) <- "predict_bssm"
  pred
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
  method = "quantile", return_MCSE = TRUE,
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
    out <- bsm_predict2(object, priors$prior_types, priors$params, n_iter,
      n_burnin, n_thin, gamma, target_acceptance, S, n_ahead, interval,
      seed, FALSE)

    pred <- list(y = y_orig, mean = ts(rowMeans(out), end = endtime, frequency = object$period),
      intervals = ts(t(apply(out, 1, quantile, probs, type = 8)), end = endtime, frequency = object$period,
        names = paste0(100 * probs, "%")))
  }
  class(pred) <- "predict_bssm"
  pred
}
#' @export
#' @rdname predict
predict.ngssm <- function(object, n_iter, nsim_states, priors,
  newdata = NULL,
  n_ahead = 1, interval = "mean",
  probs = c(0.05, 0.95),
  n_burnin = floor(n_iter / 2), n_thin = 1,
  gamma = 2/3, target_acceptance = 0.234, S,
  seed = sample(.Machine$integer.max, size = 1),  newphi = NULL, ...) {
  
  interval <- pmatch(interval, c("mean", "response"))
   Z_ind <- which(is.na(object$Z)) - 1L
  Z_n <- length(Z_ind)

  T_ind <- which(is.na(object$T)) - 1L
  T_n <- length(T_ind)
  R_ind <- which(is.na(object$R)) - 1L
  R_n <- length(R_ind)
  
  if ((Z_n + T_n + R_n + length(object$coef)) == 0) {
    stop("nothing to estimate. ")
  }
  inits <- sapply(priors, "[[", "init")
  if(length(inits) != (Z_n + T_n + R_n + length(object$coef) + nb)) {
    stop("Number of unknown parameters is not equal to the number of priors.")
  }
  if(Z_n > 0) {
    object$Z[is.na(object$Z)] <- inits[1:Z_n]
  }

  if(T_n > 0) {
    object$T[is.na(object$T)] <- inits[Z_n + 1:T_n]
  }
  if(R_n > 0) {
    object$R[is.na(object$R)] <- inits[Z_n + T_n + 1:R_n]
  }
  
  if (missing(S)) {
    S <- diag(0.1 * pmax(0.1, abs(inits)), length(inits))
  }
  
  nb <- object$distribution == "negative binomial"
  
  if (Z_n + T_n + R_n + nb == 0) {
    stop("nothing to estimate. ")
  }
  
  if (missing(S)) {
    S <- diag(Z_n  + T_n + R_n + nb)
  }
  if (nrow(S) == length(priors)) {
    stop("Number of unknown parameters is not equal to the number of priors.")
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
  
  if (is.null(newphi) || length(newphi) != n_ahead) {
    stop("newphi is missing or its length is not equal to n_ahead. ")
  }
  
  
  if (any(c(dim(object$Z)[3], dim(object$T)[3], dim(object$R)[3]) > 1)) {
    stop("Time-varying system matrices in prediction are not yet supported.")
  }
  object$phi <- c(object$phi, newphi)
  probs <- sort(unique(c(probs, 0.5)))
  init_signal <- initial_signal(object$y, object$phi, object$distribution)
  object$distribution <- pmatch(object$distribution,
    c("poisson", "binomial", "negative binomial"))
  priors <- combine_priors(priors)
  out <- ngssm_predict2(object, priors$prior_types, priors$params, n_iter,
    nsim_states, n_burnin, n_thin, gamma, target_acceptance, S, n_ahead, interval,
    Z_ind, T_ind, R_ind, init_signal, seed)
  
  pred <- list(y = object$y, mean = ts(rowMeans(out), end = endtime, frequency = frequency(object$y)),
    intervals = ts(t(apply(out, 1, quantile, probs, type = 8)), end = endtime, frequency = frequency(object$y),
      names = paste0(100 * probs, "%")))
  
  
  class(pred) <- "predict_bssm"
  pred
  
}
#' @method predict ng_bsm
#' @rdname predict
#' @export
#' @examples
#' data("poisson_series")
#' model <- ng_bsm(poisson_series, sd_level = halfnormal(0.1, 1),
#'   sd_slope=halfnormal(0.01, 0.1), distribution = "poisson")
#' pred <- predict(model, n_iter = 1e4, nsim = 10, n_ahead = 10,
#'   probs = seq(0.05,0.95, by = 0.05))
#' library("ggplot2")
#' autoplot(pred, median_color = "blue", mean_color = "red")
#'
predict.ng_bsm <- function(object, n_iter, nsim_states,
  newdata = NULL, n_ahead = 1, interval = "mean",
  probs = c(0.05, 0.95),
  n_burnin = floor(n_iter/2), n_thin = 1,
  gamma = 2/3, target_acceptance = 0.234, S,
  seed = sample(.Machine$integer.max, size = 1), newphi = NULL,  ...) {

  interval <- pmatch(interval, c("mean", "response"))

  nb <- object$distribution == "negative binomial"

  if (missing(S)) {
    S <- diag(0.1 * pmax(0.1, abs(sapply(object$priors, "[[", "init"))), length(object$priors))
  }

  priors <- combine_priors(object$priors)


  endtime <- end(object$y) + c(0, n_ahead)
  y_orig <- object$y
  phi_orig <- object$phi
  object$y <- c(object$y, rep(NA, n_ahead))

  if (length(object$coefs) > 0) {
    if (!is.null(newdata) && (nrow(newdata) != n_ahead ||
        ncol(newdata) != length(object$coefs))) {
      stop("Model contains regression part but dimensions of newdata does not match with n_ahead and length of beta. ")
    }
    if (is.null(newdata)) {
      newdata <- matrix(0, n_ahead, length(object$coefs))
    }
    object$xreg <- rbind(object$xreg, newdata)
  }
  if (nb) {
    object$phi <- c(object$phi, rep(object$phi[length(object$phi)], n_ahead))
  } else {
    if (is.null(newphi)) {
      object$phi <- c(object$phi, rep(1, n_ahead))
    } else {
      if (length(newphi) != n_ahead) {
        stop("Length of newphi is not equal to n_ahead. ")
      } else {
        object$phi <- c(object$phi, newphi)
      }
    }
  }
  probs <- sort(unique(c(probs, 0.5)))
  object$distribution <- pmatch(object$distribution, c("poisson", "binomial", "negative binomial"))
  out <- ng_bsm_predict2(object, priors$prior_types, priors$params, n_iter,
    nsim_states, n_burnin, n_thin, gamma, target_acceptance, S, n_ahead, interval,
    c(object$init_signal, rep(log(0.1), n_ahead)), seed, FALSE)

  if (interval == 1 && (object$distribution != "negative binomial")) {
    y_orig <- y_orig / phi_orig
  }
  pred <- list(y = y_orig, mean = ts(rowMeans(out), end = endtime, frequency = object$period),
    intervals = ts(t(apply(out, 1, quantile, probs, type = 8)), end = endtime, frequency = object$period,
      names = paste0(100 * probs, "%")))


  class(pred) <- "predict_bssm"
  pred

}
