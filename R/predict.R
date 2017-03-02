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
#' @param newu Vector of length \code{n_ahead} defining the future values of \eqn{u}.
#' Defaults to 1.
#' @param ... Ignored.
#' @return List containing the mean predictions, quantiles and Monte Carlo
#' standard errors .
#' @method predict mcmc_output
#' @rdname predict
#' @export
predict.mcmc_output <- function(object, future_model, type = "response",
  intervals = TRUE, probs = c(0.05, 0.95), return_MCSE = TRUE, 
  seed = sample(.Machine$integer.max, size = 1), ...) {
  
  type <- match.arg(type, c("mean", "response"))
  
  probs <- sort(unique(c(probs, 0.5)))
  n_ahead <- length(future_model$y)
  start_ts <- start(future_model$y)
  end_ts <- end(future_model$y)
  freq <- frequency(future_model$y)
  
  if (attr(object, "model_type") %in% c("gssm", "bsm")) {
    
    out <- gaussian_predict(future_model, probs,
      t(object$theta), object$alpha[nrow(object$alpha),,], object$counts, 
      type == "response", intervals, seed, 
      pmatch(attr(object, "model_type"), c("gssm", "bsm")))
    
    if (intervals) {
      
      if(return_MCSE) {
        ses <- matrix(0, n_ahead, length(probs))
        
        nsim <- nrow(out$mean_pred)
        
        for (i in 1:n_ahead) {
          for (j in 1:length(probs)) {
            pnorms <- pnorm(q = out$intervals[i, j], out$mean_pred[, i], out$sd_pred[, i])
            eff_n <-  effectiveSize(pnorms)
            ses[i, j] <- sqrt((sum((probs[j] - pnorms) ^ 2) / nsim) / eff_n) /
              sum(dnorm(x = out$intervals[i, j], out$mean_pred[, i], out$sd_pred[, i]) / nsim)
          }
        }
        
        pred <- list(mean = ts(colMeans(out$mean_pred), start = start_ts, end = end_ts, frequency = freq),
          intervals = ts(out$intervals, start = start_ts, end = end_ts, frequency = freq,
            names = paste0(100*probs, "%")),
          MCSE = ts(ses, start = start_ts, end = end_ts, frequency = freq,
            names = paste0(100*probs, "%")))
      } else {
        pred <- list(mean = ts(colMeans(out$mean_pred), start = start_ts, end = end_ts, frequency = freq),
          intervals = ts(out$intervals, start = start_ts, end = end_ts, frequency = freq,
            names = paste0(100 * probs, "%")))
      }
    } else {
      pred <- out[[1]]
    }
  } else {
    if(attr(object, "model_type") %in% c("ngssm", "ng_bsm", "svm")) {
      future_model$distribution <- pmatch(future_model$distribution, 
        c("poisson", "binomial", "negative binomial"))
      pred <- nongaussian_predict(future_model, probs,
        t(object$theta), object$alpha[nrow(object$alpha),,], object$counts, 
        type == "response", seed, 
        pmatch(attr(object, "model_type"), c("ngssm", "ng_bsm", "svm")))
    } else {
      pred <- nonlinear_predict(t(future_model$y), future_model$Z, 
        future_model$H, future_model$T, future_model$R, future_model$Z_gn, 
        future_model$T_gn, future_model$a1, future_model$P1, 
        future_model$log_prior_pdf, future_model$known_params, 
        future_model$known_tv_params, as.integer(future_model$time_varying), 
        future_model$n_states, future_model$n_etas, probs,
        t(object$theta), object$alpha[nrow(object$alpha),,], object$counts, 
        type == "response", seed)
      
  
    }
    
  }
  class(pred) <- "predict_bssm"
  pred
}
#' @method predict bsm
#' @rdname predict
#' @export
# @examples
# require("graphics")
# y <- log10(JohnsonJohnson)
# prior <- uniform(0.01, 0, 1)
# model <- bsm(y, sd_y = prior, sd_level = prior,
#   sd_slope = prior, sd_seasonal = prior)
# 
# mcmc_results <- run_mcmc(model, n_iter = 5000)
# future_model <- model
# future_model$y <- ts(rep(8, NA), start = end(y), frequency = frequency(y))
# pred1 <- predict(mcmc_results, future_model)
# pred2 <- predict(StructTS(y, type = "BSM"), n.ahead = 8)
# 
# ts.plot(pred1$mean, pred1$intervals[,-2], pred2$pred +
# cbind(0, -qnorm(0.95) * pred2$se, qnorm(0.95) * pred2$se),
#   col = c(1, 1, 1, 2, 2, 2))
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
predict.ngssm <- function(object, n_iter, nsim_states,
  newdata = NULL,
  n_ahead = 1, interval = "mean",
  probs = c(0.05, 0.95),
  n_burnin = floor(n_iter / 2), n_thin = 1,
  gamma = 2/3, target_acceptance = 0.234, S,
  seed = sample(.Machine$integer.max, size = 1),  newu = NULL, ...) {
  
  interval <- pmatch(interval, c("mean", "response"))
  
  inits <- sapply(priors, "[[", "init")
  if (missing(S)) {
    S <- diag(0.1 * pmax(0.1, abs(inits)), length(inits))
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
  
  u_orig <- object$u
  if (is.null(newu)) {
    object$u <- c(object$u, rep(1, n_ahead))
  } else {
    if (length(newu) != n_ahead) {
      stop("Length of newu is not equal to n_ahead. ")
    } else {
      object$u<- c(object$u, newu)
    }
  }
  
  if (any(c(dim(object$Z)[3], dim(object$T)[3], dim(object$R)[3]) > 1)) {
    stop("Time-varying system matrices in prediction are not yet supported.")
  }
  probs <- sort(unique(c(probs, 0.5)))
  object$distribution <- pmatch(object$distribution,
    c("poisson", "binomial", "negative binomial"))
  priors <- combine_priors(priors)
  out <- ngssm_predict2(object, priors$prior_types, priors$params, n_iter,
    nsim_states, n_burnin, n_thin, gamma, target_acceptance, S, n_ahead, interval,
    object$Z_ind, object$T_ind, object$R_ind, object$initial_mode, seed)
  
  if (interval == 1) {
    y_orig <- y_orig / u_orig
  }
  pred <- list(y = y_orig, mean = ts(rowMeans(out), end = endtime, frequency = frequency(object$y)),
    intervals = ts(t(apply(out, 1, quantile, probs, type = 8)), end = endtime, frequency = frequency(object$y),
      names = paste0(100 * probs, "%")))
  
  
  class(pred) <- "predict_bssm"
  pred
  
}
#' @method predict ng_bsm
#' @rdname predict
#' @export
# @examples
# data("poisson_series")
# model <- ng_bsm(poisson_series, sd_level = halfnormal(0.1, 1),
#   sd_slope=halfnormal(0.01, 0.1), distribution = "poisson")
# pred <- predict(model, n_iter = 1e4, nsim = 10, n_ahead = 10,
#   probs = seq(0.05,0.95, by = 0.05))
# library("ggplot2")
# autoplot(pred, median_color = "blue", mean_color = "red")
predict.ng_bsm <- function(object, n_iter, nsim_states,
  newdata = NULL, n_ahead = 1, interval = "mean",
  probs = c(0.05, 0.95),
  n_burnin = floor(n_iter/2), n_thin = 1,
  gamma = 2/3, target_acceptance = 0.234, S,
  seed = sample(.Machine$integer.max, size = 1), newu = NULL,  ...) {
  
  interval <- pmatch(interval, c("mean", "response"))
  
  if (missing(S)) {
    S <- diag(0.1 * pmax(0.1, abs(sapply(object$priors, "[[", "init"))), length(object$priors))
  }
  
  
  
  endtime <- end(object$y) + c(0, n_ahead)
  y_orig <- object$y
  
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
  u_orig <- object$u
  if (is.null(newu)) {
    object$u <- c(object$u, rep(1, n_ahead))
  } else {
    if (length(newu) != n_ahead) {
      stop("Length of newu is not equal to n_ahead. ")
    } else {
      object$u<- c(object$u, newu)
    }
  }
  priors <- combine_priors(object$priors)
  probs <- sort(unique(c(probs, 0.5)))
  object$distribution <- pmatch(object$distribution, c("poisson", "binomial", "negative binomial"))
  
  out <- ng_bsm_predict2(object, priors$prior_types, priors$params, n_iter,
    nsim_states, n_burnin, n_thin, gamma, target_acceptance, S, n_ahead, interval,
    c(object$initial_mode, rep(log(0.1), n_ahead)), seed, FALSE)
  
  if (interval == 1) {
    y_orig <- y_orig / u_orig
  }
  pred <- list(y = y_orig, mean = ts(rowMeans(out), end = endtime, frequency = object$period),
    intervals = ts(t(apply(out, 1, quantile, probs, type = 8)), end = endtime, frequency = object$period,
      names = paste0(100 * probs, "%")))
  
  
  class(pred) <- "predict_bssm"
  pred
  
}
