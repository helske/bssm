#' Predictions for State Space Models
#'
#' Posterior intervals of future observations or their means
#' (success probabilities in binomial case) for Gaussian models. These are
#' computed using either the quantile method where the intervals are computed
#' as empirical quantiles the posterior sample, or parametric method by
#' Helske (2016).
#'
#' @param object mcmc_output object obtained from \code{\link{run_mcmc}}
#' @param intervals If \code{TRUE}, intervals are returned. Otherwise samples 
#' from the posterior predictive distribution are returned.
#' @param type Compute predictions on \code{"mean"} ("confidence interval"),
#' \code{"response"} ("prediction interval"), or \code{"state"} level. 
#' Defaults to \code{"response"}.
#' @param probs Desired quantiles. Defaults to \code{c(0.05, 0.95)}. Always includes median 0.5.
#' @param future_model Model for future observations. Should have same structure
#' as the original model which was used in MCMC, in order to plug the posterior 
#' samples of the model parameters to right place.
#' @param return_MCSE For Gaussian models, if \code{TRUE}, the Monte Carlo
#' standard errors of the intervals are also returned.
#' @param seed Seed for RNG.
#' @param ... Ignored.
#' @return List containing the mean predictions, quantiles and Monte Carlo
#' standard errors .
#' @method predict mcmc_output
#' @rdname predict
#' @export
#' @examples
#' require("graphics")
#' y <- log10(JohnsonJohnson)
#' prior <- uniform(0.01, 0, 1)
#' model <- bsm(window(y, end = c(1974, 4)), sd_y = prior, sd_level = prior,
#'   sd_slope = prior, sd_seasonal = prior)
#' 
#' mcmc_results <- run_mcmc(model, n_iter = 5000)
#' future_model <- model
#' future_model$y <- ts(rep(NA, 25), start = end(model$y), 
#'   frequency = frequency(model$y))
#' pred_gaussian <- predict(mcmc_results, future_model, 
#'   probs = c(0.05, 0.1, 0.5, 0.9, 0.95))
#' ts.plot(log10(JohnsonJohnson), pred_gaussian$intervals, 
#'   col = c(1, rep(2, 5)), lty = c(1, 2, 2, 1, 2, 2))
#'
#' head(pred_gaussian$intervals)
#' head(pred_gaussian$MCSE)
#' 
#' # Non-gaussian models
#' \dontrun{
#' data("poisson_series")
#' 
#' model <- ng_bsm(poisson_series, sd_level = halfnormal(0.1, 1),
#'   sd_slope=halfnormal(0.01, 0.1), distribution = "poisson")
#' mcmc_poisson <- run_mcmc(model, n_iter = 5000, nsim = 10)
#'
#' future_model <- model
#' future_model$y <- ts(rep(NA, 25), start = end(model$y), 
#'   frequency = frequency(model$y))
#' 
#' pred <- predict(mcmc_poisson, future_model, 
#'   probs = seq(0.05,0.95, by = 0.05))
#'
#' library("ggplot2")
#' fit <- ts(colMeans(exp(expand_sample(mcmc_poisson, "alpha")$level)))
#' autoplot(pred, y = model$y, fit = fit)
#' }
predict.mcmc_output <- function(object, future_model, type = "response",
  intervals = TRUE, probs = c(0.05, 0.95), return_MCSE = TRUE, 
  seed = sample(.Machine$integer.max, size = 1), ...) {
  
  type <- match.arg(type, c("response", "mean", "state"))
  
  probs <- sort(unique(c(probs, 0.5)))
  n_ahead <- length(future_model$y)
  start_ts <- start(future_model$y)
  end_ts <- end(future_model$y)
  freq <- frequency(future_model$y)
  
  if (attr(object, "model_type") %in% c("gssm", "bsm")) {
    
    out <- gaussian_predict(future_model, probs,
      t(object$theta), object$alpha[nrow(object$alpha),,], object$counts, 
      pmatch(type, c("response", "mean", "state")), intervals, 
      seed, pmatch(attr(object, "model_type"), c("gssm", "bsm")))
    
    if (intervals) {
      
      if(return_MCSE) {
        if(type != "state") {
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
          m <- length(future_model$a1)
          ses <- replicate(m, ts(matrix(0, n_ahead, length(probs)), 
            start = start_ts, end = end_ts, frequency = freq,
            names = paste0(100*probs, "%")), simplify = FALSE)
          nsim <- nrow(out$mean_pred)
          
          for (k in 1:m) {
            for (i in 1:n_ahead) {
              for (j in 1:length(probs)) {
                pnorms <- pnorm(q = out$intervals[i, j, k], out$mean_pred[, i, k], out$sd_pred[, i, k])
                eff_n <-  effectiveSize(pnorms)
                ses[[k]][i, j] <- sqrt((sum((probs[j] - pnorms) ^ 2) / nsim) / eff_n) /
                  sum(dnorm(x = out$intervals[i, j, k], out$mean_pred[, i, k], out$sd_pred[, i, k]) / nsim)
              }
            }
          }
          
          intv <- lapply(1:m, function(i) ts(out$intervals[,,i], 
            start = start_ts, end = end_ts, frequency = freq,
            names = paste0(100*probs, "%")))
          names(ses) <- names(intv) <- names(future_model$a1)
          pred <- list(mean = ts(colMeans(out$mean_pred), start = start_ts, 
            end = end_ts, frequency = freq, names = names(intv)), 
            intervals = intv, MCSE = ses)
        }
        
        
      } else {
        if(type != "state") {
          pred <- list(mean = ts(colMeans(out$mean_pred), start = start_ts, end = end_ts, frequency = freq),
            intervals = ts(out$intervals, start = start_ts, end = end_ts, frequency = freq,
              names = paste0(100 * probs, "%"))) 
        } else {
          intv <- lapply(1:length(future_model$a1), function(i) ts(out$intervals[,,i], 
            start = start_ts, end = end_ts, frequency = freq,
            names = paste0(100*probs, "%")))
          names(intv) <- names(future_model$a1)
          pred <- list(mean = ts(colMeans(out$mean_pred), start = start_ts, 
            end = end_ts, frequency = freq, names = names(intv)), intervals = intv) 
        }
        
      }
    } else {
      pred <- aperm(out[[1]], c(2, 1, 3))
    }
  } else {
    if(attr(object, "model_type") %in% c("ngssm", "ng_bsm", "svm")) {
      
      future_model$distribution <- pmatch(future_model$distribution, 
        c("poisson", "binomial", "negative binomial"))
      out <- nongaussian_predict(future_model, probs,
        t(object$theta), object$alpha[nrow(object$alpha),,], object$counts, 
        pmatch(type, c("response", "mean", "state")), seed, 
        pmatch(attr(object, "model_type"), c("ngssm", "ng_bsm", "svm")))
      if(intervals) {
        if (type != "state") {
          pred <- list(mean = ts(rowMeans(out[1,,]),  start = start_ts, end = end_ts, 
            frequency = freq, names = names(future_model$a1)),
            intervals = ts(t(apply(out[1,,], 1, quantile, probs, type = 8)), 
              start = start_ts, end = end_ts, frequency = freq, 
              names = paste0(100 * probs, "%")))
        } else {
          intv <- lapply(1:length(future_model$a1), function(i) 
            ts(t(apply(out[i,,], 1, quantile, probs, type = 8)), 
              start = start_ts, end = end_ts, frequency = freq,
              names = paste0(100*probs, "%")))
          names(intv) <- names(future_model$a1)
          
          pred <- list(mean = ts(apply(out, 1, rowMeans), start = start_ts, end = end_ts, 
            frequency = freq, names = names(future_model$a1)),
            intervals = intv)
        }
      } else {
        pred <- out
      }
    } else {
      out <- nonlinear_predict(t(future_model$y), future_model$Z, 
        future_model$H, future_model$T, future_model$R, future_model$Z_gn, 
        future_model$T_gn, future_model$a1, future_model$P1, 
        future_model$log_prior_pdf, future_model$known_params, 
        future_model$known_tv_params, as.integer(future_model$time_varying),
        future_model$n_states, future_model$n_etas, probs,
        t(object$theta), object$alpha[nrow(object$alpha),,], object$counts, 
        pmatch(type, c("response", "mean", "state")), seed)
      
      if(intervals) {
        if (type != "state") {
          if (is.null(ncol(future_model$y)) || ncol(future_model$y) == 1) {
            intv <- ts(t(apply(out[1,,], 1, quantile, probs, type = 8)),
              start = start_ts, end = end_ts, frequency = freq, 
              names = paste0(100 * probs, "%"))
          } else {
            intv <- lapply(1:ncol(future_model$y), function(i) 
              ts(t(apply(out[i,,], 1, quantile, probs, type = 8)), 
                start = start_ts, end = end_ts, frequency = freq,
                names = paste0(100*probs, "%")))
            names(intv) <- colnames(future_model$y)
          }
          pred <- list(mean = ts(apply(out, 1, rowMeans), start = start_ts, 
            end = end_ts, frequency = freq, names = colnames(future_model$y)),
            intervals = intv)
        } else {
          intv <- lapply(1:future_model$n_states, function(i) 
            ts(t(apply(out[i,,], 1, quantile, probs, type = 8)), 
              start = start_ts, end = end_ts, frequency = freq,
              names = paste0(100*probs, "%")))
          names(intv) <- future_model$state_names
          pred <- list(mean = ts(apply(out, 1, rowMeans), start = start_ts, 
            end = end_ts, frequency = freq, names = future_model$state_names),
            intervals = intv)
        }
        
      } else {
        pred <- out
      }
      
    }
    
  }
  class(pred) <- "predict_bssm"
  pred
}
