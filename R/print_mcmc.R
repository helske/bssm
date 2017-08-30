#' Print Results from MCMC Run
#'
#' Prints some basic summaries from the MCMC run by  \code{\link{run_mcmc}}.
#' 
#' Two-types of standard error and effective sample size estimates are returned. 
#' SE-IS (ESS-IS) are based only on importance sampling estimates, with weights 
#' corresponding to the block sizes of the jump chain multiplied by the 
#' importance correction weights (if IS-corrected method was used). These estimates
#' ignore the possible autocorrelations but provide a lower-bound for the asymptotic 
#' standard error. The SE-AR (ESS-AR) estimates are based on the spectral density 
#' of \eqn{(x-hatx) * w} where \eqn{hatx} is the weighted mean of \eqn{x} and 
#' \eqn{w} contains the weights. See vignette for details.
#' 
#' @method print mcmc_output
#' @importFrom diagis weighted_mean weighted_var weighted_se ess
#' @importFrom coda mcmc spectrum0.ar
#' @param x Output from \code{\link{run_mcmc}}.
#' @param ... Ignored.
#' @export
print.mcmc_output <- function(x, ...) {
  
  theta <- mcmc(x$theta)
  alpha <- mcmc(matrix(x$alpha[nrow(x$alpha),,], ncol = ncol(x$alpha), byrow = TRUE, 
    dimnames = list(NULL, colnames(x$alpha))))
  w <- x$counts * if (x$mcmc_type %in% paste0("is", 1:3)) x$weights else 1
  
  
  cat("\nCall:\n", paste(deparse(x$call), sep = "\n", collapse = "\n"), 
    "\n", sep = "")
  
  cat("\n", "Iterations = ", x$n_burnin + 1, ":", x$n_iter, "\n", sep = "")
  cat("Thinning interval = ",x$n_thin, "\n", sep = "")
  cat("Length of the final jump chain = ", length(x$counts), "\n", sep = "")
  cat("\nAcceptance rate after the burn-in period: ", paste(x$acceptance_rate,"\n", sep = ""))
  
  cat("\nSummary for theta:\n\n")
  mean_theta <- weighted_mean(theta, w)
  sd_theta <- sqrt(diag(weighted_var(theta, w, method = "moment")))
  se_theta_is <- weighted_se(theta, w)
  spec <- sapply(1:ncol(theta), function(i) spectrum0.ar((theta[, i] - mean_theta[i]) * w)$spec)
  se_theta_ar <- sqrt(spec / length(w)) / mean(w)
  se_theta_total <- sqrt(se_theta_is^2 + se_theta_ar^2)
  stats <- matrix(c(mean_theta, sd_theta, se_theta_is, se_theta_ar, se_theta_total), ncol = 5, 
    dimnames = list(colnames(x$theta), c("Mean", "SD", "SE-IS", "SE-AR", "SE")))
  print(stats)
  
  cat("\nEffective sample sizes for theta:\n\n")
  ess_theta_is <- apply(theta, 2, function(z) ess(w, identity, z))
  ess_theta_ar <- (sd_theta / se_theta_ar)^2
  esss <- matrix(c(ess_theta_is, ess_theta_ar), ncol = 2, 
    dimnames = list(colnames(x$theta), c("ESS-IS", "ESS-AR")))
  print(esss)
  
  
  
  cat(paste0("\nSummary for alpha_",nrow(x$alpha)), ":\n\n", sep="")
  mean_alpha <- weighted_mean(alpha, w)
  sd_alpha <- sqrt(diag(weighted_var(alpha, w, method = "moment")))
  se_alpha_is <- weighted_se(alpha, w)
  spec <- sapply(1:ncol(alpha), function(i) spectrum0.ar((alpha[, i] - mean_alpha[i]) * w)$spec)
  se_alpha_ar <- sqrt(spec / length(w)) / mean(w)
  se_alpha_total <- sqrt(se_alpha_is^2 + se_alpha_ar^2)
  stats <- matrix(c(mean_alpha, sd_alpha, se_alpha_is, se_alpha_ar, se_alpha_total), ncol = 5, 
    dimnames = list(colnames(x$alpha), c("Mean", "SD", "SE-IS", "SE-AR", "SE")))
  print(stats)
  
  cat("\nEffective sample sizes for alpha:\n\n")
  ess_alpha_is <- apply(alpha, 2, function(z) ess(w, identity, z))
  ess_alpha_ar <- (sd_alpha / se_alpha_ar)^2
  esss <- matrix(c(ess_alpha_is, ess_alpha_ar), ncol = 2, 
    dimnames = list(colnames(x$alpha), c("ESS-IS", "ESS-AR")))
  print(esss)
  
  cat("\nRun time:\n")
  print(x$time)
}
#' Summary of MCMC object
#' 
#' This functions returns a list containing mean, standard deviations, standard errors, and 
#' effective sample size estimates for parameters and states.
#' 
#' Note that computing the state summaries can be slow for large models due to repeated 
#' calls to \code{\link[coda]{spectrum0.ar}}.
#' 
#' @param object Output from \code{run_mcmc}
#' @param return_se if \code{FALSE} (default), computation of standard 
#' errors and effective sample sizes is omitted. 
#' This saves time, as computing the spectral densities (by \code{coda}) can be slow for 
#' large models.
#' @param only_theta If \code{TRUE}, summaries are computed only for hyperparameters theta. 
#' @param ... Ignored.
#' @export
summary.mcmc_output <- function(object, return_se = FALSE, only_theta = FALSE, ...) {
  
  
  theta <- mcmc(object$theta)
  w <- object$counts * if (object$mcmc_type %in% paste0("is", 1:3)) object$weights else 1
  
  mean_theta <- weighted_mean(theta, w)
  sd_theta <- sqrt(diag(weighted_var(theta, w, method = "moment")))
  if(return_se) {
    se_theta_is <- weighted_se(theta, w)
    spec <- sapply(1:ncol(theta), function(i) spectrum0.ar((theta[, i] - mean_theta[i]) * w)$spec)
    se_theta_ar <- sqrt(spec / length(w)) / mean(w)
    se_theta_total <- sqrt(se_theta_is^2 + se_theta_ar^2)
    ess_theta_is <- apply(theta, 2, function(z) ess(w, identity, z))
    ess_theta_ar <- (sd_theta / se_theta_ar)^2
    summary_theta <- matrix(c(
      mean_theta, sd_theta, se_theta_is, se_theta_ar, se_theta_total,
      ess_theta_is, ess_theta_ar), 
      ncol = 7, 
      dimnames = list(
        colnames(object$theta), 
        c("Mean", "SD", "SE-IS", "SE-AR", "SE", "ESS-IS", "ESS-AR")))
  } else {
    summary_theta <- matrix(c(mean_theta, sd_theta), ncol = 2, 
      dimnames = list(colnames(object$theta), c("Mean", "SD")))
  }
  
  if (!only_theta) {
    m <- ncol(object$alpha)
    mean_alpha <- weighted_mean(object$alpha, w)
    sd_alpha <- weighted_var(object$alpha, w, method = "moment")
    sd_alpha <- if(m > 1) sqrt(t(apply(sd_alpha, 3, diag))) else matrix(sqrt(sd_alpha), ncol = 1)
    if(return_se) {
      se_alpha_is <- apply(object$alpha, 2, function(x) weighted_se(t(x), w))
      spec <- matrix(NA, ncol(object$alpha), nrow(object$alpha))
      for(j in 1:nrow(object$alpha)) {
        spec[, j] <- sapply(1:ncol(object$alpha), function(i) spectrum0.ar((object$alpha[j, i, ] - mean_alpha[j, i]) * w)$spec)
      }
      se_alpha_ar <- sqrt(t(spec) / length(w)) / mean(w)
      se_alpha_total <- sqrt(se_alpha_is^2 + se_alpha_ar^2)
      alpha_ess_is <- apply(object$alpha, 2, function(x) apply(t(x), 2, function(z) ess(w, identity, z)))
      alpha_ess_ar <- (sd_alpha / se_alpha_ar)^2
      summary_alpha <- list(
        "Mean" = mean_alpha, "SD" = sd_alpha, 
        "SE-IS" = se_alpha_is, "SE-AR" = se_alpha_ar, 
        "SE" = se_alpha_total, "ESS-IS" = alpha_ess_is, "ESS-AR" = alpha_ess_ar)
    } else {
      summary_alpha <- list("Mean" = mean_alpha, "SD" = sd_alpha)
    }
    return(list(theta = summary_theta, states = summary_alpha))
  } else summary_theta
}

#' Expand the Jump Chain representation
#'
#' The MCMC algorithms of \code{bssm} use a jump chain representation where we 
#' store the accepted values and the number of times we stayed in the current value.
#' Although this saves bit memory and is especially convinient for IS-corrected 
#' MCMC, sometimes we want to have the usual sample paths. Function \code{expand} 
#' returns the expanded sample based on the counts.
#' 
#' @param x Output from \code{\link{run_mcmc}}.
#' @param variable Expand parameters \code{"theta"} or states \code{"state"}.
#' @param times Vector of indices. In case of states, what time points to expand? Default is all.
#' @param states Vector of indices. In case of states, what states to expand? Default is all.
#' @param by_states If \code{TRUE} (default), return list by states. Otherwise by time.
#' @param ... Ignored.
#' @export
expand_sample <- function(x, variable = "theta", times, states, by_states = TRUE) {
  if(variable == "theta") {
    out <- apply(x$theta, 2, rep, times = x$counts)
  } else {
    if(missing(times)) times <- 1:nrow(x$alpha)
    if(missing(states)) states <- 1:ncol(x$alpha)
    
    if(by_states) {
      out <- lapply(states, function(i) {
        z <- apply(x$alpha[times, i, , drop = FALSE], 1, rep, x$counts)
        colnames(z) <- times
        z
      })
      names(out) <- colnames(x$alpha)[states]
    } else {
      out <- lapply(times, function(i) {
        z <- apply(x$alpha[i, states, , drop = FALSE], 2, rep, x$counts)
        colnames(z) <- colnames(x$alpha)[states]
        z
      })
      names(out) <- times
    }
  }
  mcmc(out, start = x$n_burnin + 1, thin = x$n_thin)
}
# 
# 
# resample_sample <- function(x, variable = "theta", times, states) {
#   
#   out <- expand_sample(x, variable, times, states)
#   mcmc(apply(out, 2, function(y) sample(y, replace=TRUE, prob = rep(x$weights, x$counts))))
# }
