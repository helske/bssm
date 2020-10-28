iact <- function(x) {
  n <- length(x)
  x_ <- (x - mean(x)) / sd(x)
  # Sokal: Monte Carlo Methods in Statistical Mechanics: Foundations and New Algorithms
  C <- max(5.0, log10(n))
  tau <- 1
  for(k in 1:(n-1)) {
    tau <- tau + 2.0 * (x_[1:(n-k)] %*% x_[(1+k):n]) / (n - k)
    if(k > C * tau) break
  }
  max(0.0, tau)
}

asymptotic_var <- function(x, w) {
  estimate_c <- mean(w)
  estimate_mean <- weighted_mean(x, w)
  z <- w * (x - estimate_mean)
  avar <- iact(z) * var(z) / length(z) / estimate_c^2
}

#' Print Results from MCMC Run
#'
#' Prints some basic summaries from the MCMC run by  \code{\link{run_mcmc}}.
#' 
#' In case of IS-corrected MCMC, the SE-IS is based only on importance sampling estimates, with weights 
#' corresponding to the block sizes of the jump chain multiplied by the 
#' importance correction weights (if IS-corrected method was used). These estimates
#' ignore the possible autocorrelations but provide a lower-bound for the asymptotic 
#' standard error. 
#' 
#' @method print mcmc_output
#' @importFrom diagis weighted_mean weighted_var weighted_se ess
#' @importFrom coda mcmc spectrum0.ar
#' @importFrom stats var
#' @param x Output from \code{\link{run_mcmc}}.
#' @param ... Ignored.
#' @export
print.mcmc_output <- function(x, ...) {
  
  if (x$mcmc_type %in% paste0("is", 1:3)) {
    theta <- mcmc(x$theta)
    if(x$output_type == 1)
      alpha <- mcmc(matrix(x$alpha[nrow(x$alpha),,], ncol = ncol(x$alpha), byrow = TRUE, 
                           dimnames = list(NULL, colnames(x$alpha))))
    w <- x$counts * x$weights
  } else {
    theta <- expand_sample(x, "theta")
    if(x$output_type == 1)
      alpha <- expand_sample(x, "state", times = nrow(x$alpha), by_states = FALSE)[[1]]
  }
  
  cat("\nCall:\n", paste(deparse(x$call), sep = "\n", collapse = "\n"), 
      "\n", sep = "")
  
  cat("\n", "Iterations = ", x$burnin + 1, ":", x$iter, "\n", sep = "")
  cat("Thinning interval = ",x$thin, "\n", sep = "")
  cat("Length of the final jump chain = ", length(x$counts), "\n", sep = "")
  cat("\nAcceptance rate after the burn-in period: ", paste(round(x$acceptance_rate,3),"\n", sep = ""))
  
  cat("\nSummary for theta:\n\n")
  if (x$mcmc_type %in% paste0("is", 1:3)) {
    mean_theta <- weighted_mean(theta, w)
    sd_theta <- sqrt(diag(weighted_var(theta, w, method = "moment")))
    se_theta_is <- weighted_se(theta, w)
    se_theta <- sqrt(apply(theta, 2, function(x) asymptotic_var(x, w)))
  
    stats <- matrix(c(mean_theta, sd_theta, se_theta, se_theta_is), ncol = 4, 
                    dimnames = list(colnames(x$theta), c("Mean", "SD", "SE", "SE-IS")))
  } else {
    mean_theta <- colMeans(theta)
    sd_theta <- apply(theta, 2, sd)
    se_theta <-  sqrt(spectrum0.ar(theta)$spec/nrow(theta))
    stats <- matrix(c(mean_theta, sd_theta, se_theta), ncol = 3, 
                    dimnames = list(colnames(x$theta), c("Mean", "SD", "SE")))
  }
  
  print(stats)
  
  if(x$mcmc_type %in% paste0("is", 1:3)) {
    cat("\nEffective sample sizes of weights:\n\n")
    print(ess(w))
  }
  cat("\nEffective sample sizes for theta:\n\n")
  esss <- matrix((sd_theta / se_theta)^2, ncol = 1, 
                 dimnames = list(colnames(x$theta), c("ESS")))
  print(esss)
  if(x$output_type != 3) {
    
    n <- nrow(x$alpha)
    cat(paste0("\nSummary for alpha_", n), ":\n\n", sep = "")
    
    if (is.null(x$alphahat)) {
      if (x$mcmc_type %in% paste0("is", 1:3)) {
        mean_alpha <- weighted_mean(alpha, w)
        sd_alpha <- sqrt(diag(weighted_var(alpha, w, method = "moment")))
        se_alpha_is <- weighted_se(alpha, w)
        se_alpha <- sqrt(apply(alpha, 2, function(x) asymptotic_var(x, w)))
        stats <- matrix(c(mean_alpha, sd_alpha, se_alpha,se_alpha_is), ncol = 4, 
                        dimnames = list(colnames(x$alpha), c("Mean", "SD", "SE", "SE-IS")))
      } else {
        mean_alpha <- colMeans(alpha)
        sd_alpha <- apply(alpha, 2, sd)
        se_alpha <-  sqrt(spectrum0.ar(alpha)$spec / nrow(alpha))
        stats <- matrix(c(mean_alpha, sd_alpha, se_alpha), ncol = 3, 
                        dimnames = list(colnames(x$alpha), c("Mean", "SD", "SE")))
      }
      print(stats)
      
      
      cat(paste0("\nEffective sample sizes for alpha_", n), ":\n\n", sep = "")
      esss <- matrix((sd_alpha / se_alpha)^2, ncol = 1, 
                     dimnames = list(colnames(x$alpha), c("ESS")))
      
      print(esss)
      
    } else {
      if (ncol(x$alphahat) == 1) {
        print(cbind("Mean" = x$alphahat[n, ], "SD" = sqrt(x$Vt[,,n])))
      } else {
        print(cbind("Mean" = x$alphahat[n, ], "SD" = sqrt(diag(x$Vt[,,n]))))
      }
    }
  } else cat("\nNo posterior samples for states available.\n")
  cat("\nRun time:\n")
  print(x$time)
}

#' Summary of MCMC object
#' 
#' This functions returns a list containing mean, standard deviations, standard errors, and 
#' effective sample size estimates for parameters and states.
#' 
#' @param object Output from \code{run_mcmc}
#' @param return_se if \code{FALSE} (default), computation of standard 
#' errors and effective sample sizes is omitted. 
#' @param variable Are the summary statistics computed for either \code{"theta"} (default), 
#' \code{"states"}, or \code{"both"}?
#' @param only_theta Deprecated. If \code{TRUE}, summaries are computed only for hyperparameters theta.
#' @param ... Ignored.
#' @export
summary.mcmc_output <- function(object, return_se = FALSE, variable = "theta", 
                                only_theta = FALSE, ...) {
  
  if (only_theta) {
    parameters <- "theta"
    warning("Argument 'only_theta' is deprecated. Use argument 'variable' instead. ")
  }
  variable <- match.arg(variable, c("theta", "states", "both"))
  
  if(variable %in% c("theta", "both")) {
    if (object$mcmc_type %in% paste0("is", 1:3)) {
      theta <- mcmc(object$theta)
      w <- object$counts * object$weights
      mean_theta <- weighted_mean(theta, w)
      sd_theta <- sqrt(diag(weighted_var(theta, w, method = "moment")))
      
      if(return_se) {
        mean_theta <- weighted_mean(theta, w)
        sd_theta <- sqrt(diag(weighted_var(theta, w, method = "moment")))
        se_theta_is <- weighted_se(theta, w)
        se_theta <- sqrt(apply(theta, 2, function(x) asymptotic_var(x, w)))
        ess_theta <- (sd_theta / se_theta)^2
        ess_w <- apply(object$theta, 2, function(x) ess(w, identity, x))
        summary_theta <- matrix(c(mean_theta, sd_theta, se_theta, ess_theta, se_theta_is, ess_w), ncol = 6, 
                                dimnames = list(colnames(object$theta), 
                                  c("Mean", "SD", "SE", "ESS", "SE-IS", "ESS-IS")))
      } else {
        summary_theta <- matrix(c(mean_theta, sd_theta), ncol = 2, 
                                dimnames = list(colnames(object$theta), c("Mean", "SD")))
      }
    } else {
      theta <- expand_sample(object, "theta")
      mean_theta <- colMeans(theta)
      sd_theta <- apply(theta, 2, sd)
      
      if(return_se) {
        mean_theta <- colMeans(theta)
        sd_theta <- apply(theta, 2, sd)
        se_theta <-  sqrt(spectrum0.ar(theta)$spec/nrow(theta))
        ess_theta <- (sd_theta / se_theta)^2
        summary_theta <- matrix(c(mean_theta, sd_theta, se_theta, ess_theta), ncol = 4, 
                                dimnames = list(colnames(object$theta), c("Mean", "SD", "SE", "ESS")))
      } else {
        summary_theta <- matrix(c(mean_theta, sd_theta), ncol = 2, 
                                dimnames = list(colnames(object$theta), c("Mean", "SD")))
      }
    }
  }
  
  if (variable %in% c("states", "both")) {
    if (object$output_type != 1) stop("Cannot return summary of states as the MCMC type was not 'full'. ")
    
    m <- ncol(object$alpha)
    
    if (object$mcmc_type %in% paste0("is", 1:3)) {
      w <- object$counts * object$weights
      mean_alpha <- ts(weighted_mean(object$alpha, w), start = attr(object, "ts")$start,
                       frequency = attr(object, "ts")$frequency, names = colnames(object$alpha))
      sd_alpha <- weighted_var(object$alpha, w, method = "moment")
      sd_alpha <- if(m > 1) sqrt(t(apply(sd_alpha, 3, diag))) else matrix(sqrt(sd_alpha), ncol = 1)
      
      
      if(return_se) {
        se_alpha_is <- apply(object$alpha, 2, function(x) weighted_se(t(x), w))
        
        se_alpha <- apply(object$alpha, 2, function(z) sqrt(apply(z, 1, function(x) asymptotic_var(x, w))))
        alpha_ess <- (sd_alpha / se_alpha)^2
        ess_w <- apply(object$alpha, 2, function(z) apply(z, 1, function(x) ess(w, identity, x)))
        summary_alpha <- list(
          "Mean" = mean_alpha, "SD" = sd_alpha, 
          "SE" = se_alpha, "ESS" = alpha_ess, 
          "SE-IS" = se_alpha_is, "ESS-IS" = ess_w)
      } else {
        summary_alpha <- list("Mean" = mean_alpha, "SD" = sd_alpha)
      }
      
    } else {
      alpha <- expand_sample(object, "states")
      mean_alpha <- ts(sapply(alpha, colMeans),
                       start = attr(object, "ts")$start,
                       frequency = attr(object, "ts")$frequency, names = colnames(object$alpha))
      sd_alpha <- sapply(alpha, function(x) apply(x, 2, sd))
      
      if(return_se) {
        
        se_alpha <- sapply(alpha, function(x) 
          apply(x, 2, function(z) 
            sqrt(spectrum0.ar(z)$spec / length(z))))
        ess_alpha <- (sd_alpha / se_alpha)^2
        summary_alpha <- list(
          "Mean" = mean_alpha, "SD" = sd_alpha, 
          "SE" = se_alpha, "ESS" = ess_alpha)
      } else {
        summary_alpha <- list("Mean" = mean_alpha, "SD" = sd_alpha)
      }
    }
  }
  switch(variable,
         "both" = return(list(theta = summary_theta, states = summary_alpha)),
         "theta" = return(summary_theta),
         "states" = return(summary_alpha)
  )
}

#' Expand the Jump Chain representation
#'
#' The MCMC algorithms of \code{bssm} use a jump chain representation where we 
#' store the accepted values and the number of times we stayed in the current value.
#' Although this saves bit memory and is especially convenient for IS-corrected 
#' MCMC, sometimes we want to have the usual sample paths. Function \code{expand_sample} 
#' returns the expanded sample based on the counts. Note that for IS-corrected output the expanded 
#' sample corresponds to the approximate posterior.
#' 
#' @param x Output from \code{\link{run_mcmc}}.
#' @param variable Expand parameters \code{"theta"} or states \code{"states"}.
#' @param times Vector of indices. In case of states, what time points to expand? Default is all.
#' @param states Vector of indices. In case of states, what states to expand? Default is all.
#' @param by_states If \code{TRUE} (default), return list by states. Otherwise by time.
#' @export
expand_sample <- function(x, variable = "theta", times, states, by_states = TRUE) {
  
  variable <- match.arg(variable, c("theta", "states"))
  if (x$mcmc_type %in% paste0("is", 1:3)) 
    warning("Input is based on a IS-weighted MCMC, the results correspond to the approximate posteriors.")
  if(variable == "theta") {
    out <- apply(x$theta, 2, rep, times = x$counts)
  } else {
    if (x$output_type == 1) {
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
    } else stop("MCMC output does not contain posterior samples of states.")
  }
  mcmc(out, start = x$burnin + 1, thin = x$thin)
}
