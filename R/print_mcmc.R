#' Print Results from MCMC Run
#'
#' Prints some basic summaries from the MCMC run by  \code{\link{run_mcmc}}.
#' 
#' @method print mcmc_output
#' @importFrom diagis weighted_mean weighted_var weighted_se
#' @importFrom coda spectrum0.ar
#' @param x Output from \code{\link{run_mcmc}}.
#' @param ... Ignored.
#' @export
print.mcmc_output <- function(x, ...) {
  
  cat("\nCall:\n", paste(deparse(x$call), sep = "\n", collapse = "\n"), 
    "\n", sep = "")
  
  
  cat("\n", "Iterations = ", x$n_burnin + 1, ":", x$n_iter, "\n", sep = "")
  cat("Thinning interval = ",x$n_thin, "\n", sep = "")
  cat("\nAcceptance rate after the burn-in period: ", paste(x$acceptance_rate,"\n", sep = ""))
  
  cat("\nSummary for theta:\n\n")
  if (x$isc) {
    w <- x$weights * x$counts
    mean_theta <- weighted_mean(x$theta, w)
    sd_theta <- sqrt(diag(weighted_var(x$theta, w)))
    se_theta_obm <- apply(x$theta, 2, weighted_obm, w)
    se_theta_is <- weighted_se(x$theta, w)
    stats <- matrix(c(mean_theta, sd_theta, se_theta_is, se_theta_obm), ncol = 4, 
      dimnames = list(colnames(x$theta), c("Mean", "SD", "Lower bound of SE", "Asymptotic SE")))
    print(stats)
    
    cat("\n(experimental) Effective sample sizes for theta:\n\n")
    print((sd_theta/ se_theta_obm)^2)
  } else {
    theta <- mcmc(apply(x$theta, 2, rep, times = x$counts))
    print(summary(theta)$stat)
    cat("\nEffective sample sizes for theta:\n\n")
    print(effectiveSize(theta))
  }
  
  alpha <- mcmc(matrix(x$alpha[nrow(x$alpha),,], ncol = ncol(x$alpha), byrow = TRUE, 
    dimnames = list(NULL, colnames(x$alpha))))
  
  cat(paste0("\nSummary for alpha_",nrow(x$alpha)), ":\n\n", sep="")
  
  if (x$isc) {
    w <- x$weights * x$counts
    mean_alpha <- weighted_mean(alpha, w)
    sd_alpha <- sqrt(diag(weighted_var(alpha, w)))
    se_alpha <- apply(alpha, 2, weighted_obm, w)
    se_alpha_obm <- apply(alpha, 2, weighted_obm, w)
    se_alpha_is <- weighted_se(alpha, w)
    stats <- matrix(c(mean_alpha, sd_alpha, se_alpha_is, se_alpha_obm), ncol = 4, 
      dimnames = list(colnames(alpha), c("Mean", "SD", "Lower bound of SE", "Asymptotic SE")))
    print(stats)
    
    cat(paste0("\n (experimental) Effective sample sizes for alpha_",nrow(x$alpha)), ":\n\n", sep="")
    print((sd_alpha/ se_alpha_obm)^2)
  } else {
    alpha <- mcmc(apply(alpha, 2, rep, times = x$counts))
    print(summary(alpha)$stat)
    cat(paste0("\nEffective sample sizes for alpha_",nrow(x$alpha)), ":\n\n", sep="")
    print(effectiveSize(alpha))
  }
}

#' Expand the Jump Chain representation
#'
#' The MCMC algorithms of \code{bssm} use a jump chain representation where we 
#' store the accepted values and the number of times we stayed in the current value.
#' Although this safes bit memory and is especially convinient for IS-corrected 
#' MCMC, often we want to have the usual sample path. Function \code{expand} 
#' returns the expanded sample based on the counts.
#' 
#' @param x Output from \code{\link{run_mcmc}}.
#' @param ... Ignored.
#' @export
expand <- function(x, variable = "theta", times, states, by_states = TRUE) {
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
  mcmc(out)
}
# 
# 
# resample_sample <- function(x, variable = "theta", times, states) {
#   
#   out <- expand_sample(x, variable, times, states)
#   mcmc(apply(out, 2, function(y) sample(y, replace=TRUE, prob = rep(x$weights, x$counts))))
# }
