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
  
  print("Warning!!! The summary for IS-corrected method are currently incorrect!!.")
  cat("\nCall:\n", paste(deparse(x$call), sep = "\n", collapse = "\n"), 
    "\n", sep = "")
  
  
  cat("\n", "Iterations = ", x$n_burnin + 1, ":", x$n_iter, "\n", sep = "")
  cat("Thinning interval = ",x$n_thin, "\n", sep = "")
  cat("\nAcceptance rate after the burn-in period: ", paste(x$acceptance_rate,"\n", sep = ""))
  
  cat("\nSummary for theta:\n\n")
  if (x$isc) {
    theta <- mcmc(apply(x$theta, 2, rep, times = x$counts))
    w <- rep(x$weights, x$counts)
    print(summary(theta * w)$stat / mean(w))
    cat("\nEffective sample sizes for theta:\n\n")
    print(effectiveSize(theta*w)*sum(w)/length(w))
  } else {
    theta <- mcmc(apply(x$theta, 2, rep, times = x$counts))
    print(summary(theta)$stat)
    cat("\nEffective sample sizes for theta:\n\n")
    print(effectiveSize(theta))
  }
  
  alpha <- mcmc(apply(matrix(x$alpha[nrow(x$alpha),,], ncol = ncol(x$alpha), byrow = TRUE, 
    dimnames = list(NULL, colnames(x$alpha))), 2, rep, times = x$counts))
  cat(paste0("\nSummary for alpha_",nrow(x$alpha)), ":\n\n", sep="")
  
  if (x$isc) {
    w <- rep(x$weights, x$counts)
    print(summary(alpha * w)$stat / mean(x$weights))
    cat(paste0("\nEffective sample sizes for alpha_",nrow(x$alpha)), ":\n\n", sep="")
    print(effectiveSize(alpha * w) * sum(w) / length(w))
  } else {
    print(summary(alpha)$stat)
    cat(paste0("\nEffective sample sizes for alpha_",nrow(x$alpha)), ":\n\n", sep="")
    print(effectiveSize(alpha))
  }
  print("Warning!!! The summary for IS-corrected method are currently incorrect!!.")
}

#' Expand the Jump Chain representation
#'
#' The MCMC algorithms of \code{bssm} use a jump chain representation where we 
#' store the accepted values and the number of times we stayed in the current value.
#' Although this saves bit memory and is especially convinient for IS-corrected 
#' MCMC, sometimes we want to have the usual sample path. Function \code{expand} 
#' returns the expanded sample based on the counts.
#' 
#' @param x Output from \code{\link{run_mcmc}}.
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
