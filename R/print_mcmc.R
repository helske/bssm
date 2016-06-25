#' Print Results from MCMC Run
#'
#' Prints some basic summaries from the MCMC run by  \code{\link{run_mcmc}}.
#'
#' @method print mcmc_output
#' @param x Output from \code{\link{run_mcmc}}.
#' @param ... Ignored.
#' @export
print.mcmc_output <- function(x, ...) {

  cat("\nCall:\n", paste(deparse(x$call), sep = "\n", collapse = "\n"), 
    "\n\n", sep = "")

  cat("\nAcceptance rate after the burnin period: ", paste(x$acceptance_rate,"\n\n", sep = ""))
  
  cat("\nSummary for theta:\n\n")
  
  print(summary(x$theta)$stat)
  
  cat("\nEffective sample sizes for theta:\n\n")
  
  print(effectiveSize(x$theta))
  
  cat(paste0("\nSummary for alpha_",nrow(x$alpha)), ":\n\n", sep="")
  
  print(summary(mcmc(t(x$alpha[nrow(x$alpha),,])))$stat)
  
  cat(paste0("\nEffective sample sizes for alpha_",nrow(x$alpha)), ":\n\n", sep="")
  
  print(effectiveSize(mcmc(t(x$alpha[nrow(x$alpha),,]))))
  
}