#' Print Results from MCMC Run
#'
#' Prints some basic summaries from the MCMC run by \code{\link{run_mcmc}}.
#'  
#' @method print mcmc_output
#' @importFrom diagis weighted_mean weighted_var weighted_se ess
#' @importFrom stats var
#' @param x Object of class \code{mcmc_output} from \code{\link{run_mcmc}}.
#' @param ... Ignored.
#' @srrstats {BS5.3, BS5.5, BS6.0}
#' @export
#' @examples
#' data("negbin_model")
#' print(negbin_model)
print.mcmc_output <- function(x, ...) {
  
  cat("\nCall:\n", paste(deparse(x$call), sep = "\n", collapse = "\n"), 
    "\n", sep = "")
  
  cat("\n", "Iterations = ", x$burnin + 1, ":", x$iter, "\n", sep = "")
  cat("Thinning interval = ", x$thin, "\n", sep = "")
  cat("Length of the final jump chain = ", length(x$counts), "\n", sep = "")
  cat("\nAcceptance rate after the burn-in period: ", 
    paste(round(x$acceptance_rate, 3), "\n", sep = ""))
  
  cat("\nSummary for theta:\n\n")
  stats <- as.data.frame(summary(x, variable = "theta", return_se = TRUE))
  print(stats, row.names = FALSE)
  if (x$output_type != 3) {
    n <- nrow(x$alpha)
    cat(paste0("\nSummary for alpha_", n), ":\n\n", sep = "")
    
    if (is.null(x$alphahat)) {
      stats <- as.data.frame(summary(x, variable = "states", times = n, 
        return_se = TRUE))
      print(stats, row.names = FALSE)
    } else {
      if (ncol(x$alphahat) == 1) {
        print(cbind("Mean" = x$alphahat[n, ], "SD" = sqrt(x$Vt[, , n])))
      } else {
        print(cbind("Mean" = x$alphahat[n, ], "SD" = sqrt(diag(x$Vt[, , n]))))
      }
    }
  } else cat("\nNo posterior samples for states available.\n")
  cat("\nRun time:\n")
  print(x$time)
  invisible(x)
}
