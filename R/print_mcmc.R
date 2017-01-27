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
print.mcmc_output <- function(x, jump_chain = TRUE,...) {
  
  cat("\nCall:\n", paste(deparse(x$call), sep = "\n", collapse = "\n"), 
    "\n", sep = "")
  
  if (jump_chain){
    cat("\n", "Iterations = ", x$burnin + 1, ":", x$n_iter, "\n", sep = "")
    cat("Thinning interval = 1\n")
  } else {
    cat("\n", "Iterations = ", start(x$theta), ":", end(x$theta), "\n", sep = "")
    cat("Thinning interval = ", thin(x$theta), "\n")
  }
  cat("\nAcceptance rate after the burn-in period: ", paste(x$acceptance_rate,"\n", sep = ""))
  
  cat("\nSummary for theta:\n\n")
  if (jump_chain) {
    if(!is.null(x$weights)) {
    w <- x$weights * x$counts
    } else w <- x$counts
    mean_theta <- weighted_mean(x$theta, w)
    sd_theta <- sqrt(diag(weighted_var(x$theta, w)))
    se_theta <- sqrt(weighted_se(x$theta, w)^2 + x$acceptance_rate*spectrum0.ar(x$theta)$spec/nrow(x$theta))
    print(c(Mean = mean_theta, SD = sd_theta, "Asymptotic SE" = se_theta))
    cat("Effective sample sizes for theta:\n\n")
    print((sd_theta/ se_theta)^2)
  } else {
    print(summary(x$theta)$stat)
    cat("\nEffective sample sizes for theta:\n\n")
    print(effectiveSize(x$theta))
  }
  
  alpha <- mcmc(matrix(x$alpha[nrow(x$alpha),,], ncol = ncol(x$alpha), byrow = TRUE, 
    dimnames = list(NULL, colnames(x$alpha))))
  
  cat(paste0("\nSummary for alpha_",nrow(x$alpha)), ":\n\n", sep="")
  
  print(summary(alpha)$stat)
  
  cat(paste0("\nEffective sample sizes for alpha_",nrow(x$alpha)), ":\n\n", sep="")
  
  print(effectiveSize(alpha))
}
