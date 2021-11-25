#' Quick Diagnostics Checks for \code{run_mcmc} Output
#'
#' Prints out the acceptance rate, smallest effective sample sizes (ESS) and 
#' largest Rhat values for a quick first check that the sampling worked. For 
#' further checks, see e.g. \code{bayesplot} and \code{coda} packages.
#' 
#' For methods other than IS-MCMC, the estimates are based on the improved 
#' diagnostics from the \code{posterior} package.For IS-MCMC, these Rhat, 
#' bulk-ESS, and tail-ESS estimates are based on the approximate posterior 
#' which should look reasonable, otherwise the IS-correction does not make much 
#' sense. For IS-MCMC, ESS estimates based on a weighted posterior are also 
#' computed.
#' 
#' @importFrom posterior summarise_draws default_convergence_measures
#' @param x Results object of class \code{mcmc_output} from 
#' \code{\link{run_mcmc}}.
#' @export
#' @srrstats {BS5.3, BS5.5} Several options for ESS. See also asymptotic_var.R 
#' and summary functions
#' @examples
#' set.seed(1)
#' n <- 30
#' phi <- 2
#' rho <- 0.9
#' sigma <- 0.1
#' beta <- 0.5
#' u <- rexp(n, 0.1)
#' x <- rnorm(n)
#' z <- y <- numeric(n)
#' z[1] <- rnorm(1, 0, sigma / sqrt(1 - rho^2))
#' y[1] <- rnbinom(1, mu = u * exp(beta * x[1] + z[1]), size = phi)
#' for(i in 2:n) {
#'   z[i] <- rnorm(1, rho * z[i - 1], sigma)
#'   y[i] <- rnbinom(1, mu = u * exp(beta * x[i] + z[i]), size = phi)
#' }
#' 
#' model <- ar1_ng(y, rho = uniform_prior(0.9, 0, 1), 
#'   sigma = gamma_prior(0.1, 2, 10), mu = 0., 
#'   phi = gamma_prior(2, 2, 1), distribution = "negative binomial",
#'   xreg = x, beta = normal_prior(0.5, 0, 1), u = u)
#'   
#' out <- run_mcmc(model, iter = 1000, particles = 10)
#' check_diagnostics(out)
check_diagnostics <- function(x) {
  
  cat("\nAcceptance rate after the burn-in period: ", 
    paste(round(x$acceptance_rate, 3), "\n", sep = ""))
  
  cat("\nRun time (wall-clock):\n")
  cat(paste(ifelse(x$time[3] < 10, round(x$time[3], 2), round(x$time[3])), 
    "seconds.\n"))
  
  if (any(is.na(x$theta)) || any(is.na(x$alpha))) {
    warning("NA value found in samples.")
  }
  draws <- suppressWarnings(as_draws(x))
  
  is_run <- x$mcmc_type %in% paste0("is", 1:3)
  if (is_run) {
    # removing hidden variables of draws object gives warning, we don't care
    ess <- apply(suppressWarnings(draws[, 2:(ncol(draws) - 3)]), 
      2, function(x) {
        weighted_var(x, draws$weight) / asymptotic_var(x, draws$weight)
      })
    min_ess <- which.min(ess)
    cat("\nSmallest ESS based on weighted posterior: ", 
      round(ess[min_ess]), " (", names(ess)[min_ess], ")", sep = "")
    
    ess_is <- apply(suppressWarnings(draws[, 2:(ncol(draws) - 3)]), 2, 
      function(x) ess(draws$weight, identity, x))
    min_ess <- which.min(ess_is)
    cat("\nSmallest ESS based on independent importance sampling: ", 
      round(ess[min_ess]), " (", names(ess_is)[min_ess], ")", sep = "")
    
    cat("\n\nNote: The input is based on a IS-weighted MCMC, so the ", 
      "approximate (non-weighted) posterior is used when computing the Rhat ",
      "and ESS measures below.\n", sep="")
  }
  
  sumr <- summarise_draws(draws, default_convergence_measures())
  min_ess <- which.min(sumr$ess_bulk)
  cat("\nSmallest bulk-ESS: ", round(sumr$ess_bulk[min_ess]), " (", 
    sumr$variable[min_ess], ")", sep = "")
  min_ess <- which.min(sumr$ess_tail)
  cat("\nSmallest tail-ESS: ", round(sumr$ess_tail[min_ess]), " (", 
    sumr$variable[min_ess], ")", sep = "")
  max_rhat <- which.max(sumr$rhat)
  cat("\nLargest Rhat: ", round(sumr$rhat[max_rhat], 3), " (", 
    sumr$variable[max_rhat], ")", sep = "")
  invisible(x)
}
