#' Summary Statistics of Posterior Samples
#' 
#' This functions returns a data frame containing mean, standard deviations, 
#' standard errors, and effective sample size estimates for parameters and 
#' states.
#' 
#' For IS-MCMC two types of standard errors are reported. 
#' SE-IS can be regarded as the square root of independent IS variance,
#' whereas SE corresponds to the square root of total asymptotic variance 
#' (see Remark 3 of Vihola et al. (2020)).
#' 
#' @importFrom rlang .data
#' @param object Output from \code{run_mcmc}
#' @param variable Are the summary statistics computed for either 
#' \code{"theta"} (default), \code{"states"}, or \code{"both"}?
#' @param return_se if \code{FALSE} (default), computation of standard 
#' errors and effective sample sizes is omitted (as they can take considerable 
#' time for models with large number of states and time points).
#' @param probs Numeric vector defining the quantiles of interest. Default is 
#' \code{c(0.025, 0.975)}.
#' @param times Vector of indices. For states, for what time points the 
#' summaries should be computed? Default is all, ignored if 
#' \code{variable = "theta"}.
#' @param states Vector of indices. For what states the summaries should be 
#' computed?. Default is all, ignored if 
#' \code{variable = "theta"}.
#' @param method Method for computing integrated autocorrelation time. Default 
#' is \code{"sokal"}, other option is \code{"geyer"}.
#' @param use_times If \code{TRUE} (default), transforms the values of the time 
#' variable to match the ts attribute of the input to define. If \code{FALSE}, 
#' time is based on the indexing starting from 1.
#' @param ... Ignored.
#' @return If \code{variable} is \code{"theta"} or \code{"states"}, a 
#' \code{data.frame} object. If \code{"both"}, a list of two data frames.
#' @references 
#' Vihola, M, Helske, J, Franks, J. Importance sampling type estimators based 
#' on approximate marginal Markov chain Monte Carlo. 
#' Scand J Statist. 2020; 1-38. https://doi.org/10.1111/sjos.12492
#' @export
#' @srrstats {BS5.3, BS5.5, BS6.4}
summary.mcmc_output <- function(object, return_se = FALSE, variable = "theta", 
  probs = c(0.025, 0.975), times, states, use_times = TRUE, method = "sokal", 
  ...) {
  
  if (!test_flag(return_se)) 
    stop("Argument 'return_se' should be TRUE or FALSE. ")
  
  method <- match.arg(method, c("sokal", "geyer"))
  
  variable <- match.arg(tolower(variable), c("theta", "states", "both"))
  
  if (return_se) {
    if (object$mcmc_type %in% paste0("is", 1:3)) {
      summary_f <- function(x, w) {
        c(Mean = weighted_mean(x, w), 
          SE = sqrt(asymptotic_var(x, w, method)),
          SD = sqrt(weighted_var(x, w)), 
          weighted_quantile(x, w, probs), 
          ESS = round(estimate_ess(x, w, method)),
          SE_IS = weighted_se(x, w),
          ESS_IS = round(ess(w, identity, x)))
      }
    } else {
      summary_f <- function(x, w) {
        c(Mean = mean(x), SE = sqrt(asymptotic_var(x, method = method)),
          SD = sd(x), quantile(x, probs), 
          ESS = round(estimate_ess(x, method = method)))
      }
    }
  } else {
    if (object$mcmc_type %in% paste0("is", 1:3)) {
      summary_f <- function(x, w) {
        c(Mean = weighted_mean(x, w), 
          SD = sqrt(weighted_var(x, w)), 
          weighted_quantile(x, w, probs))
      }
    } else {
      summary_f <- function(x, w) {
        c(Mean = mean(x),
          SD = sd(x), quantile(x, probs))
      }
    }
  }
  if (variable %in% c("theta", "both")) {
    sumr_theta <- 
      as.data.frame(object, variable = "theta", expand = TRUE) %>%
      group_by(variable) %>% 
      summarise(as_tibble(as.list(summary_f(value, weight))))
    if (variable == "theta") return(sumr_theta)
  }
  
  if (variable %in% c("states", "both")) {
    if (object$output_type != 1) 
      stop("Cannot return summary of states as the MCMC type was not 'full'. ")
    
    if (missing(times)) {
      times <- seq_len(nrow(object$alpha))
    } else {
      if (!test_integerish(times, lower = 1, upper = nrow(object$alpha), 
        any.missing = FALSE, unique = TRUE))
        stop(paste0("Argument 'times' should contain indices between 1 and ",
          nrow(object$alpha),"."))
    }
    if (missing(states)) {
      states <- seq_len(ncol(object$alpha))
    } else {
      if (!test_integerish(states, lower = 1, upper = ncol(object$alpha), 
        any.missing = FALSE, unique = TRUE))
        stop(paste0("Argument 'states' should contain indices between 1 and ",
          ncol(object$alpha),"."))
    }
    
    sumr_states <- 
      as.data.frame(object, variable = "states", expand = TRUE, 
        times = times, states = states, use_times = use_times) %>%
      group_by(variable, time) %>% 
      summarise(as_tibble(as.list(summary_f(value, weight))))
    if (variable == "states") return(sumr_states)
  }
  list(theta = sumr_theta, states = sumr_states)
}

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
}
