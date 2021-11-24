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
#' @param probs A numeric vector defining the quantiles of interest. Default is 
#' \code{c(0.025, 0.975)}.
#' @param times A vector of indices. For states, for what time points the 
#' summaries should be computed? Default is all, ignored if 
#' \code{variable = "theta"}.
#' @param states A vector of indices. For what states the summaries should be 
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
#' @examples
#' data("negbin_model")
#' summary(negbin_model, return_se = TRUE, method = "geyer")
#' summary(negbin_model, times = c(1, 200), prob = c(0.05, 0.5, 0.95))
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
      group_by(.data$variable) %>% 
      summarise(as_tibble(as.list(summary_f(.data$value, .data$weight)))) %>% 
      as.data.frame()
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
      group_by(.data$variable, .data$time) %>% 
      summarise(as_tibble(as.list(summary_f(.data$value, .data$weight)))) %>% 
      as.data.frame()
    if (variable == "states") return(sumr_states)
  }
  list(theta = sumr_theta, states = sumr_states)
}
