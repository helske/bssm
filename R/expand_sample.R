
#' Expand the Jump Chain representation
#'
#' The MCMC algorithms of \code{bssm} use a jump chain representation where we 
#' store the accepted values and the number of times we stayed in the current 
#' value. Although this saves bit memory and is especially convenient for 
#' IS-corrected  MCMC, sometimes we want to have the usual sample paths 
#' (for example for drawing traceplots). 
#' Function \code{expand_sample} returns the expanded sample based on the 
#' counts (in form of \code{coda::mcmc} object. Note that for 
#' the IS-MCMC the expanded sample corresponds to the approximate posterior,
#' i.e., the weights are ignored.
#' 
#' This functions is mostly for backwards compatibility, methods 
#' \code{as.data.frame} and \code{as_draws} produce likely more convenient 
#' output.
#' 
#' @importFrom coda mcmc
#' @param x Output from \code{\link{run_mcmc}}.
#' @param variable Expand parameters \code{"theta"} or states \code{"states"}.
#' @param times A vector of indices. In case of states, 
#' what time points to expand? Default is all.
#' @param states A vector of indices. In case of states, 
#' what states to expand? Default is all.
#' @param by_states If \code{TRUE} (default), return list by states. 
#' Otherwise by time.
#' @return An object of class \code{"mcmc"} of the \code{coda} package.
#' @seealso \code{as.data.frame.mcmc_output} and \code{as_draws.mcmc_output}.
#' @export
#' @examples
#' set.seed(1)
#' n <- 50
#' x <- cumsum(rnorm(n))
#' y <- rnorm(n, x)
#' model <- bsm_lg(y, sd_y = gamma_prior(1, 2, 2), 
#'   sd_level = gamma_prior(1, 2, 2))
#' fit <- run_mcmc(model, iter = 1e4)
#' # Traceplots for theta
#' plot.ts(expand_sample(fit, variable = "theta"))
#' # Traceplot for x_5
#' plot.ts(expand_sample(fit, variable = "states", times = 5, 
#'   states = 1)$level)
expand_sample <- function(x, variable = "theta", times, states, 
  by_states = TRUE) {
  
  if (!test_flag(by_states)) 
    stop("Argument 'by_states' should be TRUE or FALSE. ")
  
  variable <- match.arg(tolower(variable), c("theta", "states"))
  if (x$mcmc_type %in% paste0("is", 1:3)) 
    warning(paste("Input is based on a IS-weighted MCMC, the results", 
      "correspond to the approximate posteriors.", sep = " "))
  
  if (variable == "theta") {
    out <- apply(x$theta, 2, rep, times = x$counts)
  } else {
    if (x$output_type == 1) {
      if (missing(times)) {
        times <- seq_len(nrow(x$alpha))
      } else {
        if (!test_integerish(times, lower = 1, upper = nrow(x$alpha), 
          any.missing = FALSE, unique = TRUE))
          stop(paste0("Argument 'times' should contain indices between 1 and ",
          nrow(x$alpha),"."))
      }
      if (missing(states)) {
        states <- seq_len(ncol(x$alpha))
      } else {
        if (!test_integerish(states, lower = 1, upper = ncol(x$alpha), 
          any.missing = FALSE, unique = TRUE))
          stop(paste0("Argument 'states' should contain indices between 1 and ",
            ncol(x$alpha),"."))
      }
      
      if (by_states) {
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
