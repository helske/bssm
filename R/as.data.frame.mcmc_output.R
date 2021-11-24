#' Convert MCMC Output to data.frame
#'
#' Converts the MCMC output of \code{\link{run_mcmc}} to \code{data.frame}.
#'  
#' @method as.data.frame mcmc_output
#' @param x Object of class \code{mcmc_output} from \code{\link{run_mcmc}}.
#' @param row.names Ignored.
#' @param optional Ignored.
#' @param variable Return samples of \code{"theta"} (default) or 
#' \code{"states"}?
#' @param times A vector of indices. In case of states, 
#' what time points to return? Default is all.
#' @param states A vector of indices. In case of states, 
#' what states to return? Default is all.
#' @param expand Should the jump-chain be expanded? 
#' Defaults to \code{TRUE}. 
#' For \code{expand = FALSE} and always for IS-MCMC, 
#' the resulting data.frame contains variable weight (= counts * IS-weights).
#' @param use_times If \code{TRUE} (default), transforms the values of the time 
#' variable to match the ts attribute of the input to define. If \code{FALSE}, 
#' time is based on the indexing starting from 1.
#' @param ... Ignored.
#' @seealso \code{as_draws} which converts the output for 
#' \code{as_draws} object.
#' @export
#' @examples
#' data("poisson_series")
#' model <- bsm_ng(y = poisson_series, 
#' sd_slope = halfnormal(0.1, 0.1), 
#' sd_level = halfnormal(0.1, 1),
#'   distribution = "poisson")
#'   
#' out <- run_mcmc(model, iter = 2000, particles = 10)
#' head(as.data.frame(out, variable = "theta"))
#' head(as.data.frame(out, variable = "state"))
#' 
#' # don't expand the jump chain:
#' head(as.data.frame(out, variable = "theta", expand = FALSE))
#' 
#' # IS-weighted version:
#' out_is <- run_mcmc(model, iter = 2000, particles = 10, 
#'   mcmc_type  = "is2")
#' head(as.data.frame(out_is, variable = "theta"))
#' 
as.data.frame.mcmc_output <- function(x, 
  row.names, optional,
  variable = c("theta", "states"),
  times, states,
  expand = TRUE, 
  use_times = TRUE, ...) {
  
  variable <- match.arg(tolower(variable), c("theta", "states"))
  
  if (variable == "theta") {
    if (expand) {
      values <- suppressWarnings(expand_sample(x, "theta"))
      iters <- seq(x$burnin + 1, x$iter, by = x$thin)
      weights <- if (x$mcmc_type %in% paste0("is", 1:3)) {
        rep(x$weights, times = x$counts) 
      } else 1
    } else {
      values <- x$theta
      iters <- x$burnin + cumsum(x$counts)
      weights <- 
        x$counts * (if (x$mcmc_type %in% paste0("is", 1:3)) x$weights else 1)
    }
    d <- data.frame(iter = iters,
      value = as.numeric(values),
      variable = rep(colnames(values), each = nrow(values)),
      weight = weights)
  } else {
    if (missing(times)) times <- seq_len(nrow(x$alpha))
    if (missing(states)) states <- seq_len(ncol(x$alpha))
    if (expand) {
      values <- aperm(x$alpha[times, states, 
        rep(seq_len(nrow(x$theta)), times = x$counts), drop = FALSE], 3:1)
      iters <- seq(x$burnin + 1, x$iter, by = x$thin)
      weights <-  if (x$mcmc_type %in% paste0("is", 1:3)) {
        rep(x$weights, times = x$counts) 
      } else 1
    } else {
      values <- aperm(x$alpha[times, states, , drop = FALSE], 3:1)
      iters <- x$burnin + cumsum(x$counts)
      weights <- x$counts * 
        (if (x$mcmc_type %in% paste0("is", 1:3)) x$weights else 1)
    }
    if (use_times) {
      times <- time(ts(seq_len(nrow(x$alpha)), 
        start = attr(x, "ts")$start, 
        frequency = attr(x, "ts")$frequency))[times]
    }
    d <- data.frame(iter = iters,
      value = as.numeric(values),
      variable = rep(colnames(x$alpha)[states], each = nrow(values)),
      time = rep(times, each = nrow(values) * ncol(values)),
      weight = weights)
  }
  d
}
