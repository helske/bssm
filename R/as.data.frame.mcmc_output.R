#' Convert MCMC chain to data.frame
#'
#' Converts the MCMC chain output of \code{\link{run_mcmc}} to data.frame.
#'  
#' @method as.data.frame mcmc_output
#' @param x Output from \code{\link{run_mcmc}}.
#' @param row.names Ignored.
#' @param optional Ignored.
#' @param variable Return samples of \code{"theta"} (default) or \code{"states"}?
#' @param times Vector of indices. In case of states, what time points to return? Default is all.
#' @param states Vector of indices. In case of states, what states to return? Default is all.
#' @param expand Should the jump-chain be expanded? 
#' Defaults to \code{TRUE} for non-IS-MCMC, and \code{FALSE} for IS-MCMC. 
#' For \code{expand = FALSE} and always for IS-MCMC, 
#' the resulting data.frame contains variable weight (= counts times IS-weights).
#' @param ... Ignored.
#' @export
as.data.frame.mcmc_output <- function(x, 
  row.names, optional,
  variable = c("theta", "states"),
  times, states,
  expand = !(x$mcmc_type %in% paste0("is", 1:3)), ...) {
  
  variable <- match.arg(variable, c("theta", "states"))
  
  if (variable == "theta") {
    if (expand) {
      values <- suppressWarnings(expand_sample(x, "theta"))
      iters <- seq(x$burnin + 1, x$iter, by = x$thin)
      weights <- if(x$mcmc_type %in% paste0("is", 1:3)) rep(x$weights, times = x$counts) else 1
    } else {
      values <- x$theta
      iters <- x$burnin + cumsum(x$counts)
      weights <- x$counts * (if(x$mcmc_type %in% paste0("is", 1:3)) x$weights else 1)
    }
    d <- data.frame(iter = iters,
      value = as.numeric(values),
      variable = rep(colnames(values), each = nrow(values)),
      weight = weights)
  } else {
    if (missing(times)) times <- 1:nrow(x$alpha)
    if (missing(states)) states <- 1:ncol(x$alpha)
    if (expand) {
      values <- aperm(x$alpha[times, states, rep(1:nrow(x$theta), times = x$counts), drop = FALSE], 3:1)
      iters <- seq(x$burnin + 1, x$iter, by = x$thin)
      weights <-  if(x$mcmc_type %in% paste0("is", 1:3)) rep(x$weights, times = x$counts) else 1
    } else {
      values <- aperm(x$alpha[times, states, , drop = FALSE], 3:1)
      iters <- x$burnin + cumsum(x$counts)
      weights <- x$counts * (if(x$mcmc_type %in% paste0("is", 1:3)) x$weights else 1)
    }
    times <- time(ts(1:nrow(x$alpha), 
      start = attr(x, "ts")$start, 
      frequency = attr(x, "ts")$frequency))[times]
    d <- data.frame(iter = iters,
      value = as.numeric(values),
      variable = rep(colnames(x$alpha)[states], each = nrow(values)),
      time = rep(times, each = nrow(values) * ncol(values)),
      weight = weights)
  }
  d
}

