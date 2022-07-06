#' Convert \code{run_mcmc} Output to \code{draws_df} Format
#'
#' Converts MCMC output from \code{run_mcmc} call to a
#' \code{draws_df} format of the \code{posterior} package. This enables the use
#' of diagnostics and plotting methods of \code{posterior} and \code{bayesplot}
#' packages.
#'
#' @note The jump chain representation is automatically expanded by
#' \code{as_draws}, but if \code{run_mcmc} used IS-MCMC method, the output
#' contains additional \code{weight} column corresponding to the IS-weights
#' (without counts), which is ignored by \code{posterior} and \code{bayesplot},
#' i.e. those results correspond to approximate MCMC.
#'
#' @param x An object of class \code{mcmc_output}.
#' @param times A vector of indices defining which time points to return?
#' Default is all. If 0, no samples for the states are extracted.
#' @param states A vector of indices defining which states to return.
#' Default is all. If 0, no samples for the states are extracted.
#' @param ... Ignored.
#' @return A \code{draws_df} object.
#' @importFrom posterior as_draws as_draws_df
#' @importFrom tidyr pivot_wider
#' @aliases as_draws as_draws_df
#' @export
#' @export as_draws_df
#' @rdname as_draws-mcmc_output
#' @method as_draws_df mcmc_output
#' @examples
#'
#' model <- bsm_lg(Nile,
#'   sd_y = tnormal(init = 100, mean = 100, sd = 100, min = 0),
#'   sd_level = tnormal(init = 50, mean = 50, sd = 100, min = 0),
#'   a1 = 1000, P1 = 500^2)
#'
#' fit1 <- run_mcmc(model, iter = 2000)
#' draws <- as_draws(fit1)
#' head(draws, 4)
#' estimate_ess(draws$sd_y)
#' summary(fit1, return_se = TRUE)
#'
#' # More chains:
#' model$theta[] <- c(50, 150) # change initial value
#' fit2 <- run_mcmc(model, iter = 2000, verbose = FALSE)
#' model$theta[] <- c(150, 50) # change initial value
#' fit3 <- run_mcmc(model, iter = 2000, verbose = FALSE)
#'
#' # it is actually enough to transform first mcmc_output to draws object,
#' # rest are transformed automatically inside bind_draws
#' draws <- posterior::bind_draws(as_draws(fit1),
#'   as_draws(fit2), as_draws(fit3), along = "chain")
#'
#' posterior::rhat(draws$sd_y)
#'
as_draws_df.mcmc_output <- function(x, times, states, ...) {

  d_theta <- as.data.frame(x, variable = "theta", expand = TRUE)

  if (missing(times)) {
    times <- seq_len(nrow(x$alpha))
  } else {
    if (!identical(times, 0)) {
      if (!test_integerish(times, lower = 1, upper = nrow(x$alpha),
        any.missing = FALSE, unique = TRUE)) {
        stop("Argument 'times' should contain indices between 1 and ",
          nrow(x$alpha), ", or it should be a scalar 0.")
      }
    }
  }
  if (missing(states)) {
    states <- seq_len(ncol(x$alpha))
  } else {
    if (!identical(states, 0)) {
      if (!test_integerish(states, lower = 1, upper = ncol(x$alpha),
        any.missing = FALSE, unique = TRUE))
        stop("Argument 'states' should contain indices between 1 and ",
          ncol(x$alpha)," or it should be a scalar 0.")
    }
  }
  if (identical(times, 0) || identical(states, 0)) {
    d <-
      tidyr::pivot_wider(d_theta,
        values_from = .data$value,
        names_from = .data$variable)
  } else {
    d_states <- as.data.frame(x, variable = "states", expand = TRUE,
      times = times, states = states, use_times = FALSE)
    d <- cbind(
      tidyr::pivot_wider(d_theta,
        values_from = .data$value,
        names_from = .data$variable),
      tidyr::pivot_wider(d_states,
        values_from = .data$value,
        names_from = c(.data$variable, .data$time),
        names_glue = "{variable}[{time}]")[, -(1:2)])
  }

  names(d)[1] <- ".iteration"

  if (x$mcmc_type %in% paste0("is", 1:3)) {
    warning(paste("Input is based on a IS-MCMC and the output column 'weight'",
      "contains the IS-weights. These are not used for example in the",
      "diagnostic methods by 'posterior' package, i.e. these are based",
      "on approximate MCMC chains."))
  } else {
    d$weight <- NULL
  }

  as_draws(d)
}
#' @export
#' @export as_draws
#' @rdname as_draws-mcmc_output
#' @method as_draws mcmc_output
as_draws.mcmc_output <- function(x, times, states, ...) {
  as_draws_df.mcmc_output(x, times, states, ...)
}
