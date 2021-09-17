#' Convert \code{run_mcmc} output to \code{draws_df} format
#'
#' Converts MCMC output from \code{run_mcmc} call to a 
#' \code{draws_df} format of the \code{posterior} package. This enables the use 
#' of diagnostics and plotting methods of \code{posterior} and \code{bayesplot} 
#' packages. Note though that if \code{run_mcmc} used IS-MCMC 
#' method, the resulting \code{weight} column of the output is 
#' ignored by the aforementioned packages, i.e. the results correspond to 
#' approximate MCMC.
#' 
#' @param x An object of class \code{mcmc_output}
#' @return A \code{draws_df} object.
#' @exportS3Method posterior::as_draws_df mcmc_output
#' @export
#' @rdname as_draws
#' @examples 
#' 
#' model <- bsm_lg(Nile, 
#'   sd_y = tnormal(init = 100, mean = 100, sd = 100, min = 0),
#'   sd_level = tnormal(init = 50, mean = 50, sd = 100, min = 0),
#'   a1 = 1000, P1 = 500^2)
#' 
#' fit1 <- run_mcmc(model, iter = 2e4)
#' library("posterior")
#' draws <- as_draws(fit1)
#' head(draws, 4)
#' ess_bulk(draws$sd_y)
#' summary(fit1, return_se = TRUE)
#' 
#' # More chains:
#' model$theta[] <- c(50, 150) # change initial value
#' fit2 <- run_mcmc(model, iter = 2e4)
#' model$theta[] <- c(150, 50) # change initial value
#' fit3 <- run_mcmc(model, iter = 2e4)
#' 
#' draws <- bind_draws(as_draws(fit1),
#'   as_draws(fit2), as_draws(fit3), along = "chain")
#' # it is actually enough to transform first mcmc_output to draws object, 
#' # rest are transformed automatically inside bind_draws
#' rhat(draws$sd_y)
#' ess_bulk(draws$sd_y)
#' ess_tail(draws$sd_y)
#' 
as_draws_df.mcmc_output <- function(x) {
  
  d_theta <- as.data.frame(x, variable = "theta", expand = TRUE)
  d_states <- as.data.frame(x, variable = "states", expand = TRUE)
  
  d <- data.frame(.iteration = (x$iter - x$burnin + 1):x$iter)
  
  if (x$mcmc_type %in% paste0("is", 1:3)) {
    warning(paste("Input is based on a IS-MCMC, the output column 'weight'", 
      "contains the IS-weights, but these are not used for example in the", 
      "diagnostic methods by 'posterior' package, i.e. these are based",
      "on approximate MCMC chains."))
    d$weight <- d_theta$weight
  }
  
  for (variable in unique(d_theta$variable)) {
    d[variable] <- d_theta$value[d_theta$variable == variable]
  }
  times <- unique(d_states$time)
  for (variable in unique(d_states$variable)) {
    for (i in times)
      d[paste0(variable, "[", i, "]")] <- 
        d_states$value[d_states$variable == variable & d_states$time == i]
  }
  posterior::as_draws(d)
}
#' @exportS3Method posterior::as_draws mcmc_output
#' @export
#' @rdname as_draws
as_draws.mcmc_output <- function(x) as_draws_df.mcmc_output(x)