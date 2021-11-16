#' Convert \code{run_mcmc} output to \code{draws_df} format
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
#' @param x An object of class \code{mcmc_output}
#' @return A \code{draws_df} object.
#' @rdname as_draws
#' @importFrom posterior as_draws as_draws_df
#' @importFrom tidyr pivot_wider
#' @exportS3Method posterior::as_draws_df mcmc_output
#' @examples 
#' 
#' model <- bsm_lg(Nile, 
#'   sd_y = tnormal(init = 100, mean = 100, sd = 100, min = 0),
#'   sd_level = tnormal(init = 50, mean = 50, sd = 100, min = 0),
#'   a1 = 1000, P1 = 500^2)
#' 
#' fit1 <- run_mcmc(model, iter = 2000)
#' library("posterior")
#' draws <- as_draws(fit1)
#' head(draws, 4)
#' ess_bulk(draws$sd_y)
#' summary(fit1, return_se = TRUE)
#' 
#' # More chains:
#' model$theta[] <- c(50, 150) # change initial value
#' fit2 <- run_mcmc(model, iter = 2000)
#' model$theta[] <- c(150, 50) # change initial value
#' fit3 <- run_mcmc(model, iter = 2000)
#' 
#' draws <- bind_draws(as_draws(fit1),
#'   as_draws(fit2), as_draws(fit3), along = "chain")
#' # it is actually enough to transform first mcmc_output to draws object, 
#' # rest are transformed automatically inside bind_draws
#' posterior::rhat(draws$sd_y)
#' posterior::ess_bulk(draws$sd_y)
#' posterior::summarise_draws(draws)
#'
as_draws_df.mcmc_output <- function(x) {
  
  
  d_theta <- as.data.frame(x, variable = "theta", expand = TRUE)
  d_states <- as.data.frame(x, variable = "states", expand = TRUE, 
    use_times = FALSE)
  
  d <- merge(
    tidyr::pivot_wider(d_theta, 
      values_from = value, 
      names_from = variable),
    tidyr::pivot_wider(d_states, 
      values_from = value, 
      names_from = c(variable, time), 
      names_glue = "{variable}[{time}]"))
  names(d)[1] <- ".iteration"
  
  if (x$mcmc_type %in% paste0("is", 1:3)) {
    warning(paste("Input is based on a IS-MCMC, the output column '.weight'", 
      "contains the IS-weights, but these are not used for example in the", 
      "diagnostic methods by 'posterior' package, i.e. these are based",
      "on approximate MCMC chains."))
    names(d)[2] <- ".weight"
  } else {
    d$weight <- NULL
  }
  
  as_draws(d)
}
#' @exportS3Method posterior::as_draws mcmc_output
as_draws.mcmc_output <- function(x) as_draws_df.mcmc_output(x)