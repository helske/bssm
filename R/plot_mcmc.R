#' Trace and Density Plots for `mcmc_output`
#'
#' Plots the trace and density plots of the hyperparameters theta from the MCMC
#' run by \code{\link{run_mcmc}}.
#'
#' For further visualization (of the states), you can extract the posterior
#' samples with `as.data.frame` and `as_draws` methods to be used for example
#' with the `bayesplot` or `ggplot2` packages.
#'
#'
#' @note For IS-MCMC, these plots correspond to the approximate (non-weighted)
#' samples
#' .
#' @method plot mcmc_output
#' @importFrom bayesplot mcmc_combo
#' @param x Object of class \code{mcmc_output} from \code{\link{run_mcmc}}.
#' @param ... Further arguments to [bayesplot::mcmc_combo].
#' @return The output object from [bayesplot::mcmc_combo].
#' @seealso \code{\link{check_diagnostics}} for a quick diagnostics statistics
#' of the model.
#' @export
#' @examples
#' data("negbin_model")
#' # Note the very small number of iterations, so the plots look bad
#' plot(negbin_model)
plot.mcmc_output <- function(x, ...) {

    # suppress the duplicate warning about the IS-MCMC
    out <- suppressWarnings(as_draws(x, states = 0))

    if (x$mcmc_type %in% paste0("is", 1:3)) {
        warning("Input is based on a IS-weighted MCMC, the plots ",
                "correspond to the approximate MCMC.")
        # remove the weight variable
        out <- out[, -1]
    }
    mcmc_combo(out, ...)
}
