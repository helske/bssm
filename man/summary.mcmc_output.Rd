% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/summary.R
\name{summary.mcmc_output}
\alias{summary.mcmc_output}
\title{Summary Statistics of Posterior Samples}
\usage{
\method{summary}{mcmc_output}(
  object,
  return_se = FALSE,
  variable = "theta",
  probs = c(0.025, 0.975),
  times,
  states,
  use_times = TRUE,
  method = "sokal",
  ...
)
}
\arguments{
\item{object}{Output from \code{run_mcmc}}

\item{return_se}{if \code{FALSE} (default), computation of standard
errors and effective sample sizes is omitted (as they can take considerable
time for models with large number of states and time points).}

\item{variable}{Are the summary statistics computed for either
\code{"theta"} (default), \code{"states"}, or \code{"both"}?}

\item{probs}{A numeric vector defining the quantiles of interest. Default is
\code{c(0.025, 0.975)}.}

\item{times}{A vector of indices. For states, for what time points the
summaries should be computed? Default is all, ignored if
\code{variable = "theta"}.}

\item{states}{A vector of indices. For what states the summaries should be
computed?. Default is all, ignored if
\code{variable = "theta"}.}

\item{use_times}{If \code{TRUE} (default), transforms the values of the time
variable to match the ts attribute of the input to define. If \code{FALSE},
time is based on the indexing starting from 1.}

\item{method}{Method for computing integrated autocorrelation time. Default
is \code{"sokal"}, other option is \code{"geyer"}.}

\item{...}{Ignored.}
}
\value{
If \code{variable} is \code{"theta"} or \code{"states"}, a
\code{data.frame} object. If \code{"both"}, a list of two data frames.
}
\description{
This functions returns a data frame containing mean, standard deviations,
standard errors, and effective sample size estimates for parameters and
states.
}
\details{
For IS-MCMC two types of standard errors are reported.
SE-IS can be regarded as the square root of independent IS variance,
whereas SE corresponds to the square root of total asymptotic variance
(see Remark 3 of Vihola et al. (2020)).
}
\examples{
data("negbin_model")
summary(negbin_model, return_se = TRUE, method = "geyer")
summary(negbin_model, times = c(1, 200), prob = c(0.05, 0.5, 0.95))
}
\references{
Vihola, M, Helske, J, Franks, J. Importance sampling type estimators based
on approximate marginal Markov chain Monte Carlo.
Scand J Statist. 2020; 1-38. https://doi.org/10.1111/sjos.12492
}
