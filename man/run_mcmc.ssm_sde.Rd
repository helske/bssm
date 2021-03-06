% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/run_mcmc.R
\name{run_mcmc.ssm_sde}
\alias{run_mcmc.ssm_sde}
\title{Bayesian Inference of SDE}
\usage{
\method{run_mcmc}{ssm_sde}(
  model,
  iter,
  particles,
  output_type = "full",
  mcmc_type = "is2",
  L_c,
  L_f,
  burnin = floor(iter/2),
  thin = 1,
  gamma = 2/3,
  target_acceptance = 0.234,
  S,
  end_adaptive_phase = FALSE,
  threads = 1,
  seed = sample(.Machine$integer.max, size = 1),
  ...
)
}
\arguments{
\item{model}{Model model.}

\item{iter}{Number of MCMC iterations.}

\item{particles}{Number of state samples per MCMC iteration.}

\item{output_type}{Either \code{"full"} 
(default, returns posterior samples of states alpha and hyperparameters theta), 
\code{"theta"} (for marginal posterior of theta), 
or \code{"summary"} (return the mean and variance estimates of the states 
and posterior samples of theta). In case of \code{"summary"}, means and 
covariances are computed using the full output of particle filter 
instead of sampling one of these as in case of \code{output_type = "full"}. 
If \code{particles = 0}, this is argument ignored and set to \code{"theta"}.}

\item{mcmc_type}{What MCMC algorithm to use? Possible choices are
\code{"pm"} for pseudo-marginal MCMC,
\code{"da"} for delayed acceptance version of pseudo-marginal MCMC, 
or one of the three importance sampling type weighting schemes:
\code{"is3"} for simple importance sampling (weight is computed for each MCMC iteration independently),
\code{"is2"} for jump chain importance sampling type weighting (default), or
\code{"is1"} for importance sampling type weighting where the number of particles used for
weight computations is proportional to the length of the jump chain block.}

\item{L_c, L_f}{Integer values defining the discretization levels for first and second stages (defined as 2^L). 
For PM methods, maximum of these is used.}

\item{burnin}{Length of the burn-in period which is disregarded from the
results. Defaults to \code{iter / 2}.}

\item{thin}{Thinning rate. Defaults to 1. Increase for large models in
order to save memory. For IS-corrected methods, larger
value can also be statistically more effective. 
Note: With \code{output_type = "summary"}, the thinning does not affect the computations 
of the summary statistics in case of pseudo-marginal methods.}

\item{gamma}{Tuning parameter for the adaptation of RAM algorithm. Must be
between 0 and 1 (not checked).}

\item{target_acceptance}{Target acceptance ratio for RAM. Defaults to 0.234.
For DA-MCMC, this corresponds to first stage acceptance rate, i.e., the total acceptance 
rate will be smaller.}

\item{S}{Initial value for the lower triangular matrix of RAM
algorithm, so that the covariance matrix of the Gaussian proposal
distribution is \eqn{SS'}. Note that for some parameters 
(currently the standard deviation and dispersion parameters of bsm_ng models) the sampling
is done for transformed parameters with internal_theta = log(theta).}

\item{end_adaptive_phase}{If \code{TRUE}, S is held fixed after the burnin period. Default is \code{FALSE}.}

\item{threads}{Number of threads for state simulation.}

\item{seed}{Seed for the random number generator.}

\item{...}{Ignored.}
}
\description{
Methods for posterior inference of states and parameters.
}
\references{
Vihola, M, Helske, J, Franks, J. Importance sampling type estimators based on approximate marginal Markov chain Monte Carlo. 
Scand J Statist. 2020; 1– 38. https://doi.org/10.1111/sjos.12492
}
