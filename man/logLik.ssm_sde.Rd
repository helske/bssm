% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/loglik.R
\name{logLik.ssm_sde}
\alias{logLik.ssm_sde}
\title{Log-likelihood of a State Space Model with SDE dynamics}
\usage{
\method{logLik}{ssm_sde}(
  object,
  particles,
  L,
  seed = sample(.Machine$integer.max, size = 1),
  ...
)
}
\arguments{
\item{object}{Model model.}

\item{particles}{Number of samples for particle filter.}

\item{L}{Integer  defining the discretization level defined as (2^L).}

\item{seed}{Seed for the random number generator.}

\item{...}{Ignored.}
}
\description{
Computes the log-likelihood of a state space model of class \code{ssm_sde} package.
}
