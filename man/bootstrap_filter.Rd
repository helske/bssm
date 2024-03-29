% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/bootstrap_filter.R
\name{bootstrap_filter}
\alias{bootstrap_filter}
\alias{bootstrap_filter.lineargaussian}
\alias{bootstrap_filter.nongaussian}
\alias{bootstrap_filter.ssm_nlg}
\alias{bootstrap_filter.ssm_sde}
\title{Bootstrap Filtering}
\usage{
bootstrap_filter(model, particles, ...)

\method{bootstrap_filter}{lineargaussian}(
  model,
  particles,
  seed = sample(.Machine$integer.max, size = 1),
  ...
)

\method{bootstrap_filter}{nongaussian}(
  model,
  particles,
  seed = sample(.Machine$integer.max, size = 1),
  ...
)

\method{bootstrap_filter}{ssm_nlg}(
  model,
  particles,
  seed = sample(.Machine$integer.max, size = 1),
  ...
)

\method{bootstrap_filter}{ssm_sde}(
  model,
  particles,
  L,
  seed = sample(.Machine$integer.max, size = 1),
  ...
)
}
\arguments{
\item{model}{A model object of class \code{bssm_model}.}

\item{particles}{Number of particles as a positive integer. Suitable values
depend on the model and the data, and while larger values provide more
accurate estimates, the run time also increases with respect to the
number of particles, so it is generally a good idea to test the filter first
with a small number of particles, e.g., less than 100.}

\item{...}{Ignored.}

\item{seed}{Seed for the C++ RNG (positive integer).}

\item{L}{Positive integer defining the discretization level for SDE models.}
}
\value{
List with samples (\code{alpha}) from the filtering distribution and
corresponding weights (\code{weights}), as well as filtered and predicted
states and corresponding covariances (\code{at}, \code{att}, \code{Pt},
\code{Ptt}), and estimated log-likelihood (\code{logLik}).
}
\description{
Function \code{bootstrap_filter} performs a bootstrap filtering with
stratification resampling.
}
\examples{
set.seed(1)
x <- cumsum(rnorm(50))
y <- rnorm(50, x, 0.5) 
model <- bsm_lg(y, sd_y = 0.5, sd_level = 1, P1 = 1)
  
out <- bootstrap_filter(model, particles = 1000)
ts.plot(cbind(y, x, out$att), col = 1:3)
ts.plot(cbind(kfilter(model)$att, out$att), col = 1:3)

data("poisson_series")
model <- bsm_ng(poisson_series, sd_level = 0.1, sd_slope = 0.01, 
  P1 = diag(1, 2), distribution = "poisson")
  
out <- bootstrap_filter(model, particles = 100)
ts.plot(cbind(poisson_series, exp(out$att[, 1])), col = 1:2)

}
\references{
Gordon, NJ, Salmond, DJ, Smith, AFM (1993) Novel approach to
nonlinear/non-Gaussian Bayesian state estimation. IEE Proceedings F,
140(2), p. 107-113.
}
