% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/models.R
\name{ssm_ung}
\alias{ssm_ung}
\title{General univariate non-Gaussian state space model}
\usage{
ssm_ung(
  y,
  Z,
  T,
  R,
  a1 = NULL,
  P1 = NULL,
  distribution,
  phi = 1,
  u,
  init_theta = numeric(0),
  D = NULL,
  C = NULL,
  state_names,
  update_fn = default_update_fn,
  prior_fn = default_prior_fn
)
}
\arguments{
\item{y}{Observations as time series (or vector) of length \eqn{n}.}

\item{Z}{System matrix Z of the observation equation. Either a
vector of length m,
a m x n matrix, or object which can be coerced to such.}

\item{T}{System matrix T of the state equation. Either a m x m matrix or a
m x m x n array, or object which can be coerced to such.}

\item{R}{Lower triangular matrix R the state equation. Either
a m x k matrix or a m x k x n array, or object which can be coerced to such.}

\item{a1}{Prior mean for the initial state as a vector of length m.}

\item{P1}{Prior covariance matrix for the initial state as m x m matrix.}

\item{distribution}{Distribution of the observed time series. Possible
choices are \code{"poisson"}, \code{"binomial"}, \code{"gamma"}, and
\code{"negative binomial"}.}

\item{phi}{Additional parameter relating to the non-Gaussian distribution.
For negative binomial distribution this is the dispersion term, for gamma
distribution this is the shape parameter, and for other distributions this
is ignored. Should an object of class \code{bssm_prior} or
a positive scalar.}

\item{u}{A vector of positive constants for non-Gaussian models. For
Poisson, gamma, and negative binomial distribution, this corresponds to the
offset term. For binomial, this is the number of trials.}

\item{init_theta}{Initial values for the unknown hyperparameters theta
(i.e. unknown variables excluding latent state variables).}

\item{D}{Intercept terms \eqn{D_t} for the observations equation, given as a
scalar or vector of length n.}

\item{C}{Intercept terms \eqn{C_t} for the state equation, given as a
m times 1 or m times n matrix.}

\item{state_names}{A character vector defining the names of the states.}

\item{update_fn}{A function which returns list of updated model
components given input vector theta. This function should take only one
vector argument which is used to create list with elements named as
\code{Z}, \code{T}, \code{R}, \code{a1}, \code{P1}, \code{D}, \code{C}, and
\code{phi}, where each element matches the dimensions of the original model.
If any of these components is missing, it is assumed to be constant wrt.
theta. It's best to check the internal dimensions with
\code{str(model_object)} as the dimensions of input arguments can differ
from the final dimensions.}

\item{prior_fn}{A function which returns log of prior density
given input vector theta.}
}
\value{
An object of class \code{ssm_ung}.
}
\description{
Construct an object of class \code{ssm_ung} by directly defining the
corresponding terms of the model.
}
\details{
The general univariate non-Gaussian model is defined using the following
observational and state equations:

\deqn{p(y_t | D_t + Z_t \alpha_t), (\textrm{observation equation})}
\deqn{\alpha_{t+1} = C_t + T_t \alpha_t + R_t \eta_t, 
(\textrm{transition equation})}

where \eqn{\eta_t \sim N(0, I_k)} and
\eqn{\alpha_1 \sim N(a_1, P_1)} independently of each other,
and \eqn{p(y_t | .)} is either Poisson, binomial, gamma, or
negative binomial distribution.
Here k is the number of disturbance terms which can be less than m,
the number of states.

The \code{update_fn} function should take only one
vector argument which is used to create list with elements named as
\code{Z}, \code{phi} \code{T}, \code{R}, \code{a1}, \code{P1}, \code{D},
and \code{C},
where each element matches the dimensions of the original model.
If any of these components is missing, it is assumed to be constant
wrt. theta.
Note that while you can input say R as m x k matrix for \code{ssm_ung},
\code{update_fn} should return R as m x k x 1 in this case.
It might be useful to first construct the model without updating function
and then check the expected structure of the model components from
the output.
}
\examples{

data("drownings", package = "bssm")
model <- ssm_ung(drownings[, "deaths"], Z = 1, T = 1, R = 0.2, 
  a1 = 0, P1 = 10, distribution = "poisson", u = drownings[, "population"])

# approximate results based on Gaussian approximation
out <- smoother(model)
ts.plot(cbind(model$y / model$u, exp(out$alphahat)), col = 1:2)
}
