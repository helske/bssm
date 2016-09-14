#' Bayesian Inference of State Space Models
#'
#' This package contains functions for Bayesian inference of basic stochastic volatility model
#' and exponential family state space models, where the state equation is linear and Gaussian,
#' and the conditional observation density is either Gaussian, Poisson,
#' binomial, negative binomial or Gamma density. For formal definition of the
#' currently supported models and methods, as well as theory, see the package vignette 
#' and arXiv paper: \url{http://arxiv.org/abs/1609.02541}.
#'
#' @docType package
#' @name bssm
#' @aliases bssm
#' @importFrom Rcpp evalCpp
#' @importFrom coda mcmc effectiveSize thin
#' @importFrom ggplot2  aes aes_string scale_x_continuous scale_y_continuous
#' @importFrom stats as.ts dnorm  end frequency is.ts logLik pnorm quantile start time ts ts.union tsp tsp<- sd
#' @useDynLib bssm
NULL
#' Deaths by drowning in Finland in 1969-2014
#'
#' Dataset containing number of deaths by drowning in Finland in 1969-2014,
#' yearly average summer temperatures (June to August) and
#' corresponding population sizes (in hundreds of thousands).
#'
#' @name drownings
#' @docType data
#' @format A time series object containing 46 observations and.
#' @source Statistics Finland \url{http://pxnet2.stat.fi/PXWeb/pxweb/en/StatFin/}.
#' @keywords datasets
NULL
#' Pound/Dollar daily exchange rates
#'
#' Dataset containing daily log-returns from 1/10/81-28/6/85 as in [1]
#'
#' @name exchange
#' @docType data
#' @format A vector of length 945.
#' @source \url{http://www.ssfpack.com/DKbook.html}.
#' @keywords datasets
#' @references Durbin, Koopman (2012), "Time Series Analysis by State Space Methods".
NULL
#' Simulated Poisson time series data
#'
#' See example for code for reproducing the data.
#'
#' @name poisson_series
#' @docType data
#' @format A vector of length 100
#' @keywords datasets
#' @examples 
#' # The data is generated as follows:
#' set.seed(321)
#' slope <- cumsum(c(0, rnorm(99, sd = 0.01)))
#' y <- rpois(100, exp(cumsum(slope + c(0, rnorm(99, sd = 0.1)))))
NULL