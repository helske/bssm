#' Bayesian Inference of Exponential Family State Space Models
#'
#' This package contains functions for Bayesian inference of exponential
#' family state space models, where the state equation is linear and Gaussian,
#' and the conditional observation density is either Gaussian, Poisson,
#' binomial, negative binomial or Gamma density. For formal definition of the
#' currently supported models and methods, see the package vignette.
#'
#' @docType package
#' @name bssm
#' @aliases bssm
#' @importFrom Rcpp evalCpp
#' @importFrom coda mcmc effectiveSize
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
