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
#' @importFrom stats logLik as.ts end frequency quantile start ts tsp tsp<- pnorm dnorm sd
#' @useDynLib bssm
NULL
