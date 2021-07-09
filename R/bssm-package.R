#' Bayesian Inference of State Space Models
#'
#' This package contains functions for Bayesian inference of basic stochastic volatility model
#' and exponential family state space models, where the state equation is linear and Gaussian,
#' and the conditional observation density is either Gaussian, Poisson,
#' binomial, negative binomial or Gamma density. General non-linear Gaussian models and models 
#' with continuous SDE dynamics are also supported. For formal definition of the
#' currently supported models and methods, as well as some theory behind the IS-MCMC and \eqn{\psi}{psi}-APF, 
#' see the package vignettes and Vihola, Helske, Franks (2020).
#' 
#' @references 
#' Vihola, M, Helske, J, Franks, J. Importance sampling type estimators based on approximate marginal Markov chain Monte Carlo. 
#' Scand J Statist. 2020; 1– 38. https://doi.org/10.1111/sjos.12492
#'
#' @docType package
#' @name bssm
#' @aliases bssm
#' @importFrom Rcpp evalCpp
#' @importFrom coda mcmc
#' @importFrom stats as.ts dnorm  end frequency is.ts logLik quantile start time ts ts.union tsp tsp<- sd na.omit
#' @useDynLib bssm
NULL
#' Deaths by drowning in Finland in 1969-2019
#'
#' Dataset containing number of deaths by drowning in Finland in 1969-2019,
#' corresponding population sizes (in hundreds of thousands), and
#' yearly average summer temperatures (June to August), based on simple 
#' unweighted average of three weather stations: Helsinki (Southern Finland), 
#' Jyväskylä (Central Finland), and Sodankylä (Northern Finland).
#'
#' @name drownings
#' @docType data
#' @format A time series object containing 51 observations.
#' @source Statistics Finland \url{https://pxnet2.stat.fi/PXWeb/pxweb/en/StatFin/}.
#' @keywords datasets
#' @examples
#' data("drownings")
#' model <- bsm_ng(drownings[, "deaths"], u = drownings[, "population"],
#'   xreg = drownings[, "summer_temp"], distribution = "poisson", 
#'   beta = normal(0, 0, 1),
#'   sd_level = gamma(0.1,2, 10), sd_slope = gamma(0, 2, 10))
#'   
#' fit <- run_mcmc(model, iter = 5000, 
#'   output_type = "summary", mcmc_type = "approx")
#' fit
#' ts.plot(model$y/model$u, exp(fit$alphahat[, 1]), col = 1:2)
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
#' @references James Durbin, Siem Jan Koopman (2012). "Time Series Analysis by State Space Methods". 
#' Oxford University Press.
#' @examples
#' data("exchange")
#' model <- svm(exchange, rho = uniform(0.97,-0.999,0.999),
#'  sd_ar = halfnormal(0.175, 2), mu = normal(-0.87, 0, 2))
#' 
#' out <- particle_smoother(model, particles = 500)
#' plot.ts(cbind(model$y, exp(out$alphahat))) 
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
#' # The data was generated as follows:
#' set.seed(321)
#' slope <- cumsum(c(0, rnorm(99, sd = 0.01)))
#' y <- rpois(100, exp(cumsum(slope + c(0, rnorm(99, sd = 0.1)))))
NULL