## will add more choices later...
## add recycling of parameters later

#' Prior objects for bssm models
#'
#' These simple objects of class \code{bssm_prior} are used to construct a prior distributions for the 
#' MCMC runs of \code{bssm} package. Currently supported priors are uniform, half-Normal and Normal distribution.
#' 
#' @note Use Normal prior with care: currently there is no checks of non-negativity of standard deviation or stationarity of 
#' autoregressive coefficient in case Normal prior is used.
#' 
#' @rdname priors
#' @param init Initial value for the parameter, used in initializing the model components and as a starting value
#' in MCMC.
#' @param min Lower bound of the uniform prior .
#' @param max Upper bound of the uniform prior.
#' @param sd Standard deviation of the half-Normal and Normal priors.
#' @param mean Mean of the Normal prior.
#' @return object of class \code{bssm_prior}.
#' @export
uniform <- function(init, min, max){
  if(any(!is.numeric(init), !is.numeric(min), !is.numeric(max))) {
    stop("Parameters for priors must be numeric.")
  }
  if (any(min > max)){
    stop("Lower bound of uniform distribution must be smaller than upper bound.")
  }
  if(any(init < min) || any(init > max)) {
    stop("Initial value for parameter with uniform prior is not in the support of the prior.")
  }
  structure(list(prior_type = "uniform", init = init, min = min, max = max), class = "bssm_prior")
}

#' @rdname priors
#' @export
halfnormal <- function(init, sd){
  
  if(any(!is.numeric(init), !is.numeric(sd))) {
    stop("Parameters for priors must be numeric.")
  }
  if (any(sd < 0)) {
    stop("Standard deviation parameter for half-Normal distribution must be positive.")
  }
  if (any(init < 0)) {
    stop("Initial value for parameter with half-Normal prior must be non-negative.")
  }
  structure(list(prior_type = "halfnormal", init = init, sd = sd), class = "bssm_prior")
}


#' @rdname priors
#' @export
normal <- function(init, mean, sd){
  
  if(any(!is.numeric(init), !is.numeric(mean), !is.numeric(sd))) {
    stop("Parameters for priors must be numeric.")
  }
  if (any(sd < 0)) {
    stop("Standard deviation parameter for Normal distribution must be positive.")
  }
  structure(list(prior_type = "normal", init = init, mean = mean, sd = sd), class = "bssm_prior")
}

combine_priors <- function(x) {
  
  prior_types <- rep(sapply(x, "[[", "prior_type"), times = sapply(x, function(z) length(z$init)))
  params <- matrix(NA, 2, length(prior_types))
  for(i in 1:length(prior_types)) {
    params[1:(length(x[[i]])-2), i] <- as.numeric(x[[i]][-(1:2)])
  }
  list(prior_types = pmatch(prior_types, c("uniform", "halfnormal", "normal"), duplicates.ok = TRUE)-1, 
    params = params)
}

is_prior <- function(x){
  inherits(x, "bssm_prior")
}
