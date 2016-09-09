## will add more choices later...
## add recycling of parameters later

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
  structure(list(prior_type = "uniform", min = min, max = max), class = "bssm_prior")
}

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
  structure(list(prior_type = "halfnormal", sd = sd), class = "bssm_prior")
}


normal <- function(init, mean, sd){
  
  if(any(!is.numeric(init), !is.numeric(mean), !is.numeric(sd))) {
    stop("Parameters for priors must be numeric.")
  }
  if (any(sd < 0)) {
    stop("Standard deviation parameter for Normal distribution must be positive.")
  }
  structure(list(prior_type = "normal", mean = mean, sd = sd), class = "bssm_prior")
}

