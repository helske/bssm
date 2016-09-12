## will add more choices later...
## add recycling of parameters later

#' Title
#'
#' @param init 
#' @param min 
#' @param max 
#'
#' @return
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

#' Title
#'
#' @param init 
#' @param sd 
#'
#' @return
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


#' Title
#'
#' @param init 
#' @param mean 
#' @param sd 
#'
#' @return
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
  list(prior_types = prior_types, params = params)
}
