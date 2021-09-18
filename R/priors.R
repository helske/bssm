#' Prior objects for bssm models
#'
#' These simple objects of class \code{bssm_prior} are used to construct a 
#' prior distributions for the some of the model objects of \code{bssm} 
#' package. Currently supported priors are uniform 
#' (\code{uniform()}), half-normal (\code{halfnormal()}), 
#' normal (\code{normal()}), gamma (\code{gamma}), and 
#' truncated normal distribution  (\code{tnormal()}). All parameters are 
#' vectorized so for regression coefficient vector beta you can define prior 
#' for example as \code{normal(0, 0, c(10, 20))}.
#' 
#' The longer name versions of the prior functions with \code{_prior} ending 
#' are identical with shorter versions and they are available only to 
#' avoid clash with R's primitive function \code{gamma} (other long prior names 
#' are just for consistent naming).
#' 
#' @rdname bssm_prior
#' @aliases bssm_prior_list
#' @param init Initial value for the parameter, used in initializing the model 
#' components and as a starting values in MCMC. 
#' @param min Lower bound of the uniform and truncated normal prior.
#' @param max Upper bound of the uniform and truncated normal prior.
#' @param sd Positive value defining the standard deviation of the 
#' (underlying i.e. non-truncated) Normal distribution.
#' @param mean Mean of the Normal prior.
#' @param shape Positive shape parameter of the Gamma prior.
#' @param rate Positive rate parameter of the Gamma prior.
#' @return object of class \code{bssm_prior} or \code{bssm_prior_list} in case 
#' of multiple priors (i.e. multiple regression coefficients).
#' @export
#' @examples
#' # create uniform prior on [-1, 1] for one parameter with initial value 0.2:
#' uniform(init = 0.2, min = -1.0, max = 1.0)
#' # two normal priors at once i.e. for coefficients beta:
#' normal(init = c(0.1, 2.5), mean = 0.1, sd = c(1.5, 2.8))
#' # Gamma prior
#' gamma(init = 0.1, shape = 2.5, rate = 1.1)
#' # Same as
#' gamma_prior(init = 0.1, shape = 2.5, rate = 1.1)
#' # Half-normal
#' halfnormal(init = 0.01, sd = 0.1)
#' # Truncated normal
#' tnormal(init = 5.2, mean = 5.0, sd = 3.0, min = 0.5, max = 9.5)
#' 
#' # Further examples for diagnostic purposes:
#' uniform(c(0, 0.2), c(-1.0, 0.001), c(1.0, 1.2))
#' normal(c(0, 0.2), c(-1.0, 0.001), c(1.0, 1.2))
#' tnormal(c(0, 0.2), c(-1.0, 0.001), c(1.0, 1.2), c(1.2, 2), c(3.3, 3.3))
#' halfnormal(c(0, 0.2), c(1.0, 1.2))
#' gamma(c(0.1, 0.2), c(1.2, 2), c(3.3, 3.3))
#' 
#' # longer versions:
#' uniform_prior(c(0, 0.2), c(-1.0, 0.001), c(1.0, 1.2))
#' normal_prior(c(0, 0.2), c(-1.0, 0.001), c(1.0, 1.2))
#' tnormal_prior(c(0, 0.2), c(-1.0, 0.001), c(1.0, 1.2), c(1.2, 2), c(3.3, 3.3))
#' halfnormal_prior(c(0, 0.2), c(1.0, 1.2))
#' gamma_prior(c(0.1, 0.2), c(1.2, 2), c(3.3, 3.3))
#' 
uniform_prior <- function(init, min, max) {
  if (any(!is.numeric(init), !is.numeric(min), !is.numeric(max))) {
    stop("Parameters for priors must be numeric.")
  }
  if (any(min > max)) {
    stop(paste("Lower bound of uniform distribution must be smaller than",
     "upper bound.", sep = " "))
  }
  if (any(init < min) || any(init > max)) {
    stop(paste("Initial value for parameter with uniform prior is not",
    "in the support of the prior.", sep = " "))
  }
  n <- max(length(init), length(min), length(max))
  
  if (n > 1) {
    structure(lapply(1:n, function(i) 
      structure(list(prior_distribution = "uniform", init = safe_pick(init, i),
      min = safe_pick(min, i), max = safe_pick(max, i)), 
        class = "bssm_prior_list")), 
      class = "bssm_prior_list")
  } else {
    structure(list(prior_distribution = "uniform", init = init, 
      min = min, max = max), class = "bssm_prior")
  }
}
#' @rdname bssm_prior
#' @export
uniform <- uniform_prior

#' @rdname bssm_prior
#' @export
halfnormal_prior <- function(init, sd) {
  
  if (any(!is.numeric(init), !is.numeric(sd))) {
    stop("Parameters for priors must be numeric.")
  }
  if (any(sd < 0)) {
    stop(paste("Standard deviation parameter for half-Normal distribution must",
    "be positive.", sep = " "))
  }
  if (any(init < 0)) {
    stop(paste("Initial value for parameter with half-Normal prior must be",
    "non-negative.", sep = " "))
  }
  n <- max(length(init), length(sd))
  
  if (n > 1) {
    structure(lapply(1:n, function(i) 
      structure(list(prior_distribution = "halfnormal", 
        init = safe_pick(init, i),
      sd = safe_pick(sd, i)), class = "bssm_prior")), class = "bssm_prior_list")
  } else {
    structure(list(prior_distribution = "halfnormal", init = init, sd = sd), 
      class = "bssm_prior")
  }
}
#' @rdname bssm_prior
#' @export
halfnormal <- halfnormal_prior

#' @rdname bssm_prior
#' @export
normal_prior <- function(init, mean, sd) {
  
  if (any(!is.numeric(init), !is.numeric(mean), !is.numeric(sd))) {
    stop("Parameters for priors must be numeric.")
  }
  if (any(sd < 0)) {
    stop(paste("Standard deviation parameter for Normal distribution must",
     "be positive.", sep = " "))
  }
  
  n <- max(length(init), length(mean), length(sd))
  if (n > 1) {
    structure(lapply(1:n, function(i) 
      structure(list(prior_distribution = "normal", 
      init = safe_pick(init, i), mean = safe_pick(mean, i), 
        sd = safe_pick(sd, i)), 
      class = "bssm_prior")), class = "bssm_prior_list")

  } else {
    structure(list(prior_distribution = "normal", init = init, mean = mean, 
      sd = sd), class = "bssm_prior")
  }
}

#' @rdname bssm_prior
#' @export
normal <- normal_prior

#' @rdname bssm_prior
#' @export
tnormal_prior <- function(init, mean, sd, min = -Inf, max = Inf) {
  
  if (any(!is.numeric(init), !is.numeric(mean), !is.numeric(sd))) {
    stop("Parameters for priors must be numeric.")
  }
  if (any(sd < 0)) {
    stop(paste("Standard deviation parameter for Normal distribution must be",
    "positive.", sep = " "))
  }
  
  n <- max(length(init), length(mean), length(sd))
  if (n > 1) {
    structure(lapply(1:n, function(i) 
      structure(list(prior_distribution = "tnormal", 
      init = safe_pick(init, i), mean = safe_pick(mean, i), 
        sd = safe_pick(sd, i),
      min = safe_pick(min, i), max = safe_pick(max, i)), 
      class = "bssm_prior")), class = "bssm_prior_list")
  } else {
    structure(list(prior_distribution = "tnormal", init = init, mean = mean,
      sd = sd, min = min, max = max), class = "bssm_prior")
  }
}

#' @rdname bssm_prior
#' @export
tnormal <- tnormal_prior

#' @rdname bssm_prior
#' @export
gamma_prior <- function(init, shape, rate) {
  
  if (any(!is.numeric(init), !is.numeric(shape), !is.numeric(rate))) {
    stop("Parameters for priors must be numeric.")
  }
  if (any(shape < 0)) {
    stop("Shape parameter for Gamma distribution must be positive.")
  }
  if (any(rate < 0)) {
    stop("Rate parameter for Gamma distribution must be positive.")
  }
  n <- max(length(init), length(shape), length(rate))
  if (n > 1) {
    structure(lapply(1:n, function(i) 
      structure(list(prior_distribution = "gamma", 
        init = safe_pick(init, i), shape = safe_pick(shape, i), 
        rate = safe_pick(rate, i)), 
        class = "bssm_prior")), class = "bssm_prior_list")
    
  } else {
    structure(list(prior_distribution = "gamma", init = init, 
      shape = shape, rate = rate), 
      class = "bssm_prior")
  }
}
#' @rdname bssm_prior
#' @export
gamma <- gamma_prior

combine_priors <- function(x) {
  
  if (length(x) == 0) 
    return(list(prior_distributions = 0, parameters = matrix(0, 0, 0)))
  
  prior_distributions <- vapply(x, "[[", "prior_distribution", 
    FUN.VALUE = character(1))
  parameters <- matrix(NA, 4, length(prior_distributions))
  for (i in seq_along(prior_distributions)) {
    parameters[1:(length(x[[i]]) - 2), i] <- as.numeric(x[[i]][-(1:2)])
  }
  list(prior_distributions = 
      pmatch(prior_distributions, c("uniform", "halfnormal", "normal", 
        "tnormal", "gamma"), duplicates.ok = TRUE) - 1, 
    parameters = parameters)
}




is_prior <- function(x) {
  inherits(x, "bssm_prior")
}
is_prior_list <- function(x) {
  inherits(x, "bssm_prior_list")
}
safe_pick <- function(x, i) {
  x[min(length(x), i)]
}
