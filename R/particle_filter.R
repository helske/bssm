#' Particle Filtering
#'
#' Function \code{particle_filter} performs a bootstrap filtering with stratification
#' resampling. For non-Gaussian models, psi-auxiliary particle filter based on 
#' the Gaussian approximating model is also available.
#'
#' @param object of class \code{bsm}, \code{ng_bsm} or \code{svm}.
#' @param nsim Number of samples.
#' @param seed Seed for RNG.
#' @param filter_type Eiher \code{"bootstrap"} or \code{"psi"}.
#' @param ... Ignored.
#' @return A list containing samples, weights from the last time point, and an
#' estimate of log-likelihood.
#' @export
#' @rdname particle_filter
particle_filter <- function(object, nsim, ...) {
  UseMethod("particle_filter", object)
}
#' @method particle_filter gssm
#' @export
particle_filter.gssm <- function(object, nsim,
  seed = sample(.Machine$integer.max, size = 1), ...) {
  
  out <- gssm_particle_filter(object, nsim, seed)
  
  rownames(out$alpha) <- names(object$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}
#' @method particle_filter bsm
#' @export
particle_filter.bsm <- function(object, nsim,
  seed = sample(.Machine$integer.max, size = 1), ...) {
  
  out <- bsm_particle_filter(object, nsim, seed)
  
  rownames(out$alpha) <- names(object$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}

#' @method particle_filter ngssm
#' @rdname particle_filter
#' @export
particle_filter.ngssm <- function(object, nsim, filter_type = "bootstrap",
  seed = sample(.Machine$integer.max, size = 1), ...) {
  
  filter_type <- match.arg(filter_type, c("bootstrap", "psi"))
  
  object$distribution <- pmatch(object$distribution, c("poisson", "binomial", "negative binomial"))
  
  out <- ngssm_particle_filter(object, nsim, seed, filter_type == "bootstrap", object$initial_mode)
  
  rownames(out$alpha) <- names(object$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}
#' @method particle_filter ng_bsm
#' @export
particle_filter.ng_bsm <- function(object, nsim, filter_type = "bootstrap",
  seed = sample(.Machine$integer.max, size = 1), ...) {
  
  filter_type <- match.arg(filter_type, c("bootstrap", "psi"))
  object$distribution <- pmatch(object$distribution, c("poisson", "binomial", "negative binomial"))
  
  out <- ng_bsm_particle_filter(object, nsim, seed, filter_type == "bootstrap", object$initial_mode)
  
  rownames(out$alpha) <- names(object$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}
#' @method particle_filter svm
#' @rdname particle_filter
#' @export
particle_filter.svm <- function(object, nsim, filter_type = "bootstrap",
  seed = sample(.Machine$integer.max, size = 1), ...) {
  
  filter_type <- match.arg(filter_type, c("bootstrap", "psi"))
  out <- svm_particle_filter(object, nsim, seed, filter_type == "bootstrap", object$initial_mode)
  
  rownames(out$alpha) <- names(object$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}
