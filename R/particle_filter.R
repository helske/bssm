#' Particle Filtering
#'
#' Function \code{particle_filter} performs a bootstrap filtering with stratification
#' resampling. For non-Gaussian models, psi-auxiliary particle filter based on 
#' the Gaussian approximating model is also available.
#'
#' @param object of class \code{bsm}, \code{ng_bsm} or \code{svm}.
#' @param nsim Number of samples.
#' @param seed Seed for RNG.
#' @param filter_type Eiher \code{"bsf"} or \code{"psi"}.
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
particle_filter.ngssm <- function(object, nsim, filter_type = "bsf",
  seed = sample(.Machine$integer.max, size = 1), ...) {
  
  filter_type <- match.arg(filter_type, c("bsf", "psi"))
  
  object$distribution <- pmatch(object$distribution, c("poisson", "binomial", "negative binomial"))
  
  out <- ngssm_particle_filter(object, nsim, seed, filter_type == "bsf", object$initial_mode)
  
  rownames(out$alpha) <- names(object$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}
#' @method particle_filter ng_bsm
#' @export
particle_filter.ng_bsm <- function(object, nsim, filter_type = "bsf",
  seed = sample(.Machine$integer.max, size = 1), ...) {
  
  filter_type <- match.arg(filter_type, c("bsf", "psi"))
  object$distribution <- pmatch(object$distribution, c("poisson", "binomial", "negative binomial"))
  
  out <- ng_bsm_particle_filter(object, nsim, seed, filter_type == "bsf", object$initial_mode)
  
  rownames(out$alpha) <- names(object$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}
#' @method particle_filter svm
#' @rdname particle_filter
#' @export
particle_filter.svm <- function(object, nsim, filter_type = "bsf",
  seed = sample(.Machine$integer.max, size = 1), ...) {
  
  filter_type <- match.arg(filter_type, c("bsf", "psi"))
  out <- svm_particle_filter(object, nsim, seed, filter_type == "bsf", object$initial_mode)
  
  rownames(out$alpha) <- names(object$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}

#' @method particle_filter nlg_ssm
#' @rdname particle_filter
#' @export
particle_filter.nlg_ssm <- function(object, nsim, filter_type = "bsf",
  seed = sample(.Machine$integer.max, size = 1), ...) {
  
  filter_type <- match.arg(filter_type, c("bsf", "psi"))
  if(filter_type == "bsf") {
  out <- bootstrap_filter_nlg(t(object$y), object$Z, object$H, object$T, 
    object$R, object$Z_gn, object$T_gn, object$a1, object$P1, 
    object$theta, object$log_prior_pdf, object$known_params, 
    object$known_tv_params, object$n_states, object$n_etas, 
    as.integer(object$time_varying), nsim, seed)
  }
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}
