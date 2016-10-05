#' Backwards sampling
#'
#' Function \code{particle_simulate} performs backwards sampling from
#' the conditional distribution of the states $p(alpha|y, theta)$.
#'
#' @param object Model.
#' @param nsim Number of samples used in particle filtering.
#' @param nsim_store Number of samples to store.
#' @param seed Seed for RNG.
#' @param ... Ignored.
#' @rdname particle_simulate
#' @export
particle_simulate <- function(object, nsim, nsim_store, seed, ...) {
  UseMethod("particle_simulate", object)
}
#' @method particle_simulate gssm
#' @rdname particle_simulate
#' @export
particle_simulate.gssm <- function(object, nsim, nsim_store = 1,
  seed = sample(.Machine$integer.max, size = 1), ...) {

  out <- gssm_backward_simulate(object, nsim, seed, nsim_store)

  rownames(out$alpha) <- names(object$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}

#' @method particle_simulate bsm
#' @rdname particle_simulate
#' @export
particle_simulate.bsm <- function(object, nsim, nsim_store = 1,
  seed = sample(.Machine$integer.max, size = 1), ...) {

  out <- bsm_backward_simulate(object, nsim, seed, nsim_store)

  rownames(out$alpha) <- names(object$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}

#' @method particle_simulate ngssm
#' @rdname particle_simulate
#' @export
particle_simulate.ngssm <- function(object, nsim, nsim_store = 1,
  seed = sample(.Machine$integer.max, size = 1), ...) {
  
  object$distribution <- pmatch(object$distribution,
    c("poisson", "binomial", "negative binomial"))
  out <- ngssm_backward_simulate(object, nsim, seed, nsim_store)
  
  rownames(out$alpha) <- names(object$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}
#' @method particle_simulate ng_bsm
#' @rdname particle_simulate
#' @export
particle_simulate.ng_bsm <- function(object, nsim, nsim_store = 1,
  seed = sample(.Machine$integer.max, size = 1), ...) {
  
  object$distribution <- pmatch(object$distribution,
    c("poisson", "binomial", "negative binomial"))
  out <- ng_bsm_backward_simulate(object, nsim, seed, nsim_store)
  
  rownames(out$alpha) <- names(object$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}

#' @method particle_simulate svm
#' @rdname particle_simulate
#' @export
particle_simulate.svm <- function(object, nsim, nsim_store = 1,
  seed = sample(.Machine$integer.max, size = 1), ...) {

  object$distribution <- 0
  object$phi <- object$sigma
  object$u <- 1
  out <- svm_backward_simulate(object, nsim, seed, nsim_store)

  rownames(out$alpha) <- names(object$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}
