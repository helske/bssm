
#' Auxiliary Particle Filtering
#'
#' Function \code{auxiliary_filter} performs a auxiliary particle filtering with stratification
#' resampling.
#'
#' @param object of class \code{gssm}, \code{bsm}, or \code{nlg_ssm}.
#' @param nsim Number of samples.
#' @param optimal For Gaussian models, use optimal proposals? Default is \code{TRUE}.
#' @param seed Seed for RNG.
#' @param use_ekf For non-linear Gaussian models, use extended Kalman filter for proposals? 
#' Default is \code{TRUE}.
#' @param ... Ignored.
#' @return A list containing samples, filtered estimates and the corresponding covariances,
#' weights from the last time point, and an estimate of log-likelihood.
#' @export
#' @rdname auxiliary_filter
auxiliary_filter <- function(object, nsim, ...) {
  UseMethod("auxiliary_filter", object)
}
#' @method auxiliary_filter nlg_ssm
#' @export
auxiliary_filter.nlg_ssm <- function(object, nsim,
  seed = sample(.Machine$integer.max, size = 1), use_ekf = TRUE, ...) {
  
  out <- aux_nlg(t(object$y), object$Z, object$H, object$T, 
    object$R, object$Z_gn, object$T_gn, object$a1, object$P1, 
    object$theta, object$log_prior_pdf, object$known_params, 
    object$known_tv_params, object$n_states, object$n_etas, 
    as.integer(object$time_varying), as.integer(object$state_varying), nsim, 
    seed, use_ekf)
  colnames(out$at) <- colnames(out$att) <- colnames(out$Pt) <-
    colnames(out$Ptt) <- rownames(out$Pt) <- rownames(out$Ptt) <- 
    rownames(out$alpha) <- object$state_names
  out$at <- ts(out$at, start = start(object$y), frequency = frequency(object$y))
  out$att <- ts(out$att, start = start(object$y), frequency = frequency(object$y))
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}
#' @method auxiliary_filter gssm
#' @export
auxiliary_filter.gssm <- function(object, nsim,
  seed = sample(.Machine$integer.max, size = 1), optimal = TRUE, ...) {
  
  out <- aux(object, nsim, seed, TRUE, 1L, optimal)
  colnames(out$at) <- colnames(out$att) <- colnames(out$Pt) <-
    colnames(out$Ptt) <- rownames(out$Pt) <- rownames(out$Ptt) <- names(object$a1)
  out$at <- ts(out$at, start = start(object$y), frequency = frequency(object$y))
  out$att <- ts(out$att, start = start(object$y), frequency = frequency(object$y))
  rownames(out$alpha) <- names(object$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}

#' @method auxiliary_filter bsm
#' @export
auxiliary_filter.bsm <- function(object, nsim,
  seed = sample(.Machine$integer.max, size = 1), optimal = TRUE, ...) {
  
  out <- aux(object, nsim, seed, TRUE, 2L, optimal)
  colnames(out$at) <- colnames(out$att) <- colnames(out$Pt) <-
    colnames(out$Ptt) <- rownames(out$Pt) <- rownames(out$Ptt) <- names(object$a1)
  out$at <- ts(out$at, start = start(object$y), frequency = frequency(object$y))
  out$att <- ts(out$att, start = start(object$y), frequency = frequency(object$y))
  rownames(out$alpha) <- names(object$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}
