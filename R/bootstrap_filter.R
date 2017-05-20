#' Bootstrap Filtering
#'
#' Function \code{bootstrap_filter} performs a bootstrap filtering with stratification
#' resampling.
#'
#' @param object of class \code{bsm}, \code{ng_bsm} or \code{svm}.
#' @param nsim Number of samples.
#' @param seed Seed for RNG.
#' @param ... Ignored.
#' @return A list containing samples, weights from the last time point, and an
#' estimate of log-likelihood.
#' @export
#' @rdname bootstrap_filter
bootstrap_filter <- function(object, nsim, ...) {
  UseMethod("bootstrap_filter", object)
}
#' @method bootstrap_filter gssm
#' @export
bootstrap_filter.gssm <- function(object, nsim,
  seed = sample(.Machine$integer.max, size = 1), ...) {
  
  out <- bsf(object, nsim, seed, TRUE, 1L)
  colnames(out$at) <- colnames(out$att) <- colnames(out$Pt) <-
    colnames(out$Ptt) <- rownames(out$Pt) <- rownames(out$Ptt) <- names(object$a1)
  out$at <- ts(out$at, start = start(object$y), frequency = frequency(object$y))
  out$att <- ts(out$att, start = start(object$y), frequency = frequency(object$y))
  rownames(out$alpha) <- names(object$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}
#' @method bootstrap_filter bsm
#' @export
bootstrap_filter.bsm <- function(object, nsim,
  seed = sample(.Machine$integer.max, size = 1), ...) {
  
  out <- bsf(object, nsim, seed, TRUE, 2L)
  colnames(out$at) <- colnames(out$att) <- colnames(out$Pt) <-
    colnames(out$Ptt) <- rownames(out$Pt) <- rownames(out$Ptt) <- names(object$a1)
  out$at <- ts(out$at, start = start(object$y), frequency = frequency(object$y))
  out$att <- ts(out$att, start = start(object$y), frequency = frequency(object$y))
  rownames(out$alpha) <- names(object$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}

#' @method bootstrap_filter ngssm
#' @rdname bootstrap_filter
#' @export
bootstrap_filter.ngssm <- function(object, nsim,
  seed = sample(.Machine$integer.max, size = 1), ...) {
  
  object$distribution <- pmatch(object$distribution, c("poisson", "binomial", "negative binomial"))
  
  out <- bsf(object, nsim, seed, FALSE, 1L)
  colnames(out$at) <- colnames(out$att) <- colnames(out$Pt) <-
    colnames(out$Ptt) <- rownames(out$Pt) <- rownames(out$Ptt) <- names(object$a1)
  out$at <- ts(out$at, start = start(object$y), frequency = frequency(object$y))
  out$att <- ts(out$att, start = start(object$y), frequency = frequency(object$y))
  rownames(out$alpha) <- names(object$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}
#' @method bootstrap_filter ng_bsm
#' @export
bootstrap_filter.ng_bsm <- function(object, nsim,
  seed = sample(.Machine$integer.max, size = 1), ...) {
  
  object$distribution <- pmatch(object$distribution, c("poisson", "binomial", "negative binomial"))
  
  out <- bsf(object, nsim, seed, FALSE, 2L)
  colnames(out$at) <- colnames(out$att) <- colnames(out$Pt) <-
    colnames(out$Ptt) <- rownames(out$Pt) <- rownames(out$Ptt) <- names(object$a1)
  out$at <- ts(out$at, start = start(object$y), frequency = frequency(object$y))
  out$att <- ts(out$att, start = start(object$y), frequency = frequency(object$y))
  rownames(out$alpha) <- names(object$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}
#' @method bootstrap_filter svm
#' @rdname bootstrap_filter
#' @export
bootstrap_filter.svm <- function(object, nsim, 
  seed = sample(.Machine$integer.max, size = 1), ...) {
  
  out <- bsf(object, nsim, seed, FALSE, 3L)
  colnames(out$at) <- colnames(out$att) <- colnames(out$Pt) <-
    colnames(out$Ptt) <- rownames(out$Pt) <- rownames(out$Ptt) <- names(object$a1)
  out$at <- ts(out$at, start = start(object$y), frequency = frequency(object$y))
  out$att <- ts(out$att, start = start(object$y), frequency = frequency(object$y))
  rownames(out$alpha) <- names(object$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}

#' @method bootstrap_filter nlg_ssm
#' @rdname bootstrap_filter
#' @export
bootstrap_filter.nlg_ssm <- function(object, nsim,
  seed = sample(.Machine$integer.max, size = 1), ...) {
  
  out <- bsf_nlg(t(object$y), object$Z, object$H, object$T, 
    object$R, object$Z_gn, object$T_gn, object$a1, object$P1, 
    object$theta, object$log_prior_pdf, object$known_params, 
    object$known_tv_params, object$n_states, object$n_etas, 
    as.integer(object$time_varying), as.integer(object$state_varying), nsim, seed)
  colnames(out$at) <- colnames(out$att) <- colnames(out$Pt) <-
    colnames(out$Ptt) <- rownames(out$Pt) <- rownames(out$Ptt) <- 
    rownames(out$alpha) <- object$state_names
  out$at <- ts(out$at, start = start(object$y), frequency = frequency(object$y))
  out$att <- ts(out$att, start = start(object$y), frequency = frequency(object$y))
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}
