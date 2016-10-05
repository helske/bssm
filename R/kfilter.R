#' Kalman Filtering
#'
#' Function \code{kfilter} runs the Kalman filter for the given model, 
#' and returns the filtered estimates and one-step-ahead predictions of the 
#' states \eqn{\alpha_t} given the data up to time \eqn{t}.
#'
#' For non-Gaussian models, the Kalman filtering is based on the approximate Gaussian model.
#'
#' @param object Model object
#' @param ... Ignored.
#' @return List containing the log-likelihood (approximate in non-Gaussian case),
#' one-step-ahead predictions \code{at} and filtered
#' estimates \code{att} of states, and the corresponding variances \code{Pt} and
#'  \code{Ptt}.
#' @seealso \code{\link{particle_filter}}
#' @export
#' @rdname kfilter
kfilter <- function(object, ...) {
  UseMethod("kfilter", object)
}

#' @method kfilter gssm
#' @export
kfilter.gssm <- function(object, ...) {
  
  out <- gssm_filter(object)
  colnames(out$at) <- colnames(out$att) <- colnames(out$Pt) <-
    colnames(out$Ptt) <- rownames(out$Pt) <- rownames(out$Ptt) <- names(object$a1)
  out$at <- ts(out$at, start = start(object$y), frequency = frequency(object$y))
  out$att <- ts(out$att, start = start(object$y), frequency = frequency(object$y))
  out
}
#' @method kfilter bsm
#' @export
kfilter.bsm <- function(object, ...) {
  
  out <- bsm_filter(object)
  colnames(out$at) <- colnames(out$att) <- colnames(out$Pt) <-
    colnames(out$Ptt) <- rownames(out$Pt) <- rownames(out$Ptt) <- names(object$a1)
  out$at <- ts(out$at, start = start(object$y), frequency = frequency(object$y))
  out$att <- ts(out$att, start = start(object$y), frequency = frequency(object$y))
  out
}

#' @method kfilter ngssm
#' @export
kfilter.ngssm <- function(object, ...) {
  
  object$distribution <- pmatch(object$distribution,
    c("poisson", "binomial", "negative binomial"))
  
  out <- ngssm_filter(object, object$init_signal)
  colnames(out$at) <- colnames(out$att) <- colnames(out$Pt) <-
    colnames(out$Ptt) <- rownames(out$Pt) <- rownames(out$Ptt) <- names(object$a1)
  out$at <- ts(out$at, start = start(object$y), frequency = frequency(object$y))
  out$att <- ts(out$att, start = start(object$y), frequency = frequency(object$y))
  out
}

#' @method kfilter ng_bsm
#' @export
kfilter.ng_bsm <- function(object, ...) {
  
  object$distribution <- pmatch(object$distribution, c("poisson", "binomial", "negative binomial"))
  
  out <- ng_bsm_filter(object, object$init_signal)
  colnames(out$at) <- colnames(out$att) <- colnames(out$Pt) <-
    colnames(out$Ptt) <- rownames(out$Pt) <- rownames(out$Ptt) <- names(object$a1)
  out$at <- ts(out$at, start = start(object$y), frequency = frequency(object$y))
  out$att <- ts(out$att, start = start(object$y), frequency = frequency(object$y))
  out
}

#' @method kfilter svm
#' @export
kfilter.svm <- function(object, ...) {
  
  object$distribution <- 0L
  object$phi <- object$sigma
  object$u <- 1
  out <- ngssm_filter(object, object$init_signal)
  
  colnames(out$at) <- colnames(out$att) <- colnames(out$Pt) <-
    colnames(out$Ptt) <- rownames(out$Pt) <- rownames(out$Ptt) <- names(object$a1)
  out$at <- ts(out$at, start = start(object$y), frequency = frequency(object$y))
  out$att <- ts(out$att, start = start(object$y), frequency = frequency(object$y))
  out
}