#' Extended Kalman Particle Filtering
#'
#' Function \code{ekpf_filter} performs a extended Kalman particle filtering 
#' with stratification resampling, based on Van Der Merwe et al (2001).
#'
#' @param object Model of class \code{ssm_nlg}.
#' @param particles Number of particles as a positive integer.
#' @param seed Seed for RNG  (positive integer).
#' @param ... Ignored.
#' @return A list containing samples, filtered estimates and the 
#' corresponding covariances, weights, and an estimate of log-likelihood.
#' @references Van Der Merwe, R., Doucet, A., De Freitas, N., & Wan, E. A. 
#' (2001). The unscented particle filter. In Advances in neural 
#' information processing systems (pp. 584-590).
#' @export
#' @rdname ekpf_filter
ekpf_filter <- function(object, particles, ...) {
  UseMethod("ekpf_filter", object)
}
#' @method ekpf_filter ssm_nlg
#' @export
#' @rdname ekpf_filter
#' @examples
#' 
#' set.seed(1)
#' n <- 50
#' x <- y <- numeric(n)
#' y[1] <- rnorm(1, exp(x[1]), 0.1)
#' for(i in 1:(n-1)) {
#'  x[i+1] <- rnorm(1, sin(x[i]), 0.1)
#'  y[i+1] <- rnorm(1, exp(x[i+1]), 0.1)
#' }
#' 
#' pntrs <- cpp_example_model("nlg_sin_exp")
#' 
#' model_nlg <- ssm_nlg(y = y, a1 = pntrs$a1, P1 = pntrs$P1, 
#'   Z = pntrs$Z_fn, H = pntrs$H_fn, T = pntrs$T_fn, R = pntrs$R_fn, 
#'   Z_gn = pntrs$Z_gn, T_gn = pntrs$T_gn,
#'   theta = c(log_H = log(0.1), log_R = log(0.1)), 
#'   log_prior_pdf = pntrs$log_prior_pdf,
#'   n_states = 1, n_etas = 1, state_names = "state")
#'
#' out <- ekpf_filter(model_nlg, particles = 100)
#' ts.plot(cbind(x, out$at[1:n], out$att[1:n]), col = 1:3)
#'
ekpf_filter.ssm_nlg <- function(object, particles, 
  seed = sample(.Machine$integer.max, size = 1), ...) {
  
  if (missing(particles)) {
    nsim <- eval(match.call(expand.dots = TRUE)$nsim)
    if (!is.null(nsim)) {
      warning(paste("Argument `nsim` is deprecated. Use argument",
        "`particles` instead.", sep = " "))
      particles <- nsim
    }
  }
  particles <- check_integer(particles, "particles")
  
  nsamples <- ifelse(!is.null(nrow(object$y)), nrow(object$y), 
    length(object$y)) * object$n_states * particles
  if (particles > 100 & nsamples > 1e12) {
    warning(paste("Trying to sample ", nsamples, 
      "particles, you might run out of memory."))
  }
  
  seed <- check_integer(seed, "seed", FALSE, max = .Machine$integer.max)
  
  out <- ekpf(t(object$y), object$Z, object$H, object$T, 
    object$R, object$Z_gn, object$T_gn, object$a1, object$P1, 
    object$theta, object$log_prior_pdf, object$known_params, 
    object$known_tv_params, object$n_states, object$n_etas, 
    as.integer(object$time_varying), particles, 
    seed)
  colnames(out$at) <- colnames(out$att) <- colnames(out$Pt) <-
    colnames(out$Ptt) <- rownames(out$Pt) <- rownames(out$Ptt) <- 
    rownames(out$alpha) <- object$state_names
  out$at <- ts(out$at, start = start(object$y), 
    frequency = frequency(object$y))
  out$att <- ts(out$att, start = start(object$y), 
    frequency = frequency(object$y))
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}
