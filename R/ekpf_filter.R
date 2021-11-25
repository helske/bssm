#' Extended Kalman Particle Filtering
#'
#' Function \code{ekpf_filter} performs a extended Kalman particle filtering 
#' with stratification resampling, based on Van Der Merwe et al (2001).
#'
#' @inheritParams bootstrap_filter
#' @param model Model of class \code{ssm_nlg}.
#' @param ... Ignored.
#' @return A list containing samples, filtered estimates and the 
#' corresponding covariances, weights, and an estimate of log-likelihood.
#' @references Van Der Merwe, R., Doucet, A., De Freitas, N., & Wan, E. A. 
#' (2001). The unscented particle filter. In Advances in neural 
#' information processing systems (pp. 584-590).
#' @export
#' @rdname ekpf_filter
ekpf_filter <- function(model, particles, ...) {
  UseMethod("ekpf_filter", model)
}
#' @method ekpf_filter ssm_nlg
#' @export
#' @rdname ekpf_filter
#' @examples
#' \donttest{ # Takes a while
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
#'}
ekpf_filter.ssm_nlg <- function(model, particles, 
  seed = sample(.Machine$integer.max, size = 1), ...) {
  
  check_missingness(model)
  
  if (missing(particles)) {
    nsim <- eval(match.call(expand.dots = TRUE)$nsim)
    if (!is.null(nsim)) {
      warning(paste("Argument `nsim` is deprecated. Use argument",
        "`particles` instead.", sep = " "))
      particles <- nsim
    }
  }
  particles <- check_intmax(particles, "particles")
  
  nsamples <- ifelse(!is.null(nrow(model$y)), nrow(model$y), 
    length(model$y)) * model$n_states * particles
  if (particles > 100 & nsamples > 1e10) {
    warning(paste("Trying to sample ", nsamples, 
      "particles, you might run out of memory."))
  }
  
  seed <- check_intmax(seed, "seed", FALSE, max = .Machine$integer.max)
  
  out <- ekpf(t(model$y), model$Z, model$H, model$T, 
    model$R, model$Z_gn, model$T_gn, model$a1, model$P1, 
    model$theta, model$log_prior_pdf, model$known_params, 
    model$known_tv_params, model$n_states, model$n_etas, 
    as.integer(model$time_varying), particles, 
    seed)
  colnames(out$at) <- colnames(out$att) <- colnames(out$Pt) <-
    colnames(out$Ptt) <- rownames(out$Pt) <- rownames(out$Ptt) <- 
    rownames(out$alpha) <- model$state_names
  out$at <- ts(out$at, start = start(model$y), 
    frequency = frequency(model$y))
  out$att <- ts(out$att, start = start(model$y), 
    frequency = frequency(model$y))
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}
