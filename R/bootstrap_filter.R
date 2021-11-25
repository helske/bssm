#' Bootstrap Filtering
#'
#' Function \code{bootstrap_filter} performs a bootstrap filtering with 
#' stratification resampling.
#' 
#' @param model A model object of class \code{bssm_model}.
#' @param particles Number of particles as a positive integer. Suitable values 
#' depend on the model and the data, and while larger values provide more 
#' accurate estimates, the run time also increases with respect to the 
#' number of particles, so it is generally a good idea to test the filter first 
#' with a small number of particles, e.g., less than 100.
#' @param seed Seed for the C++ RNG (positive integer).
#' @param ... Ignored.
#' @return List with samples (\code{alpha}) from the filtering distribution and 
#' corresponding weights (\code{weights}), as well as filtered and predicted 
#' states and corresponding covariances (\code{at}, \code{att}, \code{Pt}, 
#' \code{Ptt}), and estimated log-likelihood (\code{logLik}).
#' @export
#' @references 
#' Gordon, NJ, Salmond, DJ, Smith, AFM (1993) Novel approach to 
#' nonlinear/non-Gaussian Bayesian state estimation. IEE Proceedings F, 
#' 140(2), p. 107-113.
#' @rdname bootstrap_filter
bootstrap_filter <- function(model, particles, ...) {
  UseMethod("bootstrap_filter", model)
}
#' @method bootstrap_filter lineargaussian
#' @rdname bootstrap_filter
#' @export
#' @examples 
#' set.seed(1)
#' x <- cumsum(rnorm(50))
#' y <- rnorm(50, x, 0.5) 
#' model <- bsm_lg(y, sd_y = 0.5, sd_level = 1, P1 = 1)
#'   
#' out <- bootstrap_filter(model, particles = 1000)
#' ts.plot(cbind(y, x, out$att), col = 1:3)
#' ts.plot(cbind(kfilter(model)$att, out$att), col = 1:3)
#' 
bootstrap_filter.lineargaussian <- function(model, particles,
  seed = sample(.Machine$integer.max, size = 1), ...) {
  
  check_missingness(model)
  
  if (missing(particles)) {
    nsim <- eval(match.call(expand.dots = TRUE)$nsim)
    if (!is.null(nsim)) {
      warning(paste0("Argument `nsim` is deprecated. Use argument `particles`",
        "instead.", sep = " "))
      particles <- nsim
    }
  }
  particles <- check_intmax(particles, "particles")
  seed <- check_intmax(seed, "seed", FALSE, max = .Machine$integer.max)
  
  nsamples <- ifelse(!is.null(nrow(model$y)), nrow(model$y), length(model$y)) * 
    length(model$a1) * particles
  if (particles > 100 & nsamples > 1e10) {
    warning(paste("Trying to sample ", nsamples, 
      "particles, you might run out of memory."))
  }
  out <- bsf(model, particles, seed, TRUE, model_type(model))
  colnames(out$at) <- colnames(out$att) <- colnames(out$Pt) <-
    colnames(out$Ptt) <- rownames(out$Pt) <- rownames(out$Ptt) <- 
    names(model$a1)
  out$at <- ts(out$at, start = start(model$y), frequency = frequency(model$y))
  out$att <- ts(out$att, start = start(model$y), 
    frequency = frequency(model$y))
  rownames(out$alpha) <- names(model$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}

#' @method bootstrap_filter nongaussian
#' @rdname bootstrap_filter
#' @export
#' @examples 
#' data("poisson_series")
#' model <- bsm_ng(poisson_series, sd_level = 0.1, sd_slope = 0.01, 
#'   P1 = diag(1, 2), distribution = "poisson")
#'   
#' out <- bootstrap_filter(model, particles = 100)
#' ts.plot(cbind(poisson_series, exp(out$att[, 1])), col = 1:2)
#' 
bootstrap_filter.nongaussian <- function(model, particles,
  seed = sample(.Machine$integer.max, size = 1), ...) {

  check_missingness(model)
  
  if (missing(particles)) {
    nsim <- eval(match.call(expand.dots = TRUE)$nsim)
    if (!is.null(nsim)) {
      warning(paste0("Argument `nsim` is deprecated. Use argument `particles`",
      "instead.", sep = " "))
      particles <- nsim
    }
  }
  particles <- check_intmax(particles, "particles")
  seed <- check_intmax(seed, "seed", FALSE, max = .Machine$integer.max)
  nsamples <- ifelse(!is.null(nrow(model$y)), nrow(model$y), length(model$y)) * 
    length(model$a1) * particles
  if (particles > 100 & nsamples > 1e10) {
    warning(paste("Trying to sample ", nsamples, 
      "particles, you might run out of memory."))
  }
  model$distribution <- 
    pmatch(model$distribution, 
      c("svm", "poisson", "binomial", "negative binomial", "gamma", "gaussian"),
      duplicates.ok = TRUE) - 1
  
  out <- bsf(model, particles, seed, FALSE, model_type(model))
  colnames(out$at) <- colnames(out$att) <- colnames(out$Pt) <-
    colnames(out$Ptt) <- rownames(out$Pt) <- rownames(out$Ptt) <- 
    names(model$a1)
  out$at <- ts(out$at, start = start(model$y), frequency = frequency(model$y))
  out$att <- ts(out$att, start = start(model$y), frequency = frequency(model$y))
  rownames(out$alpha) <- names(model$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}
#' @method bootstrap_filter ssm_nlg
#' @rdname bootstrap_filter
#' @export
bootstrap_filter.ssm_nlg <- function(model, particles,
  seed = sample(.Machine$integer.max, size = 1), ...) {
  
  check_missingness(model)
  
  if (missing(particles)) {
    nsim <- eval(match.call(expand.dots = TRUE)$nsim)
    if (!is.null(nsim)) {
      warning(paste("Argument `nsim` is deprecated. Use argument `particles`",
      "instead.", sep = " "))
      particles <- nsim
    }
  }
  particles <- check_intmax(particles, "particles")
  
  nsamples <- ifelse(!is.null(nrow(model$y)), nrow(model$y), length(model$y)) * 
    model$n_states * particles
  if (particles > 100 & nsamples > 1e10) {
    warning(paste("Trying to sample ", nsamples, 
      "particles, you might run out of memory."))
  }
  seed <- check_intmax(seed, "seed", FALSE, max = .Machine$integer.max)
  
  out <- bsf_nlg(t(model$y), model$Z, model$H, model$T,
    model$R, model$Z_gn, model$T_gn, model$a1, model$P1,
    model$theta, model$log_prior_pdf, model$known_params,
    model$known_tv_params, model$n_states, model$n_etas,
    as.integer(model$time_varying), particles, seed)
  colnames(out$at) <- colnames(out$att) <- colnames(out$Pt) <-
    colnames(out$Ptt) <- rownames(out$Pt) <- rownames(out$Ptt) <-
    rownames(out$alpha) <- model$state_names
  out$at <- ts(out$at, start = start(model$y), frequency = frequency(model$y))
  out$att <- ts(out$att, start = start(model$y), frequency = frequency(model$y))
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}

#' @method bootstrap_filter ssm_sde
#' @rdname bootstrap_filter
#' @param L Positive integer defining the discretization level for SDE models.
#' @export
bootstrap_filter.ssm_sde <- function(model, particles, L,
  seed = sample(.Machine$integer.max, size = 1), ...) {
  
  check_missingness(model)
  
  if (!test_count(L, positive=TRUE)) 
    stop("Discretization level L must be a positive integer.")
  
  if (missing(particles)) {
    nsim <- eval(match.call(expand.dots = TRUE)$nsim)
    if (!is.null(nsim)) {
      warning(paste("Argument `nsim` is deprecated. Use argument `particles`",
      "instead.", sep = " "))
      particles <- nsim
    }
  }
  
  particles <- check_intmax(particles, "particles")
  
  nsamples <- length(model$y) * particles
  if (particles > 100 & nsamples > 1e10) {
    warning(paste("Trying to sample ", nsamples, 
      "particles, you might run out of memory."))
  }
  seed <- check_intmax(seed, "seed", FALSE, max = .Machine$integer.max)
  
  out <- bsf_sde(model$y, model$x0, model$positive,
    model$drift, model$diffusion, model$ddiffusion,
    model$prior_pdf, model$obs_pdf, model$theta,
    particles, round(L), seed)
  colnames(out$at) <- colnames(out$att) <- colnames(out$Pt) <-
    colnames(out$Ptt) <- rownames(out$Pt) <- rownames(out$Ptt) <-
    rownames(out$alpha) <- model$state_names
  out$at <- ts(out$at, start = start(model$y), frequency = frequency(model$y))
  out$att <- ts(out$att, start = start(model$y), frequency = frequency(model$y))
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}
