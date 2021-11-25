#' Particle Smoothing
#'
#' Function \code{particle_smoother} performs particle smoothing 
#' based on either bootstrap particle filter (Gordon et al. 1993), 
#' \eqn{\psi}-auxiliary particle filter (\eqn{\psi}-APF) (Vihola et al. 2020), 
#' extended Kalman particle filter (Van Der Merwe et al. 2001), 
#' or its version based on iterated EKF (Jazwinski, 1970). 
#' The smoothing phase is based on the filter-smoother algorithm by 
#' Kitagawa (1996).
#' 
#' See one of the vignettes for \eqn{\psi}-APF in case of nonlinear models.
#'
#' @importFrom stats cov
#' @inheritParams bootstrap_filter
#' @param method Choice of particle filter algorithm. 
#' For Gaussian and non-Gaussian models with linear dynamics,
#' options are \code{"bsf"} (bootstrap particle filter, default for 
#' non-linear models) 
#' and \code{"psi"} (\eqn{\psi}-APF, the default for other models), and 
#' for non-linear models option \code{"ekf"} (extended Kalman particle filter) 
#' is also available.
#' @param particles Number of particles as a positive integer. Suitable values 
#' depend on the model, the data, and the chosen algorithm. While larger values 
#' provide more accurate estimates, the run time also increases with respect to 
#' the number of particles, so it is generally a good idea to test the filter 
#' first with a small number of particles, e.g., less than 100.
#' @param max_iter Maximum number of iterations used in Gaussian approximation,
#' as a positive integer. 
#' Default is 100 (although typically only few iterations are needed).
#' @param conv_tol Positive tolerance parameter used in Gaussian approximation. 
#' Default is 1e-8.
#' @param iekf_iter Non-negative integer. If zero (default), first 
#' approximation for non-linear Gaussian models is obtained from extended 
#' Kalman filter. If \code{iekf_iter > 0}, iterated extended Kalman filter is
#' used with \code{iekf_iter} iterations.
#' @param ... Ignored.
#' @return List with samples (\code{alpha}) from the smoothing distribution 
#' and corresponding weights (\code{weights}),
#'  as well as smoothed means and covariances (\code{alphahat} and \code{Vt}) 
#'  of the states and 
#'  estimated log-likelihood (\code{logLik}).
#' @references 
#' Gordon, NJ, Salmond, DJ, Smith, AFM (1993). 
#' Novel approach to nonlinear/non-Gaussian Bayesian state estimation. 
#' IEE Proceedings-F, 140, 107-113.
#' https://doi.org/10.1049/ip-f-2.1993.0015 
#' 
#' Vihola, M, Helske, J, Franks, J. Importance sampling type estimators 
#' based on approximate marginal Markov chain Monte Carlo. 
#' Scand J Statist. 2020; 1-38. 
#' https://doi.org/10.1111/sjos.12492
#' 
#' Van Der Merwe, R, Doucet, A, De Freitas, N,  Wan, EA (2001). 
#' The unscented particle filter. 
#' In Advances in neural information processing systems, p 584-590.
#' 
#' Jazwinski, A 1970. Stochastic Processes and Filtering Theory. 
#' Academic Press.
#' 
#' Kitagawa, G (1996). Monte Carlo filter and smoother for non-Gaussian 
#' nonlinear state space models. 
#' Journal of Computational and Graphical Statistics, 5, 1-25.
#' https://doi.org/10.2307/1390750
#' 
#' @export
#' @rdname particle_smoother
particle_smoother <- function(model, particles, ...) {
  UseMethod("particle_smoother", model)
}

#' @method particle_smoother lineargaussian
#' @export
#' @rdname particle_smoother
#' @examples 
#' set.seed(1)
#' x <- cumsum(rnorm(100))
#' y <- rnorm(100, x)
#' model <- ssm_ulg(y, Z = 1, T = 1, R = 1, H = 1, P1 = 1)
#' system.time(out <- particle_smoother(model, particles = 1000))
#' # same with simulation smoother:
#' system.time(out2 <- sim_smoother(model, particles = 1000, 
#'   use_antithetic = TRUE))
#' ts.plot(out$alphahat, rowMeans(out2), col = 1:2)
#' 
particle_smoother.lineargaussian <- function(model, particles,  method = "psi",
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
    length(model$a1) * particles
  if (particles > 100 & nsamples > 1e10) {
    warning(paste("Trying to sample ", nsamples, 
      "particles, you might run out of memory."))
  }
  seed <- check_intmax(seed, "seed", FALSE, max = .Machine$integer.max)
  
  if (method == "psi") {
    out <- list()
    out$alpha <- gaussian_psi_smoother(model, particles, seed, 
      model_type(model))
    out$alphahat <- t(apply(out$alpha, 1:2, mean))
    if (ncol(out$alphahat) == 1L) {
      out$Vt <- array(apply(out$alpha[1, , ], 1, var), 
        c(1, 1, nrow(out$alphahat)))
    } else {
      out$Vt <- array(NA, c(ncol(out$alphahat), ncol(out$alphahat), 
        nrow(out$alphahat)))
      for (i in seq_len(nrow(out$alphahat))) {
        out$Vt[, , i] <- cov(t(out$alpha[, i, ]))
      }
    }
  } else {
    out <- bsf_smoother(model, particles, seed, TRUE, model_type(model))
    
  }
  colnames(out$alphahat) <- colnames(out$Vt) <-
    rownames(out$Vt) <- names(model$a1)
  out$Vt <- out$Vt[, , -nrow(out$alphahat), drop = FALSE]
  out$alphahat <- ts(out$alphahat[-nrow(out$alphahat), , drop = FALSE], 
    start = start(model$y), frequency = frequency(model$y))
  rownames(out$alpha) <- names(model$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}
#' @rdname particle_smoother
#' @method particle_smoother nongaussian
#' @export
particle_smoother.nongaussian <- function(model, particles, 
  method = "psi", 
  seed = sample(.Machine$integer.max, size = 1), 
  max_iter = 100, conv_tol = 1e-8, ...) {
  
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
    length(model$a1) * particles
  if (particles > 100 & nsamples > 1e10) {
    warning(paste("Trying to sample ", nsamples, 
      "particles, you might run out of memory."))
  }
  seed <- check_intmax(seed, "seed", FALSE, max = .Machine$integer.max)
  
  model$max_iter <- check_intmax(max_iter, "max_iter", positive = FALSE)
  model$conv_tol <- check_positive_real(conv_tol, "conv_tol")
  
  method <- match.arg(tolower(method), c("bsf", "psi"))

  model$distribution <- pmatch(model$distribution,
    c("svm", "poisson", "binomial", "negative binomial", "gamma", "gaussian"), 
    duplicates.ok = TRUE) - 1
  
  if (method == "psi") {
    out <- psi_smoother(model, particles, seed, model_type(model))
  } else {
    out <- bsf_smoother(model, particles, seed, FALSE, model_type(model))
  }
  colnames(out$alphahat) <- colnames(out$Vt) <-
    colnames(out$Vt) <- names(model$a1)
  out$Vt <- out$Vt[, , -nrow(out$alphahat), drop = FALSE]
  out$alphahat <- ts(out$alphahat[-nrow(out$alphahat), , drop = FALSE], 
    start = start(model$y), frequency = frequency(model$y))
  rownames(out$alpha) <- names(model$a1)
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}

#' @rdname particle_smoother
#' @method particle_smoother ssm_nlg
#' @export
particle_smoother.ssm_nlg <- function(model, particles, 
  method = "bsf", 
  seed = sample(.Machine$integer.max, size = 1),
  max_iter = 100, conv_tol = 1e-8, iekf_iter = 0, ...) {
  
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
  max_iter <- check_intmax(max_iter, "max_iter", positive = FALSE)
  conv_tol <- check_positive_real(conv_tol, "conv_tol")
  iekf_iter <- check_intmax(iekf_iter, "iekf_iter", positive = FALSE)
  
  method <- match.arg(tolower(method), c("bsf", "psi", "ekf"))
  
  out <- switch(method,
    psi = psi_smoother_nlg(t(model$y), model$Z, model$H, model$T, 
      model$R, model$Z_gn, model$T_gn, model$a1, model$P1, 
      model$theta, model$log_prior_pdf, model$known_params, 
      model$known_tv_params, model$n_states, model$n_etas, 
      as.integer(model$time_varying), particles, seed,
      max_iter, conv_tol, iekf_iter),
    bsf = bsf_smoother_nlg(t(model$y), model$Z, model$H, model$T, 
      model$R, model$Z_gn, model$T_gn, model$a1, model$P1, 
      model$theta, model$log_prior_pdf, model$known_params, 
      model$known_tv_params, model$n_states, model$n_etas, 
      as.integer(model$time_varying), particles, seed),
    ekf = ekpf_smoother(t(model$y), model$Z, model$H, model$T, 
      model$R, model$Z_gn, model$T_gn, model$a1, model$P1, 
      model$theta, model$log_prior_pdf, model$known_params, 
      model$known_tv_params, model$n_states, model$n_etas, 
      as.integer(model$time_varying), particles, 
      seed)
  )
  colnames(out$alphahat) <- colnames(out$Vt) <-
    colnames(out$Vt) <- model$state_names
  out$Vt <- out$Vt[, , -nrow(out$alphahat), drop = FALSE]
  out$alphahat <- ts(out$alphahat[-nrow(out$alphahat), , drop = FALSE], 
    start = start(model$y), frequency = frequency(model$y))
  rownames(out$alpha) <- model$state_names
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}


#' @rdname particle_smoother
#' @method particle_smoother ssm_sde
#' @param L Positive integer defining the discretization level for SDE model.
#' @export
particle_smoother.ssm_sde <- function(model, particles, L, 
  seed = sample(.Machine$integer.max, size = 1), ...) {
  
  check_missingness(model)
  
  if (L < 1) stop("Discretization level L must be larger than 0.")
  
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
  out <-  bsf_smoother_sde(model$y, model$x0, model$positive, 
    model$drift, model$diffusion, model$ddiffusion, 
    model$prior_pdf, model$obs_pdf, model$theta, 
    particles, round(L), seed)
  
  colnames(out$alphahat) <- colnames(out$Vt) <-
    colnames(out$Vt) <- model$state_names
  out$Vt <- out$Vt[, , -nrow(out$alphahat), drop = FALSE]
  out$alphahat <- ts(out$alphahat[-nrow(out$alphahat), , drop = FALSE], 
    start = start(model$y), frequency = frequency(model$y))
  rownames(out$alpha) <- model$state_names
  out$alpha <- aperm(out$alpha, c(2, 1, 3))
  out
}

