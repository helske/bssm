#' Non-Gaussian Basic Structural (Time Series) Model
#'
#' Constructs a non-Gaussian basic structural model with local level or
#' local trend component, a seasonal component, and regression component
#' (or subset of these components).
#'
#' @param y Vector or a \code{\link{ts}} object of observations.
#' @param sd_level Standard error of level equation. Used as an initial
#' value in MCMC. If missing, \code{sd_level} is fixed to zero.
#' @param sd_slope Standard error of slope equation. Used as an initial
#' value in MCMC. If missing, \code{sd_slope} is fixed to zero.
#' Ignored if \code{slope = FALSE}.
#' @param sd_seasonal Standard error of seasonal equation. Used as an initial
#' value in MCMC. If missing, \code{sd_seasonal} is fixed to zero.
#' @param sd_noise Standard error of additional noise term. Used as an initial
#' value in MCMC. If missing, additional noise term is omitted from the model.
#' @param distribution distribution of the observation. Possible choices are
#' \code{"poisson"} and \code{"binomial"}.
#' @param phi Additional parameter vector relating to the non-Gaussian distribution.
#' For Poisson distribution, this corresponds to offset term. For binomial, this
#' is the number of trials.
#' @param xreg Matrix containing covariates.
#' @param beta Regression coefficients. Used as an initial
#' value in MCMC. Defaults to vector of zeros.
#' @param period Length of the seasonal component i.e. the number of
#' observations per season. Default is \code{frequency(y)}.
#' @param slope Should the model contain the slope term. Default is \code{TRUE}.
#' If \code{FALSE}, local level component is used instead of local trend component.
#' @param seasonal Should the model contain seasonal term. Default is
#' \code{frequency(y) > 1}.
#' @param a1 Prior means for the initial states (level, slope, seasonals).
#' Defaults to vector of zeros.
#' @param P1 Prior covariance for the initial states (level, slope, seasonals).
#' Default is diagonal matrix with 1e5 on the diagonal.
#' @param prior_type Vector defining the prior types for standard deviations and beta.
#'  Possible values are \code{"uniform"} (default) and \code{"normal"}, where latter
#'  is a half-Normal distribution for standard deviation parameters and 
#'  zero-mean Normal distribution for beta parameters.
#' @param lower_prior,upper_prior Lower and upper bounds for the uniform prior
#' on standard deviations (sd_level, sd_slope, sd_seasonal) and regression
#' coefficients. Defaults to zero for lower bound and and
#' \code{sd(init_signal)} for upper bound of standard deviations, and
#' (-1000, 1000) for regression coefficients.
#' @return Object of class \code{ng_bsm}.
#' @export
#' @examples
#' model <- ng_bsm(Seatbelts[, "VanKilled"], distribution = "poisson",
#'   sd_level = 0.01, sd_seasonal = 0.01, slope = FALSE,
#'   xreg = Seatbelts[, "law"])
#' \dontrun{
#' set.seed(123)
#' mcmc_out <- run_mcmc(model, n_iter = 5000, nsim = 20)
#' mcmc_out$acceptance_rate
#' plot(mcmc_out$theta)
#' summary(mcmc_out$theta)
#'
#' require("ggplot2")
#' ggplot(as.data.frame(mcmc_out$theta[,1:2]), aes(x = sd_level, y = sd_seasonal)) +
#'   geom_point() + stat_density2d(aes(fill = ..level.., alpha = ..level..),
#'   geom = "polygon") + scale_fill_continuous(low = "green",high = "blue") +
#'   guides(alpha = "none")
#'
#' pred <- predict(model, n_iter = 5000, nsim_states = 25, n_ahead = 36,
#'   probs = seq(0.05, 0.95, by = 0.05), newdata = matrix(1, 36, 1),
#'   newphi = rep(1, 36))
#' autoplot(pred)
#' }
ng_bsm <- function(y, sd_level, sd_slope, sd_seasonal, sd_noise,
  distribution, phi = 1, beta, xreg = NULL,
  period = frequency(y), slope = TRUE, seasonal = frequency(y) > 1, a1, P1) {

  ## negative binomial not working currently, need to work out prior for phi
  
  check_y(y)
  n <- length(y)
  
  if (period == 1) {
    seasonal <- FALSE
  } else {
    if (missing(seasonal)) {
      seasonal <- TRUE
    }
  }
  
  #easier this way...
  notfixed <- c("level" = 1, "slope" = 1, "seasonal" = 1)
  
  npar_R <- !missing(sd_level) + !missing(sd_slope) + 
    !missing(sd_seasonal) & c(TRUE, slope, seasonal) + !missing(sd_noise)
  
  npar <- 1L + npar_R
  
  if(!missing(beta)) {
    npar <- npar + length(beta$init)
  }
  
  
  if (missing(sd_noise)) {
    noise <- FALSE
    sd_noise <- NULL
  } else {
    check_sd(sd_noise$init, "noise")
    noise <- TRUE
  }


  if (is.null(xreg)) {
    
    xreg <- matrix(0, 0, 0)
    coefs <- numeric(0)
    beta <- NULL
    
  } else {
    
    if (missing(beta)) {
      stop("No prior defined for beta. ")
    } 
    if (is.null(dim(xreg)) && length(xreg) == n) {
      xreg <- matrix(xreg, n, 1)
    }
    
    check_xreg(xreg, n)
    check_beta(beta$init, ncol(xreg))
    coefs <- beta$init    
    if (is.null(colnames(xreg))) {
      colnames(xreg) <- paste0("coef_",1:ncol(xreg))
    }
    names(coefs) <- colnames(xreg)
    
  }
  
  check_sd(sd$init, "y")
  
  if (missing(sd_level)) {
    notfixed[1] <- 0
  sd_level <- NULL
    } else {
    check_sd(sd_level, "level")
  }
  
  if (slope) {
    if (missing(sd_slope)) {
      notfixed[2] <- 0
      sd_slope <- NULL
    } else {
      check_sd(sd_slope$init, "slope")
    }
  } sd_slope <- NULL
  
  if (seasonal) {
    if (missing(sd_seasonal)) {
      notfixed[3] <- 0
      sd_seasonal <- NULL
    } else {
      check_sd(sd_seasonal$init, "seasonal")
    }
    seasonal_names <- paste0("seasonal_", 1:(period - 1))
  } else {
    seasonal_names <- NULL
    sd_seasonal <- NULL
  }
  

  m <- as.integer(1L + slope + seasonal * (period - 1) + noise)

  if (missing(a1)) {
    a1 <- numeric(m)
  } else {
    if (length(a1) != m) {
      stop("Argument a1 must be a vector of length ", m)
    }
  }
  if (missing(P1)) {
    P1 <- diag(1e3, m)
  } else {
    if (is.null(dim(P1)) && length(P1) == 1L) {
      P1 <- matrix(P1)
    }
    if (!identical(dim(P1), c(m, m))) {
      stop("Argument P1 must be m x m matrix, where m = ", m)
    }
  }

  if (slope) {
    state_names <- c("level", "slope", seasonal_names)
  } else {
    state_names <- c("level", seasonal_names)
  }


  Z <- matrix(0, m, 1)
  Z[1, 1] <- 1

  if (seasonal) {
    Z[2 + slope,1] <- 1
  }

  T <- matrix(0, m, m)
  T[1, 1] <- 1

  if (slope) {
    T[1:2, 2] <- 1
  }

  if (seasonal) {
    T[(2 + slope), (2 + slope):(m - noise)] <- -1
    diag(T[(2 + slope + 1):(m - noise), (2 + slope):(m - 1 - noise)]) <- 1
  }

  if (!(length(phi) %in% c(1, n))) {
    stop("Argument phi must have length 1 or n. ")
  }

  if (length(phi) != n) {
    phi <- rep(phi, length.out = n)
  }

  distribution <- match.arg(distribution, c("poisson", "binomial",
    "negative binomial"))
  nb <- distribution == "negative binomial"
  init_signal <- initial_signal(y, phi, distribution)

  R <- matrix(0, m, max(1, npar_R))
  
  if (notfixed[1]) {
    R[1, 1] <- sd_level$init
  }
  if (slope && notfixed[2]) {
    R[2, 1 + notfixed[1]] <- sd_slopel$init
  }
  if (seasonal && notfixed[3]) {
    R[2 + slope, 1 + notfixed[1] + (slope && notfixed[2])] <- sd_seasonal$init
  }
  
  #additional noise term
  if (noise) {
    P1[m, m] <- sd_noise$init^2
    Z[m] <- 1
    state_names <- c(state_names, "noise")
    R[m, max(1, ncol(R) - 1)] <- sd_noise$init
  }
  dim(T) <- c(m, m, 1)
  dim(R) <- c(m, ncol(R), 1)

  names(a1) <- rownames(P1) <- colnames(P1) <- rownames(Z) <-
    rownames(T) <- colnames(T) <- rownames(R) <- state_names
  
  priors <- c(sd_y, sd_level, sd_slope, sd_seasonal, sd_noise, beta)
  
  names_ind <- c(notfixed & c(TRUE, slope, seasonal), noise)
  
  names(priors) <-
    c(c("sd_level", "sd_slope", "sd_seasonal", "sd_noise")[names_ind], names(beta), if(nb) "nb_dispersion")
  
  if(distribution == "negative binomial") {
    stop("negbin not working at the moment... let me know if you really need it and I'll move it higher on todo list...")
  }
  structure(list(y = as.ts(y), Z = Z, T = T, R = R,
    a1 = a1, P1 = P1, phi = phi, xreg = xreg, coefs = coefs,
    slope = slope, seasonal = seasonal, noise = noise, 
    period = period, fixed = as.integer(!notfixed), priors = priors,
    distribution = distribution, init_signal = init_signal), class = "ng_bsm")
}

#' @method logLik ng_bsm
#' @rdname logLik
#' @inheritParams logLik.ngssm
#' @export
logLik.ng_bsm <- function(object, nsim_states,
  seed = 1, ...) {

  ng_bsm_loglik(object$y, object$Z, object$T, object$R, object$a1,
    object$P1, object$phi, object$slope, object$seasonal, object$noise, object$fixed,
    object$xreg, object$coef,
    pmatch(object$distribution, c("poisson", "binomial", "negative binomial")),
    object$init_signal, nsim_states, seed)
}

#' @method kfilter ng_bsm
#' @rdname kfilter
#' @export
kfilter.ng_bsm <- function(object, ...) {

  out <- ng_bsm_filter(object$y, object$Z, object$T, object$R,
    object$a1, object$P1, object$phi, object$slope, object$seasonal, object$noise,
    object$fixed,  object$xreg, object$coef,
    pmatch(object$distribution, c("poisson", "binomial", "negative binomial")),
    object$init_signal)

  colnames(out$at) <- colnames(out$att) <- colnames(out$Pt) <-
    colnames(out$Ptt) <- rownames(out$Pt) <-
    rownames(out$Ptt) <- names(object$a1)
  out$at <- ts(out$at, start = start(object$y), frequency = object$period)
  out$att <- ts(out$att, start = start(object$y), frequency = object$period)
  out
}

#' @method fast_smoother ng_bsm
#' @export
fast_smoother.ng_bsm <- function(object, ...) {

  out <- ng_bsm_fast_smoother(object$y, object$Z, object$T,
    object$R, object$a1, object$P1, object$phi, object$slope, object$seasonal,
    object$noise, object$fixed, object$xreg, object$coef,
    pmatch(object$distribution, c("poisson", "binomial", "negative binomial")),
    object$init_signal)

  colnames(out) <- names(object$a1)
  ts(out, start = start(object$y), frequency = object$period)
}
#' @method sim_smoother ng_bsm
#' @export
sim_smoother.ng_bsm <- function(object, nsim = 1, seed = sample(.Machine$integer.max, size = 1), ...) {

  out <- ng_bsm_sim_smoother(object$y, object$Z, object$T, object$R,
    object$a1, object$P1, object$phi, nsim, object$slope, object$seasonal,
    object$noise, object$fixed, object$xreg, object$coef,
    pmatch(object$distribution, c("poisson", "binomial", "negative binomial")),
    object$init_signal, seed)

  rownames(out) <- names(object$a1)
  aperm(out, c(2, 1, 3))
}

#' @method smoother ng_bsm
#' @export
smoother.ng_bsm <- function(object, ...) {

  out <- ng_bsm_smoother(object$y, object$Z, object$T, object$R,
    object$a1, object$P1, object$phi, object$slope, object$seasonal,
    object$noise, object$fixed, object$xreg, object$coef,
    pmatch(object$distribution, c("poisson", "binomial", "negative binomial")),
    object$init_signal)

  colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <- names(object$a1)
  out$alphahat <- ts(out$alphahat, start = start(object$y),
    frequency = object$period)
  out
}

#' @method run_mcmc ng_bsm
#' @rdname run_mcmc_ng
#' @param log_space Generate proposals for standard deviations in log-space. Default is \code{TRUE}.
#' @param method Use \code{"standard"} MCMC or \code{"delayed acceptance"} approach.
#' @inheritParams run_mcmc.ngssm
#' @export
run_mcmc.ng_bsm <- function(object, n_iter, nsim_states = 1, type = "full",
  lower_prior, upper_prior, n_burnin = floor(n_iter/2),
  n_thin = 1, gamma = 2/3, target_acceptance = 0.234, S, end_adaptive_phase = TRUE,
  adaptive_approx = TRUE,
  method = "delayed acceptance", log_space = TRUE, n_threads = 1,
  seed = sample(.Machine$integer.max, size = 1), ...) {

  type <- match.arg(type, c("full", "parameters", "summary"))

  method <- match.arg(method, c("standard", "delayed acceptance",
    "IS correction", "block IS correction", "IS2", "DABSF", "BSF"))

  if (n_thin > 1 && method %in% c("block IS correction", "IS2")) {
    stop ("Cannot use thinning with block-IS algorithm.")
  }

  if (missing(lower_prior)) {
    lower_prior <- object$lower_prior
  }
  if (missing(upper_prior)) {
    upper_prior <- object$upper_prior
  }
  nb <- object$distribution == "negative binomial"
  n_sd_par <- length(lower_prior) - ncol(object$xreg) - nb

  if (log_space && n_sd_par > 0) {
    lower_prior[1:(length(lower_prior) - ncol(object$xreg) - nb)] <-
      log(lower_prior[1:(length(lower_prior) - ncol(object$xreg) - nb)])
    upper_prior[1:(length(lower_prior) - ncol(object$xreg) - nb)] <-
      log(upper_prior[1:(length(lower_prior) - ncol(object$xreg) - nb)])
  }

  if (missing(S)) {
    sd_init <- sd(object$init_signal, na.rm = TRUE)
    if (log_space) {
      sd_init <- abs(log(sd_init))
    }
    S <- diag(pmin(c(rep(0.1 * sd_init, length.out = n_sd_par),
      pmax(1, abs(object$coef)), if (nb) 1),
      abs(upper_prior - lower_prior)), length(lower_prior))
  }
  if (nsim_states < 2) {
    #approximate inference
    method <- "standard"
    nsim_states <- 1
  }

  out <-  switch(type,
    full = {
      out <- ng_bsm_mcmc_full(object$y, object$Z, object$T, object$R,
        object$a1, object$P1, object$phi,
        pmatch(object$distribution, c("poisson", "binomial", "negative binomial")),
        lower_prior, upper_prior, n_iter,
        nsim_states, n_burnin, n_thin, gamma, target_acceptance, S, object$slope,
        object$seasonal, object$noise, object$fixed, object$xreg, object$coef,
        object$init_signal, pmatch(method,  c("standard", "delayed acceptance",
          "IS correction", "block IS correction", "IS2", "DABSF","BSF")), seed, log_space,
        n_threads, end_adaptive_phase, adaptive_approx)

      out$alpha <- aperm(out$alpha, c(2, 1, 3))
      colnames(out$alpha) <- names(object$a1)
      out
    },
    parameters = {
      ng_bsm_mcmc_param(object$y, object$Z, object$T, object$R,
        object$a1, object$P1, object$phi,
        pmatch(object$distribution, c("poisson", "binomial", "negative binomial")),
        lower_prior, upper_prior, n_iter,
        nsim_states, n_burnin, n_thin, gamma, target_acceptance, S, object$slope,
        object$seasonal, object$noise, object$fixed, object$xreg, object$coef,
        object$init_signal, pmatch(method,  c("standard", "delayed acceptance",
          "IS correction", "block IS correction", "IS2")), seed, log_space,
        n_threads, end_adaptive_phase, adaptive_approx)
    },
    summary = {
      out <- ng_bsm_mcmc_summary(object$y, object$Z, object$T, object$R,
        object$a1, object$P1, object$phi,
        pmatch(object$distribution, c("poisson", "binomial", "negative binomial")),
        lower_prior, upper_prior, n_iter,
        nsim_states, n_burnin, n_thin, gamma, target_acceptance, S, object$slope,
        object$seasonal, object$noise, object$fixed, object$xreg, object$coef,
        object$init_signal, pmatch(method,  c("standard", "delayed acceptance",
          "IS correction", "block IS correction", "IS2")), seed, log_space,
        n_threads, end_adaptive_phase, adaptive_approx)

      colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <- names(object$a1)
      out$alphahat <- ts(out$alphahat, start = start(object$y),
        frequency = object$period)
      out$muhat <- ts(out$muhat, start = start(object$y),
        frequency = object$period)
      out
    })

  if (log_space && n_sd_par > 0) {
    out$theta[, 1:n_sd_par] <- exp(out$theta[, 1:n_sd_par])
  }
  if (nb) {
    out$theta[, ncol(out$theta)] <- exp(out$theta[, ncol(out$theta)])
  }
  out$S <- matrix(out$S, length(lower_prior), length(lower_prior))
  names_ind <-
    c(!object$fixed & c(TRUE, object$slope, object$seasonal), object$noise)
  colnames(out$theta) <- rownames(out$S) <- colnames(out$S) <-
    c(c("sd_level", "sd_slope", "sd_seasonal", "sd_noise")[names_ind],
      colnames(object$xreg), if (nb) "nb_dispersion")
  if(method %in% c("standard", "delayed acceptance")) {
    out$theta <- mcmc(out$theta, start = n_burnin + 1, thin = n_thin)
  }
  out$call <- match.call()
  class(out) <- "mcmc_output"
  out
}

#' @method predict ng_bsm
#' @rdname predict.ngssm
#' @export
predict.ng_bsm <- function(object, n_iter, nsim_states, lower_prior, upper_prior,
  newdata = NULL, n_ahead = 1, interval = "mean", probs = c(0.05, 0.95),
  n_burnin = floor(n_iter/2), n_thin = 1,
  gamma = 2/3, target_acceptance = 0.234, S,
  seed = sample(.Machine$integer.max, size = 1), newphi = NULL, log_space = FALSE, ...) {

  interval <- pmatch(interval, c("mean", "response"))

  if (missing(lower_prior)) {
    lower_prior <- object$lower_prior
  }
  if (missing(upper_prior)) {
    upper_prior <- object$upper_prior
  }
  nb <- object$distribution == "negative binomial"
  n_sd_par <- length(lower_prior) - ncol(object$xreg) - nb

  if (log_space && n_sd_par > 0) {
    lower_prior[1:(length(lower_prior) - ncol(object$xreg) - nb)] <-
      log(lower_prior[1:(length(lower_prior) - ncol(object$xreg) - nb)])
    upper_prior[1:(length(lower_prior) - ncol(object$xreg) - nb)] <-
      log(upper_prior[1:(length(lower_prior) - ncol(object$xreg) - nb)])
  }

  if (missing(S)) {
    sd_init <- sd(object$init_signal, na.rm = TRUE)
    if (log_space) {
      sd_init <- abs(log(sd_init))
    }
    S <- diag(pmin(c(rep(0.1 * sd_init, length.out = n_sd_par),
      pmax(1, abs(object$coef)), if(nb) 1),
      abs(upper_prior - lower_prior)), length(lower_prior))
  }


  endtime <- end(object$y) + c(0, n_ahead)
  y <- c(object$y, rep(NA, n_ahead))

  if (length(object$coef) > 0) {
    if (!is.null(newdata) && (nrow(newdata) != n_ahead ||
        ncol(newdata) != length(object$coef))) {
      stop("Model contains regression part but dimensions of newdata does not match with n_ahead and length of beta. ")
    }
    if (is.null(newdata)) {
      newdata <- matrix(0, n_ahead, length(object$coef))
    }
    object$xreg <- rbind(object$xreg, newdata)
  }
  if (nb) {
    phi <- c(object$phi, rep(object$phi[length(object$phi)], n_ahead))
  } else {
    if (is.null(newphi)) {
      phi <- c(object$phi, rep(1, n_ahead))
    } else {
      if (length(newphi) != n_ahead) {
        stop("Length of newphi is not equal to n_ahead. ")
      } else {
        phi <- c(object$phi, newphi)
      }
    }
  }
  probs <- sort(unique(c(probs, 0.5)))
  out <- ng_bsm_predict2(y, object$Z, object$T, object$R,
    object$a1, object$P1, phi,
    pmatch(object$distribution, c("poisson", "binomial", "negative binomial")),
    lower_prior, upper_prior, n_iter,
    nsim_states, n_burnin, n_thin, gamma, target_acceptance, S, n_ahead, interval,
    object$slope, object$seasonal, object$noise, object$fixed, object$xreg, object$coef,
    c(object$init_signal, rep(log(0.1), n_ahead)), seed, log_space)

  if (interval == 1 && (object$distribution != "negative binomial")) {
    object$y <- object$y / object$phi
  }
  pred <- list(y = object$y, mean = ts(rowMeans(out), end = endtime, frequency = object$period),
    intervals = ts(t(apply(out, 1, quantile, probs, type = 8)), end = endtime, frequency = object$period,
      names = paste0(100 * probs, "%")))


  class(pred) <- "predict_bssm"
  pred

}


#' @method importance_sample ng_bsm
#' @rdname importance_sample
#' @export
importance_sample.ng_bsm <- function(object, nsim,
  seed = sample(.Machine$integer.max, size = 1), ...) {

  ng_bsm_importance_sample(object$y, object$Z, object$T, object$R, object$a1,
    object$P1, object$phi, object$slope, object$seasonal, object$noise, object$fixed,
    object$xreg, object$coef,
    pmatch(object$distribution, c("poisson", "binomial", "negative binomial")),
    object$init_signal, nsim, seed)
}


#' @method gaussian_approx ng_bsm
#' @rdname gaussian_approx
#' @export
gaussian_approx.ng_bsm <- function(object, max_iter =  100, conv_tol = 1e-8, ...) {

  ng_bsm_approx_model(object$y, object$Z, object$T, object$R, object$a1,
    object$P1, object$phi, object$slope, object$seasonal, object$noise, object$fixed,
    object$xreg, object$coef,
    pmatch(object$distribution, c("poisson", "binomial", "negative binomial")),
    object$init_signal, max_iter, conv_tol)
}

#' @method particle_filter ng_bsm
#' @rdname particle_filter
#' @export
particle_filter.ng_bsm <- function(object, nsim,
  seed = sample(.Machine$integer.max, size = 1), ...) {
  
  ng_bsm_particle_filter(object$y, object$Z, object$T, object$R, object$a1,
    object$P1, object$phi, object$slope, object$seasonal, object$noise, object$fixed,
    object$xreg, object$coef,
    pmatch(object$distribution, c("poisson", "binomial", "negative binomial")),
    object$init_signal, nsim, seed)
}
