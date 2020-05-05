#' Basic Structural (Time Series) Model
#'
#' Constructs a basic structural model with local level or local trend component
#' and seasonal component.
#'
#' @param y Vector or a \code{\link{ts}} object of observations.
#' @param sd_y A fixed value or prior for the standard error of
#' observation equation. See \link[=uniform]{priors} for details.
#' @param sd_level A fixed value or a prior for the standard error
#' of the noise in level equation. See \link[=uniform]{priors} for details.
#' @param sd_slope A fixed value or a prior for the standard error
#' of the noise in slope equation. See \link[=uniform]{priors} for details.
#' If missing, the slope term is omitted from the model.
#' @param sd_seasonal A fixed value or a prior for the standard error
#' of the noise in seasonal equation. See \link[=uniform]{priors} for details.
#' If missing, the seasonal component is omitted from the model.
#' @param xreg Matrix containing covariates.
#' @param beta Prior for the regression coefficients.
#' @param period Length of the seasonal component i.e. the number of
#' @param a1 Prior means for the initial states (level, slope, seasonals).
#' Defaults to vector of zeros.
#' @param P1 Prior covariance for the initial states (level, slope, seasonals).
#' Default is diagonal matrix with 1000 on the diagonal.
#' @param obs_intercept,state_intercept Intercept terms for observation and
#' state equations, given as a length n vector and m times n matrix respectively.
#' @return Object of class \code{bsm}.
#' @export
#' @examples
#'
#' prior <- uniform(0.1 * sd(log10(UKgas)), 0, 1)
#' model <- bsm(log10(UKgas), sd_y = prior, sd_level =  prior,
#'   sd_slope =  prior, sd_seasonal =  prior)
#'
#' mcmc_out <- run_mcmc(model, n_iter = 5000)
#' summary(expand_sample(mcmc_out, "theta"))$stat
#' mcmc_out$theta[which.max(mcmc_out$posterior), ]
#' sqrt((fit <- StructTS(log10(UKgas), type = "BSM"))$coef)[c(4, 1:3)]
#'
bsm <- function(y, sd_y, sd_level, sd_slope, sd_seasonal,
  beta, xreg = NULL, period = frequency(y), a1, P1, obs_intercept, state_intercept) {
  
  check_y(y)
  n <- length(y)
  
  if (is.null(xreg)) {
    xreg <- matrix(0, 0, 0)
    coefs <- numeric(0)
    beta <- NULL
  } else {
    
    if (missing(beta) || is.null(beta)) {
      stop("No prior defined for beta. ")
    }
    if(!is_prior(beta) && !is_prior_list(beta)) {
      stop("Prior for beta must be of class 'bssm_prior' or 'bssm_prior_list.")
    }
    
    if (is.null(dim(xreg)) && length(xreg) == n) {
      xreg <- matrix(xreg, n, 1)
    }
    
    check_xreg(xreg, n)
    nx <- ncol(xreg)
    if (nx == 1 && is_prior_list(beta)) beta <- beta[[1]]
    if(nx > 1) {
      coefs <- sapply(beta, "[[", "init")
    } else {
      coefs <- beta$init
    }
    check_beta(coefs, nx)
    if (is.null(colnames(xreg))) {
      colnames(xreg) <- paste0("coef_",1:ncol(xreg))
    }
    names(coefs) <- colnames(xreg)
  }
  
  notfixed <- c("y" = 1, "level" = 1, "slope" = 1, "seasonal" = 1)
  
  
  if (missing(sd_y) || is.null(sd_y)) {
    stop("Provide either prior or fixed value for sd_y.")
  } else {
    if (is_prior(sd_y)) {
      check_sd(sd_y$init, "y")
      H <- matrix(sd_y$init)
    } else {
      notfixed[1] <- 0
      check_sd(sd_y, "y")
      H <- matrix(sd_y)
    }
  }
  
  if (missing(sd_level) || is.null(sd_level)) {
    stop("Provide either prior or fixed value for sd_level.")
  } else {
    if (is_prior(sd_level)) {
      check_sd(sd_level$init, "level")
    } else {
      notfixed["level"] <- 0
      check_sd(sd_level, "level")
    }
  }
  
  if (missing(sd_slope) || is.null(sd_slope)) {
    notfixed["slope"] <- 0
    slope <- FALSE
    sd_slope <- NULL
  } else {
    if (is_prior(sd_slope)) {
      check_sd(sd_slope$init, "slope")
    } else {
      notfixed["slope"] <- 0
      check_sd(sd_slope, "slope")
    }
    slope <- TRUE
  }
  
  if (missing(sd_seasonal) || is.null(sd_seasonal)) {
    notfixed["seasonal"] <- 0
    seasonal_names <- NULL
    seasonal <- FALSE
    sd_seasonal <- NULL
  } else {
    if (period < 2) {
      stop("Period of seasonal component must be larger than 1. ")
    }
    if (is_prior(sd_seasonal)) {
      check_sd(sd_seasonal$init, "seasonal")
    } else {
      notfixed["seasonal"] <- 0
      check_sd(sd_seasonal, "seasonal")
    }
    seasonal <- TRUE
    seasonal_names <- paste0("seasonal_", 1:(period - 1))
  }
  
  npar_R <- 1L + as.integer(slope) + as.integer(seasonal)
  
  m <- as.integer(1L + as.integer(slope) + as.integer(seasonal) * (period - 1))
  
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
    Z[2 + slope, 1] <- 1
  }
  
  T <- matrix(0, m, m)
  T[1, 1] <- 1
  if (slope) {
    T[1:2, 2] <- 1
  }
  if (seasonal) {
    T[(2 + slope), (2 + slope):m] <- -1
    T[cbind(1 + slope + 2:(period - 1), 1 + slope + 1:(period - 2))] <- 1
  }
  
  R <- matrix(0, m, max(1, npar_R))
  
  if (notfixed["level"]) {
    R[1, 1] <- sd_level$init
  } else {
    R[1, 1] <- sd_level
  }
  if (slope) {
    if (notfixed["slope"]) {
      R[2, 2] <- sd_slope$init
    } else {
      R[2, 2] <- sd_slope
    }
  }
  if (seasonal) {
    if (notfixed["seasonal"]) {
      R[2 + slope, 2 + slope] <- sd_seasonal$init
    } else {
      R[2 + slope, 2 + slope] <- sd_seasonal
    }
  }
  
  dim(H) <- 1
  dim(T) <- c(m, m, 1)
  dim(R) <- c(m, ncol(R), 1)
  
  names(a1) <- rownames(P1) <- colnames(P1) <- rownames(Z) <-
    rownames(T) <- colnames(T) <- rownames(R) <- state_names
  
  
  if(ncol(xreg) > 1) {
    priors <- c(list(sd_y, sd_level, sd_slope, sd_seasonal), beta)
  } else {
    priors <- list(sd_y, sd_level, sd_slope, sd_seasonal, beta)
  }
  names(priors) <- c("sd_y", "sd_level", "sd_slope", "sd_seasonal", names(coefs))
  priors <- priors[sapply(priors, is_prior)]
  
  if (!missing(obs_intercept)) {
    check_obs_intercept(obs_intercept, 1L, n)
  } else {
    obs_intercept <- matrix(0)
  }
  if (!missing(state_intercept)) {
    check_state_intercept(state_intercept, m, n)
  } else {
    state_intercept <- matrix(0, m, 1)
  }
  theta <- if (length(priors) > 0) sapply(priors, "[[", "init") else numeric(0)
  priors <- combine_priors(priors)
  
  structure(list(y = as.ts(y), Z = Z, H = H, T = T, R = R,
    a1 = a1, P1 = P1, xreg = xreg, coefs = coefs,
    obs_intercept = obs_intercept,
    state_intercept = state_intercept,
    slope = slope, seasonal = seasonal, period = period, 
    fixed = as.integer(!notfixed), 
    prior_distributions = priors$prior_distribution, prior_parameters = priors$parameters,
    theta = theta), class = c("bsm", "gssm"))
}

#' Non-Gaussian Basic Structural (Time Series) Model
#'
#' Constructs a non-Gaussian basic structural model with local level or
#' local trend component, a seasonal component, and regression component
#' (or subset of these components).
#'
#' @param y Vector or a \code{\link{ts}} object of observations.
#' @param sd_level A fixed value or a prior for the standard error
#' of the noise in level equation. See \link[=uniform]{priors} for details.
#' @param sd_slope A fixed value or a prior for the standard error
#' of the noise in slope equation. See \link[=uniform]{priors} for details.
#' If missing, the slope term is omitted from the model.
#' @param sd_seasonal A fixed value or a prior for the standard error
#' of the noise in seasonal equation. See \link[=uniform]{priors} for details.
#' If missing, the seasonal component is omitted from the model.
#' @param sd_noise Prior for the standard error of the additional noise term.
#' See \link[=uniform]{priors} for details. If missing, no additional noise term is used.
#' @param distribution distribution of the observation. Possible choices are
#' \code{"poisson"}, \code{"binomial"}, \code{"negative binomial"}.
#' @param phi Additional parameter relating to the non-Gaussian distribution.
#' For Negative binomial distribution this is the dispersion term, and for other
#' distributions this is ignored.
#' @param u Constant parameter for non-Gaussian models. For Poisson and negative binomial distribution, this corresponds to the offset
#' term. For binomial, this is the number of trials.
#' @param beta Prior for the regression coefficients.
#' @param xreg Matrix containing covariates.
#' @param period Length of the seasonal component i.e. the number of
#' observations per season. Default is \code{frequency(y)}.
#' @param a1 Prior means for the initial states (level, slope, seasonals).
#' Defaults to vector of zeros.
#' @param P1 Prior covariance for the initial states (level, slope, seasonals).
#' Default is diagonal matrix with 1e5 on the diagonal.
#' @param state_intercept Intercept terms for state equation, given as a
#'  m times n matrix.
#' @return Object of class \code{ng_bsm}.
#' @export
#' @examples
#' model <- ng_bsm(Seatbelts[, "VanKilled"], distribution = "poisson",
#'   sd_level = halfnormal(0.01, 1),
#'   sd_seasonal = halfnormal(0.01, 1),
#'   beta = normal(0, 0, 10),
#'   xreg = Seatbelts[, "law"])
#' \dontrun{
#' set.seed(123)
#' mcmc_out <- run_mcmc(model, n_iter = 5000, nsim = 10)
#' mcmc_out$acceptance_rate
#' theta <- expand_sample(mcmc_out, "theta")
#' plot(theta)
#' summary(theta)
#'
#' library("ggplot2")
#' ggplot(as.data.frame(theta[,1:2]), aes(x = sd_level, y = sd_seasonal)) +
#'   geom_point() + stat_density2d(aes(fill = ..level.., alpha = ..level..),
#'   geom = "polygon") + scale_fill_continuous(low = "green",high = "blue") +
#'   guides(alpha = "none")
#'
# pred <- predict(model, n_iter = 5000, nsim_states = 10, n_ahead = 36,
#   probs = seq(0.05, 0.95, by = 0.05), newdata = matrix(1, 36, 1),
#   newphi = rep(1, 36))
# autoplot(pred)
#' }
ng_bsm <- function(y, sd_level, sd_slope, sd_seasonal, sd_noise,
  distribution, phi, u = 1, beta, xreg = NULL, period = frequency(y), a1, P1,
  state_intercept) {
  
  
  check_y(y)
  n <- length(y)
  
  if (is.null(xreg)) {
    xreg <- matrix(0, 0, 0)
    coefs <- numeric(0)
    beta <- NULL
  } else {
    
    if (missing(beta) || is.null(beta)) {
      stop("No prior defined for beta. ")
    }
    if(!is_prior(beta) && !is_prior_list(beta)) {
      stop("Prior for beta must be of class 'bssm_prior' or 'bssm_prior_list.")
    }
    
    if (is.null(dim(xreg)) && length(xreg) == n) {
      xreg <- matrix(xreg, n, 1)
    }
    
    check_xreg(xreg, n)
    nx <- ncol(xreg)
    if (nx == 1 && is_prior_list(beta)) beta <- beta[[1]]
    if(nx > 1) {
      coefs <- sapply(beta, "[[", "init")
    } else {
      coefs <- beta$init
    }
    check_beta(coefs, nx)
    
    if (is.null(colnames(xreg))) {
      colnames(xreg) <- paste0("coef_",1:ncol(xreg))
    }
    names(coefs) <- colnames(xreg)
    
  }
  
  notfixed <- c("level" = 1, "slope" = 1, "seasonal" = 1)
  
  if (missing(sd_level) || missing(sd_level)) {
    stop("Provide either prior or fixed value for sd_level.")
  } else {
    if (is_prior(sd_level)) {
      check_sd(sd_level$init, "level")
    } else {
      notfixed["level"] <- 0
      check_sd(sd_level, "level")
    }
  }
  if (missing(sd_slope) || is.null(sd_slope)) {
    notfixed["slope"] <- 0
    slope <- FALSE
    sd_slope <- NULL
  } else {
    if (is_prior(sd_slope)) {
      check_sd(sd_slope$init, "slope")
    } else {
      notfixed["slope"] <- 0
      check_sd(sd_slope, "slope")
    }
    slope <- TRUE
  }
  
  if (missing(sd_seasonal) || is.null(sd_seasonal)) {
    notfixed["seasonal"] <- 0
    seasonal_names <- NULL
    seasonal <- FALSE
    sd_seasonal <- NULL
  } else {
    if (period < 2) {
      stop("Period of seasonal component must be larger than 1. ")
    }
    if (is_prior(sd_seasonal)) {
      check_sd(sd_seasonal$init, "seasonal")
    } else {
      notfixed["seasonal"] <- 0
      check_sd(sd_seasonal, "seasonal")
    }
    seasonal <- TRUE
    seasonal_names <- paste0("seasonal_", 1:(period - 1))
  }
  
  if (missing(sd_noise) || is.null(sd_noise)) {
    noise <- FALSE
    sd_noise <- NULL
  } else {
    check_sd(sd_noise$init, "noise")
    noise <- TRUE
  }
  
  npar_R <- 1L + as.integer(slope) + as.integer(seasonal) + as.integer(noise)
  
  m <- as.integer(1L + as.integer(slope) + as.integer(seasonal) * (period - 1) + as.integer(noise))
  
  
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
    Z[2 + slope, 1] <- 1
  }
  
  T <- matrix(0, m, m)
  T[1, 1] <- 1
  if (slope) {
    T[1:2, 2] <- 1
  }
  if (seasonal) {
    T[(2 + slope), (2 + slope):m] <- -1
    T[cbind(1 + slope + 2:(period - 1), 1 + slope + 1:(period - 2))] <- 1
  }
  
  R <- matrix(0, m, max(1, npar_R))
  
  if (notfixed["level"]) {
    R[1, 1] <- sd_level$init
  } else {
    R[1, 1] <- sd_level
  }
  if (slope) {
    if (notfixed["slope"]) {
      R[2, 2] <- sd_slope$init
    } else {
      R[2, 2] <- sd_slope
    }
  }
  if (seasonal) {
    if (notfixed["seasonal"]) {
      R[2 + slope, 2 + slope] <- sd_seasonal$init
    } else {
      R[2 + slope, 2 + slope] <- sd_seasonal
    }
  }
  
  #additional noise term
  if (noise) {
    P1[m, m] <- sd_noise$init^2
    Z[m] <- 1
    state_names <- c(state_names, "noise")
    R[m, max(1, ncol(R) - 1)] <- sd_noise$init
  }
  
  distribution <- match.arg(distribution, c("poisson", "binomial",
    "negative binomial"))
  
  use_phi <- distribution %in% c("negative binomial")
  phi_est <- FALSE
  if (use_phi) {
    if (is_prior(phi)) {
      check_phi(phi$init, distribution)
      phi_est <- TRUE
    } else {
      check_phi(phi, distribution)
    }
  } else {
    phi <- 1
  }
  
  use_u <- distribution %in% c("poisson", "binomial", "negative binomial")
  if (use_u) {
    check_u(u)
    if (length(u) != n) {
      u <- rep(u, length.out = n)
    }
  }
  
  initial_mode <- init_mode(y, u, distribution, if (ncol(xreg) > 0) xreg %*% coefs else NULL)
  
  
  dim(T) <- c(m, m, 1)
  dim(R) <- c(m, ncol(R), 1)
  
  names(a1) <- rownames(P1) <- colnames(P1) <- rownames(Z) <-
    rownames(T) <- colnames(T) <- rownames(R) <- state_names
  
  if(ncol(xreg) > 1) {
    priors <- c(list(sd_level, sd_slope, sd_seasonal, sd_noise, phi), beta)
  } else {
    priors <- list(sd_level, sd_slope, sd_seasonal, sd_noise, phi, beta)
  }
  names(priors) <- c("sd_level", "sd_slope", "sd_seasonal", "sd_noise", "phi",
    names(coefs))
  priors <- priors[sapply(priors, is_prior)]
  
  if (phi_est) {
    phi <- phi$init
  }
  
  obs_intercept <- matrix(0)
  
  if (!missing(state_intercept)) {
    check_state_intercept(state_intercept, m, n)
  } else {
    state_intercept <- matrix(0, m, 1)
  }
  theta <- if (length(priors) > 0) sapply(priors, "[[", "init") else numeric(0)
  priors <- combine_priors(priors)
  
  structure(list(y = as.ts(y), Z = Z, T = T, R = R,
    a1 = a1, P1 = P1, phi = phi, u = u, xreg = xreg, coefs = coefs, 
    obs_intercept = obs_intercept,
    state_intercept = state_intercept,
    slope = slope, seasonal = seasonal, noise = noise,
    period = period, fixed = as.integer(!notfixed),
    distribution = distribution, initial_mode = initial_mode, 
    prior_distributions = priors$prior_distribution, prior_parameters = priors$parameters,
    theta = theta, phi_est = phi_est), class = c("ng_bsm", "ngssm"))
}

#' Stochastic Volatility Model
#'
#' Constructs a simple stochastic volatility model with Gaussian errors and
#' first order autoregressive signal.
#'
#' @param y Vector or a \code{\link{ts}} object of observations.
#' @param rho prior for autoregressive coefficient.
#' @param sigma Prior for sigma parameter of observation equation.
#' @param mu Prior for mu parameter of transition equation.
#' Ignored if \code{sigma} is provided.
#' @param sd_ar Prior for the standard deviation of noise of the AR-process.
#' @return Object of class \code{svm} or \code{svm2}.
#' @export
#' @rdname svm
#' @examples
#'
#' data("exchange")
#' exchange <- exchange[1:100] # faster CRAN check
#' model <- svm(exchange, rho = uniform(0.98,-0.999,0.999),
#'  sd_ar = halfnormal(0.15, 5), sigma = halfnormal(0.6, 2))
#'
#' obj <- function(pars) {
#'    -logLik(svm(exchange, rho = uniform(pars[1],-0.999,0.999),
#'    sd_ar = halfnormal(pars[2],sd=5),
#'    sigma = halfnormal(pars[3],sd=2)), nsim_states = 0)
#' }
#' opt <- nlminb(c(0.98, 0.15, 0.6), obj, lower = c(-0.999, 1e-4, 1e-4), upper = c(0.999,10,10))
#' pars <- opt$par
#' model <- svm(exchange, rho = uniform(pars[1],-0.999,0.999),
#'   sd_ar = halfnormal(pars[2],sd=5),
#'   sigma = halfnormal(pars[3],sd=2))
#'
svm <- function(y, rho, sd_ar, sigma, mu) {
  
  if(!missing(sigma) && !missing(mu)) {
    stop("Define either sigma or mu, but not both.")
  }
  
  check_y(y)
  
  xreg <- matrix(0, 0, 0)
  coefs <- numeric(0)
  beta <- NULL
  
  
  check_rho(rho$init)
  check_sd(sd_ar$init, "rho")
  if(missing(sigma)) {
    svm_type <- 1L
    check_mu(mu$init)
    initial_mode <- log(pmax(1e-4, y^2))
  } else {
    svm_type <- 0L
    check_sd(sigma$init, "sigma", FALSE)
    initial_mode <- log(pmax(1e-4, y^2)) - 2 * log(sigma$init)
  }
  a1 <- if(svm_type) mu$init else 0
  P1 <- matrix(sd_ar$init^2 / (1 - rho$init^2))
  
  Z <- matrix(1)
  T <- array(rho$init, c(1, 1, 1))
  R <- array(sd_ar$init, c(1, 1, 1))
  
  
  
  names(a1) <- rownames(P1) <- colnames(P1) <- rownames(Z) <-
    rownames(T) <- colnames(T) <- rownames(R) <- "signal"
  
  if(ncol(xreg) > 1) {
    priors <- c(list(rho, sd_ar, if(svm_type==0) sigma else mu), beta)
  } else {
    priors <- list(rho, sd_ar, if(svm_type==0) sigma else mu, beta)
  }
  priors <- priors[!sapply(priors, is.null)]
  names(priors) <-
    c("rho", "sd_ar", if(svm_type==0) "sigma" else "mu", names(coefs))
  
  state_intercept <- if (svm_type) matrix(mu$init * (1 - T[1])) else matrix(0)
  obs_intercept <- matrix(0)
  
  theta <- if (length(priors) > 0) sapply(priors, "[[", "init") else numeric(0)
  priors <- combine_priors(priors)
  
  structure(list(y = as.ts(y), Z = Z, T = T, R = R,
    a1 = a1, P1 = P1, phi = if (svm_type == 0) sigma$init else 1, xreg = xreg, 
    coefs = coefs, obs_intercept = obs_intercept, state_intercept = state_intercept, 
    initial_mode = initial_mode, 
    svm_type = svm_type, distribution = 0L, u = 1, phi_est = !as.logical(svm_type),
    prior_distributions = priors$prior_distribution, prior_parameters = priors$parameters,
    theta = theta),
    class = c("svm", "ngssm"))
}
#' Non-Gaussian model with AR(1) latent process
#'
#' Constructs a simple non-Gaussian model where the state dynamics follow an AR(1) process.
#'
#' @param y Vector or a \code{\link{ts}} object of observations.
#' @param rho prior for autoregressive coefficient.
#' @param mu A fixed value or a prior for the stationary mean of the latent AR(1) process. Parameter is omitted if this is set to 0.
#' @param sigma Prior for the standard deviation of noise of the AR-process.
#' @param beta Prior for the regression coefficients.
#' @param xreg Matrix containing covariates.
#' @param distribution distribution of the observation. Possible choices are
#' \code{"poisson"}, \code{"binomial"} and \code{"negative binomial"}.
#' @param phi Additional parameter relating to the non-Gaussian distribution.
#' For Negative binomial distribution this is the dispersion term, and for other
#' distributions this is ignored.
#' @param u Constant parameter for non-Gaussian models. For Poisson and negative binomial distribution, this corresponds to the offset
#' term. For binomial, this is the number of trials.
#' @return Object of class \code{ng_ar1}.
#' @export
#' @rdname ng_ar1
ng_ar1 <- function(y, rho, sigma, mu, distribution, phi, u = 1, beta, xreg = NULL) {
  
  check_y(y)
  n <- length(y)
  if (is.null(xreg)) {
    xreg <- matrix(0, 0, 0)
    coefs <- numeric(0)
    beta <- NULL
  } else {
    
    if (missing(beta) || is.null(beta)) {
      stop("No prior defined for beta. ")
    }
    if(!is_prior(beta) && !is_prior_list(beta)) {
      stop("Prior for beta must be of class 'bssm_prior' or 'bssm_prior_list.")
    }
    n <- length(y)
    if (is.null(dim(xreg)) && length(xreg) == n) {
      xreg <- matrix(xreg, n, 1)
    }
    check_xreg(xreg, n)
    nx <- ncol(xreg)
    if (nx == 1 && is_prior_list(beta)) beta <- beta[[1]]
    if(nx > 1) {
      coefs <- sapply(beta, "[[", "init")
    } else {
      coefs <- beta$init
    }
    check_beta(coefs, nx)
    
    if (is.null(colnames(xreg))) {
      colnames(xreg) <- paste0("coef_",1:ncol(xreg))
    }
    names(coefs) <- colnames(xreg)
    
  }
  
  
  check_rho(rho$init)
  check_sd(sigma$init, "rho")
  
  if (is_prior(mu)) {
    check_mu(mu$init)
    mu_est <- TRUE
    a1 <- mu$init
    state_intercept <- matrix(mu$init * (1 - rho$init))
  } else {
    mu_est <- FALSE
    check_mu(mu)
    a1 <- mu
    state_intercept <- matrix(mu * (1 - rho$init))
  }
  distribution <- match.arg(distribution, c("poisson", "binomial",
    "negative binomial"))
  
  use_phi <- distribution %in% c("negative binomial", "gamma")
  phi_est <- FALSE
  if (use_phi) {
    if (is_prior(phi)) {
      check_phi(phi$init, distribution)
      phi_est <- TRUE
    } else {
      check_phi(phi, distribution)
    }
  } else {
    phi <- 1
  }
  
  use_u <- distribution %in% c("poisson", "binomial", "negative binomial")
  if (use_u) {
    check_u(u)
    if (length(u) != n) {
      u <- rep(u, length.out = n)
    }
  }
  initial_mode <- init_mode(y, u, distribution, if (ncol(xreg) > 0) xreg %*% coefs else NULL)
  P1 <- matrix(sigma$init^2 / (1 - rho$init^2))
  
  Z <- matrix(1)
  T <- array(rho$init, c(1, 1, 1))
  R <- array(sigma$init, c(1, 1, 1))
  
  names(a1) <- rownames(P1) <- colnames(P1) <- rownames(Z) <-
    rownames(T) <- colnames(T) <- rownames(R) <- "signal"
  
  
  if(ncol(xreg) > 1) {
    priors <- c(list(rho, sigma, mu, phi), beta)
  } else {
    priors <- list(rho, sigma, mu, phi, beta)
  }
  names(priors) <-
    c("rho", "sigma", "mu", "phi", names(coefs))
  priors <- priors[sapply(priors, is_prior)]
  
  if (phi_est) {
    phi <- phi$init
  }
  obs_intercept <- matrix(0)
  
  theta <- if (length(priors) > 0) sapply(priors, "[[", "init") else numeric(0)
  priors <- combine_priors(priors)
  
  structure(list(y = as.ts(y), Z = Z, T = T, R = R,
    a1 = a1, P1 = P1, phi = phi, u = u, xreg = xreg, coefs = coefs,
    obs_intercept = obs_intercept, state_intercept = state_intercept,
    initial_mode = initial_mode,
    distribution = distribution, mu_est = mu_est, phi_est = phi_est,
    prior_distributions = priors$prior_distribution, prior_parameters = priors$parameters,
    theta = theta),
    class = c("ng_ar1", "ngssm"))
}
#' Univariate Gaussian model with AR(1) latent process
#'
#' Constructs a simple Gaussian model where the state dynamics follow an AR(1) process.
#'
#' @param y Vector or a \code{\link{ts}} object of observations.
#' @param rho prior for autoregressive coefficient.
#' @param mu A fixed value or a prior for the stationary mean of the latent AR(1) process. Parameter is omitted if this is set to 0.
#' @param sigma Prior for the standard deviation of noise of the AR-process.
#' @param sd_y Prior for the standard deviation of observation equation.
#' @param beta Prior for the regression coefficients.
#' @param xreg Matrix containing covariates.
#' @return Object of class \code{ar1}.
#' @export
#' @rdname ar1
ar1 <- function(y, rho, sigma, mu, sd_y, beta, xreg = NULL) {
  
  check_y(y)
  n <- length(y)
  if (is.null(xreg)) {
    xreg <- matrix(0, 0, 0)
    coefs <- numeric(0)
    beta <- NULL
  } else {
    
    if (missing(beta) || is.null(beta)) {
      stop("No prior defined for beta. ")
    }
    if(!is_prior(beta) && !is_prior_list(beta)) {
      stop("Prior for beta must be of class 'bssm_prior' or 'bssm_prior_list.")
    }
    n <- length(y)
    if (is.null(dim(xreg)) && length(xreg) == n) {
      xreg <- matrix(xreg, n, 1)
    }
    check_xreg(xreg, n)
    nx <- ncol(xreg)
    if (nx == 1 && is_prior_list(beta)) beta <- beta[[1]]
    if(nx > 1) {
      coefs <- sapply(beta, "[[", "init")
    } else {
      coefs <- beta$init
    }
    check_beta(coefs, nx)
    
    if (is.null(colnames(xreg))) {
      colnames(xreg) <- paste0("coef_",1:ncol(xreg))
    }
    names(coefs) <- colnames(xreg)
    
  }
  
  
  check_rho(rho$init)
  check_sd(sigma$init, "rho")
  
  if (is_prior(mu)) {
    check_mu(mu$init)
    mu_est <- TRUE
    a1 <- mu$init
    state_intercept <- matrix(mu$init * (1 - rho$init))
  } else {
    mu_est <- FALSE
    check_mu(mu)
    a1 <- mu
    state_intercept <- matrix(mu * (1 - rho$init))
  }
 
  if (is_prior(sd_y)) {
    check_sd(sd_y$init, "y")
    sd_y_est <- TRUE
    H <- matrix(sd_y$init)
  } else {
    sd_y_est <- FALSE
    check_sd(sd_y, "y")
    H <- matrix(sd_y)
  }
  
 
  P1 <- matrix(sigma$init^2 / (1 - rho$init^2))
  
  Z <- matrix(1)
  T <- array(rho$init, c(1, 1, 1))
  R <- array(sigma$init, c(1, 1, 1))
  
  names(a1) <- rownames(P1) <- colnames(P1) <- rownames(Z) <-
    rownames(T) <- colnames(T) <- rownames(R) <- "signal"
  
  
  if(ncol(xreg) > 1) {
    priors <- c(list(rho, sigma, mu, sd_y), beta)
  } else {
    priors <- list(rho, sigma, mu, sd_y, beta)
  }
  names(priors) <-
    c("rho", "sigma", "mu", "sd_y", names(coefs))
  priors <- priors[sapply(priors, is_prior)]

  obs_intercept <- matrix(0)
  
  theta <- if (length(priors) > 0) sapply(priors, "[[", "init") else numeric(0)
  priors <- combine_priors(priors)
  
  structure(list(y = as.ts(y), Z = Z, H = H, T = T, R = R,
    a1 = a1, P1 = P1, xreg = xreg, coefs = coefs,
    obs_intercept = obs_intercept, state_intercept = state_intercept,
    mu_est = mu_est, sd_y_est = sd_y_est,
    prior_distributions = priors$prior_distribution, prior_parameters = priors$parameters,
    theta = theta),
    class = c("ar1", "gssm"))
}

#'
#' General univariate linear-Gaussian state space models
#'
#' Construct an object of class \code{gssm} by defining the corresponding terms
#' of the observation and state equation:
#'
#' \deqn{y_t = D_t + Z_t \alpha_t + H_t \epsilon_t, (\textrm{observation equation})}
#' \deqn{\alpha_{t+1} = C_t + T_t \alpha_t + R_t \eta_t, (\textrm{transition equation})}
#'
#' where \eqn{\epsilon_t \sim N(0, 1)}, \eqn{\eta_t \sim N(0, I_k)} and
#' \eqn{\alpha_1 \sim N(a_1, P_1)} independently of each other.
#' 
#' The priors are defined for each NA value of the system matrices, in the same order as 
#' these values are naturally read in R. For more flexibility, see \code{\link{lgg_ssm}}.
#' 
#' @param y Observations as time series (or vector) of length \eqn{n}.
#' @param Z System matrix Z of the observation equation. Either a vector of
#' length m or a m x n array, or an object which can be coerced to such.
#' @param H Vector of standard deviations. Either a scalar or a vector of length
#' n, or an object which can be coerced to such.
#' @param T System matrix T of the state equation. Either a m x m matrix or a
#' m x m x n array, or object which can be coerced to such.
#' @param R Lower triangular matrix R the state equation. Either a m x k matrix or a
#' m x k x n array, or object which can be coerced to such.
#' @param a1 Prior mean for the initial state as a vector of length m.
#' @param P1 Prior covariance matrix for the initial state as m x m matrix.
#' @param xreg Matrix containing covariates.
#' @param beta Regression coefficients. Used as an initial
#' value in MCMC. Defaults to vector of zeros.
#' @param state_names Names for the states.
#' @param H_prior,Z_prior,T_prior,R_prior Priors for the NA values in system matrices.
#' @param obs_intercept,state_intercept Intercept terms for observation and
#' state equations, given as a length n vector and m times n matrix respectively.
#' @return Object of class \code{gssm}.
#' @export
gssm <- function(y, Z, H, T, R, a1, P1, xreg = NULL, beta, state_names,
  H_prior, Z_prior, T_prior, R_prior, obs_intercept, state_intercept) {
  
  check_y(y)
  n <- length(y)
  
  if (is.null(xreg)) {
    xreg <- matrix(0, 0, 0)
    coefs <- numeric(0)
    beta <- NULL
  } else {
    if (missing(beta) || is.null(beta)) {
      stop("No prior defined for beta. ")
    }
    if(!is_prior(beta) && !is_prior_list(beta)) {
      stop("Prior for beta must be of class 'bssm_prior' or 'bssm_prior_list.")
    }
    
    if (is.null(dim(xreg)) && length(xreg) == n) {
      xreg <- matrix(xreg, n, 1)
    }
    
    check_xreg(xreg, n)
    nx <- ncol(xreg)
    if (nx == 1 && is_prior_list(beta)) beta <- beta[[1]]
    if(nx > 1) {
      coefs <- sapply(beta, "[[", "init")
    } else {
      coefs <- beta$init
    }
    check_beta(coefs, nx)
    if (is.null(colnames(xreg))) {
      colnames(xreg) <- paste0("coef_",1:ncol(xreg))
    }
    names(coefs) <- colnames(xreg)
    
  }
  if (length(Z) == 1) {
    dim(Z) <- c(1, 1)
    m <- 1
  } else {
    if (!(dim(Z)[2] %in% c(1, NA, n)))
      stop("Argument Z must be a (m x 1) or (m x n) matrix,
        where m is the number of states and n is the length of the series. ")
    m <- dim(Z)[1]
    dim(Z) <- c(m, (n - 1) * (max(dim(Z)[2], 0, na.rm = TRUE) > 1) + 1)
  }
  if (length(T) == 1 && m == 1) {
    dim(T) <- c(1, 1, 1)
  } else {
    if ((length(T) == 1) || any(dim(T)[1:2] != m) || !dim(T)[3] %in% c(1, NA, n))
      stop("Argument T must be a (m x m) matrix, (m x m x 1) or (m x m x n) array, where m is the number of states. ")
    dim(T) <- c(m, m, (n - 1) * (max(dim(T)[3], 0, na.rm = TRUE) > 1) + 1)
  }
  
  if (length(R) == m) {
    dim(R) <- c(m, 1, 1)
    k <- 1
  } else {
    if (!(dim(R)[1] == m) || dim(R)[2] > m || !dim(R)[3] %in% c(1, NA, n))
      stop("Argument R must be a (m x k) matrix, (m x k x 1) or (m x k x n) array, where k<=m is the number of disturbances eta, and m is the number of states. ")
    k <- dim(R)[2]
    dim(R) <- c(m, k, (n - 1) * (max(dim(R)[3], 0, na.rm = TRUE) > 1) + 1)
  }
  
  if (missing(a1)) {
    a1 <- rep(0, m)
  } else {
    if (length(a1) <= m) {
      a1 <- rep(a1, length.out = m)
    } else stop("Misspecified a1, argument a1 must be a vector of length m, where m is the number of state_names and 1<=t<=m.")
  }
  if (missing(P1)) {
    P1 <- matrix(0, m, m)
  } else {
    if (length(P1) == 1 && m == 1) {
      dim(P1) <- c(1, 1)
    } else {
      if (any(dim(P1)[1:2] != m))
        stop("Argument P1 must be (m x m) matrix, where m is the number of states. ")
    }
  }
  if (length(H)[3] %in% c(1, n))
    stop("Argument H must be a scalar or a vector of length n, where n is the length of the time series y.")
  
  if (missing(state_names)) {
    state_names <- paste("State", 1:m)
  }
  rownames(Z) <- colnames(T) <- rownames(T) <- rownames(R) <- names(a1) <-
    rownames(P1) <- colnames(P1) <- state_names
  
  H_ind <- which(is.na(H)) - 1L
  H_n <- length(H_ind)
  Z_ind <- which(is.na(Z)) - 1L
  Z_n <- length(Z_ind)
  T_ind <- which(is.na(T)) - 1L
  T_n <- length(T_ind)
  R_ind <- which(is.na(R)) - 1L
  R_n <- length(R_ind)
  
  if (H_n > 0) {
    check_prior(H_prior, "H_prior")
    if (H_n == 1) {
      H[is.na(H)] <- H_prior$init
    } else {
      H[is.na(H)] <-  sapply(H_prior, "[[", "init")
    }
  } else H_prior <- NULL
  
  if (Z_n > 0) {
    check_prior(Z_prior, "Z_prior")
    if (Z_n == 1) {
      Z[is.na(Z)] <- Z_prior$init
    } else {
      Z[is.na(Z)] <-  sapply(Z_prior, "[[", "init")
    }
  } else Z_prior <- NULL
  
  if (T_n > 0) {
    check_prior(T_prior, "T_prior")
    if (T_n == 1) {
      T[is.na(T)] <- T_prior$init
    } else {
      T[is.na(T)] <-  sapply(T_prior, "[[", "init")
    }
  } else T_prior <- NULL
  
  if (R_n > 0) {
    check_prior(R_prior, "R_prior")
    if (R_n == 1) {
      R[is.na(R)] <- R_prior$init
    } else {
      R[is.na(R)] <-  sapply(R_prior, "[[", "init")
    }
  } else R_prior <- NULL
  
  priors <- c(if(H_n > 1) Z_prior else list(H_prior),
    if(Z_n > 1) Z_prior else list(Z_prior),
    if(T_n > 1) T_prior else list(T_prior),
    if(R_n > 1) R_prior else list(R_prior),
    if(ncol(xreg) > 1) beta else list(beta))
  
  names(priors) <- c(if(H_n > 0) paste0("H_",1:H_n),
    if(Z_n > 0) paste0("Z_",1:Z_n),
    if(T_n > 0) paste0("T_",1:T_n),
    if(R_n > 0) paste0("R_",1:R_n), names(coefs))
  priors <- priors[sapply(priors, is_prior)]
  
  if (!missing(obs_intercept)) {
    check_obs_intercept(obs_intercept, 1L, n)
  } else {
    obs_intercept <- matrix(0)
  }
  if (!missing(state_intercept)) {
    check_state_intercept(state_intercept, m, n)
  } else {
    state_intercept <- matrix(0, m, 1)
  }
  theta <- if (length(priors) > 0) sapply(priors, "[[", "init") else numeric(0)
  priors <- combine_priors(priors)
  structure(list(y = as.ts(y), Z = Z, H = H, T = T, R = R, a1 = a1, P1 = P1,
    xreg = xreg, coefs = coefs, obs_intercept = obs_intercept,
    state_intercept = state_intercept, Z_ind = Z_ind,
    H_ind = H_ind, T_ind = T_ind, R_ind = R_ind,
    prior_distributions = priors$prior_distribution, prior_parameters = priors$parameters,
    theta = theta), class = "gssm")
}
#' General univariate non-Gaussian/non-linear state space models
#'
#' Construct an object of class \code{ngssm} by defining the corresponding terms
#' of the observation and state equation:
#'
#' \deqn{p(y_t | Z_t \alpha_t), (\textrm{observation equation})}
#' \deqn{\alpha_{t+1} = T_t \alpha_t + R_t \eta_t, (\textrm{transition equation})}
#'
#' where \eqn{\eta_t \sim N(0, I_k)} and
#' \eqn{\alpha_1 \sim N(a_1, P_1)} independently of each other, and \eqn{p(y_t | .)}
#' is either Poisson, binomial or negative binomial distribution.
#'
#' @param y Observations as time series (or vector) of length \eqn{n}.
#' @param Z System matrix Z of the observation equation. Either a vector of length m,
#' a m x n matrix, or object which can be coerced to such.
#' @param T System matrix T of the state equation. Either a m x m matrix or a
#' m x m x n array, or object which can be coerced to such.
#' @param R Lower triangular matrix R the state equation. Either a m x k matrix or a
#' m x k x n array, or object which can be coerced to such.
#' @param a1 Prior mean for the initial state as a vector of length m.
#' @param P1 Prior covariance matrix for the initial state as m x m matrix.
#' @param distribution distribution of the observation. Possible choices are
#' \code{"poisson"}, \code{"binomial"}, and \code{"negative binomial"}.
#' @param phi Additional parameter relating to the non-Gaussian distribution.
#' For Negative binomial distribution this is the dispersion term, and for other
#' distributions this is ignored.
#' @param u Constant parameter for non-Gaussian models. For Poisson and negative binomial distribution, this corresponds to the offset
#' term. For binomial, this is the number of trials.
#' @param xreg Matrix containing covariates.
#' @param beta Regression coefficients. Used as an initial
#' value in MCMC. Defaults to vector of zeros.
#' @param state_names Names for the states.
#' @param Z_prior,T_prior,R_prior Priors for the NA values in system matrices.
#' @param state_intercept Intercept terms for state equation, given as a
#'  m times n matrix.
#' @return Object of class \code{ngssm}.
#' @export
ngssm <- function(y, Z, T, R, a1, P1, distribution, phi, u = 1, xreg = NULL,
  beta, state_names, Z_prior, T_prior, R_prior, state_intercept) {
  
  check_y(y)
  n <- length(y)
  
  if (is.null(xreg)) {
    xreg <- matrix(0, 0, 0)
    coefs <- numeric(0)
    beta <- NULL
  } else {
    
    if (missing(beta) || is.null(beta)) {
      stop("No prior defined for beta. ")
    }
    if(!is_prior(beta) && !is_prior_list(beta)) {
      stop("Prior for beta must be of class 'bssm_prior' or 'bssm_prior_list.")
    }
    
    if (is.null(dim(xreg)) && length(xreg) == n) {
      xreg <- matrix(xreg, n, 1)
    }
    
    check_xreg(xreg, n)
    nx <- ncol(xreg)
    if (nx == 1 && is_prior_list(beta)) beta <- beta[[1]]
    if(nx > 1) {
      coefs <- sapply(beta, "[[", "init")
    } else {
      coefs <- beta$init
    }
    check_beta(coefs, nx)
    if (is.null(colnames(xreg))) {
      colnames(xreg) <- paste0("coef_",1:ncol(xreg))
    }
    names(coefs) <- colnames(xreg)
    
  }
  
  if (length(Z) == 1) {
    dim(Z) <- c(1, 1)
    m <- 1
  } else {
    if (!(dim(Z)[2] %in% c(1, NA, n)))
      stop("Argument Z must be a vector of length m, or  (m x 1) or (m x n) matrix,
        where m is the number of states and n is the length of the series. ")
    m <- dim(Z)[1]
    dim(Z) <- c(m, (n - 1) * (max(dim(Z)[2], 0, na.rm = TRUE) > 1) + 1)
  }
  if (length(T) == 1 && m == 1) {
    dim(T) <- c(1, 1, 1)
  } else {
    if ((length(T) == 1) || any(dim(T)[1:2] != m) || !dim(T)[3] %in% c(1, NA, n))
      stop("Argument T must be a (m x m) matrix, (m x m x 1) or (m x m x n) array, where m is the number of states. ")
    dim(T) <- c(m, m, (n - 1) * (max(dim(T)[3], 0, na.rm = TRUE) > 1) + 1)
  }
  
  if (length(R) == m) {
    dim(R) <- c(m, 1, 1)
    k <- 1
  } else {
    if (!(dim(R)[1] == m) || dim(R)[2] > m || !dim(R)[3] %in% c(1, NA, n))
      stop("Argument R must be a (m x k) matrix, (m x k x 1) or (m x k x n) array, where k<=m is the number of disturbances eta, and m is the number of states. ")
    k <- dim(R)[2]
    dim(R) <- c(m, k, (n - 1) * (max(dim(R)[3], 0, na.rm = TRUE) > 1) + 1)
  }
  
  if (missing(a1)) {
    a1 <- rep(0, m)
  } else {
    if (length(a1) <= m) {
      a1 <- rep(a1, length.out = m)
    } else stop("Misspecified a1, argument a1 must be a vector of length m, where m is the number of state_names and 1<=t<=m.")
  }
  if (missing(P1)) {
    P1 <- matrix(0, m, m)
  } else {
    if (length(P1) == 1 && m == 1) {
      dim(P1) <- c(1, 1)
    } else {
      if (any(dim(P1)[1:2] != m))
        stop("Argument P1 must be (m x m) matrix, where m is the number of states. ")
    }
  }
  
  
  Z_ind <- which(is.na(Z)) - 1L
  Z_n <- length(Z_ind)
  T_ind <- which(is.na(T)) - 1L
  T_n <- length(T_ind)
  R_ind <- which(is.na(R)) - 1L
  R_n <- length(R_ind)
  
  if (Z_n > 0) {
    check_prior(Z_prior, "Z_prior")
    if (Z_n == 1) {
      Z[is.na(Z)] <- Z_prior$init
    } else {
      Z[is.na(Z)] <-  sapply(Z_prior, "[[", "init")
    }
  } else Z_prior <- NULL
  
  if (T_n > 0) {
    check_prior(T_prior, "T_prior")
    if (T_n == 1) {
      T[is.na(T)] <- T_prior$init
    } else {
      T[is.na(T)] <-  sapply(T_prior, "[[", "init")
    }
  } else T_prior <- NULL
  
  if (R_n > 0) {
    check_prior(R_prior, "R_prior")
    if (R_n == 1) {
      R[is.na(R)] <- R_prior$init
    } else {
      R[is.na(R)] <-  sapply(R_prior, "[[", "init")
    }
  } else R_prior <- NULL
  
  distribution <- match.arg(distribution, c("poisson", "binomial",
    "negative binomial"))
  
  use_phi <- distribution %in% c("negative binomial", "gamma")
  phi_est <- FALSE
  if (use_phi) {
    if (is_prior(phi)) {
      check_phi(phi$init, distribution)
      phi_est <- TRUE
    } else {
      check_phi(phi, distribution)
    }
  } else {
    phi <- 1
  }
  
  use_u <- distribution %in% c("poisson", "binomial", "negative binomial")
  if (use_u) {
    check_u(u)
    if (length(u) != n) {
      u <- rep(u, length.out = n)
    }
  }
  
  initial_mode <- init_mode(y, u, distribution, if (ncol(xreg) > 0) xreg %*% coefs else NULL)
  
  if (missing(state_names)) {
    state_names <- paste("State", 1:m)
  }
  rownames(Z) <- colnames(T) <- rownames(T) <- rownames(R) <- names(a1) <-
    rownames(P1) <- colnames(P1) <- state_names
  
  
  
  priors <- c(if(Z_n > 1) Z_prior else list(Z_prior),
    if(T_n > 1) T_prior else list(T_prior),
    if(R_n > 1) R_prior else list(R_prior), list(phi),
    if(ncol(xreg) > 1) beta else list(beta))
  
  names(priors) <- c(if(Z_n > 0) paste0("Z_",1:Z_n),
    if(T_n > 0) paste0("T_",1:T_n),
    if(R_n > 0) paste0("R_",1:R_n), "phi", names(coefs))
  priors <- priors[sapply(priors, is_prior)]
  
  if (phi_est) {
    phi <- phi$init
  }
  
  obs_intercept <- matrix(0)
  
  if (!missing(state_intercept)) {
    check_state_intercept(state_intercept, m, n)
  } else {
    state_intercept <- matrix(0, m, 1)
  }
  
  theta <- if (length(priors) > 0) sapply(priors, "[[", "init") else numeric(0)
  priors <- combine_priors(priors)
  
  structure(list(y = y, Z = Z, T = T, R = R, a1 = a1, P1 = P1, phi = phi, u = u,
    xreg = xreg, coefs = coefs, obs_intercept = obs_intercept,
    state_intercept = state_intercept, distribution = distribution,
    initial_mode = initial_mode, Z_ind = Z_ind,
    T_ind = T_ind, R_ind = R_ind, phi_est = phi_est, 
    prior_distributions = priors$prior_distribution, prior_parameters = priors$parameters,
    theta = theta), class = "ngssm")
}

#' General multivariate linear Gaussian state space models
#'
#' Constructs an object of class \code{llg_ssm} by defining the corresponding terms
#' of the observation and state equation:
#'
#' \deqn{y_t = D(t,\theta) + Z(t,\theta)  \alpha_t + H(t, \theta) \epsilon_t, (\textrm{observation equation})}
#' \deqn{\alpha_{t+1} = C(t,\theta) + T(t, \theta) \alpha_t + R(t, \theta)\eta_t, (\textrm{transition equation})}
#'
#' where \eqn{\epsilon_t \sim N(0, I_p)}, \eqn{\eta_t \sim N(0, I_m)} and
#' \eqn{\alpha_1 \sim N(a_1, P_1)} independently of each other.
#'
#' Compared to other models, these general models need a bit more effort from
#' the user, as you must provide the several small C++ snippets which define the
#' model structure. See examples in the vignette.
#' @param y Observations as multivariate time series (or matrix) of length \eqn{n}.
#' @param Z,H,T,R,a1,P1,obs_intercept,state_intercept An external pointers for the C++ functions which
#' define the corresponding model functions.
#' @param theta Parameter vector passed to all model functions.
#' @param known_params Vector of known parameters passed to all model functions.
#' @param known_tv_params Matrix of known parameters passed to all model functions.
#' @param n_states Number of states in the model.
#' @param n_etas Dimension of the noise term of the transition equation.
#' @param log_prior_pdf An external pointer for the C++ function which
#' computes the log-prior density given theta.
#' @param time_varying Optional logical vector of length 6, denoting whether the values of
#' Z, H, T, R, D and C can vary with respect to time variable.
#' If used, can speed up some computations.
#' @param state_names Names for the states.
#' @return Object of class \code{llg_ssm}.
#' @export
lgg_ssm <- function(y, Z, H, T, R, a1, P1, theta,
  obs_intercept, state_intercept,
  known_params = NA, known_tv_params = matrix(NA), n_states, n_etas,
  log_prior_pdf, time_varying = rep(TRUE, 6), 
  state_names = paste0("state",1:n_states)) {
  
  if (is.null(dim(y))) {
    dim(y) <- c(length(y), 1)
  }
  
  if(missing(n_etas)) {
    n_etas <- n_states
  }
  structure(list(y = as.ts(y), Z = Z, H = H, T = T,
    R = R, a1 = a1, P1 = P1, theta = theta,
    obs_intercept = obs_intercept, state_intercept = state_intercept,
    log_prior_pdf = log_prior_pdf, known_params = known_params,
    known_tv_params = known_tv_params, time_varying = time_varying,
    n_states = n_states, n_etas = n_etas,
    state_names = state_names), class = "lgg_ssm")
}

#'
#' General multivariate nonlinear Gaussian state space models
#'
#' Constructs an object of class \code{nlg_ssm} by defining the corresponding terms
#' of the observation and state equation:
#'
#' \deqn{y_t = Z(t, \alpha_t, \theta) + H(t, \theta) \epsilon_t, (\textrm{observation equation})}
#' \deqn{\alpha_{t+1} = T(t, \alpha_t, \theta) + R(t, \theta)\eta_t, (\textrm{transition equation})}
#'
#' where \eqn{\epsilon_t \sim N(0, I_p)}, \eqn{\eta_t \sim N(0, I_m)} and
#' \eqn{\alpha_1 \sim N(a_1, P_1)} independently of each other, and functions
#' \eqn{Z, H, T, R} can depend on \eqn{\alpha_t} and parameter vector \eqn{\theta}.
#'
#' Compared to other models, these general models need a bit more effort from
#' the user, as you must provide the several small C++ snippets which define the
#' model structure. See examples in the vignette.
#' @param y Observations as multivariate time series (or matrix) of length \eqn{n}.
#' @param Z,H,T,R  An external pointers for the C++ functions which
#' define the corresponding model functions.
#' @param Z_gn,T_gn An external pointers for the C++ functions which
#' define the gradients of the corresponding model functions.
#' @param a1 Prior mean for the initial state as a vector of length m.
#' @param P1 Prior covariance matrix for the initial state as m x m matrix.
#' @param theta Parameter vector passed to all model functions.
#' @param known_params Vector of known parameters passed to all model functions.
#' @param known_tv_params Matrix of known parameters passed to all model functions.
#' @param n_states Number of states in the model.
#' @param n_etas Dimension of the noise term of the transition equation.
#' @param log_prior_pdf An external pointer for the C++ function which
#' computes the log-prior density given theta.
#' @param time_varying Optional logical vector of length 4, denoting whether the values of
#' Z, H, T, and R vary with respect to time variable (given identical states).
#' If used, this can speed up some computations.
#' @param state_names Names for the states.
#' @return Object of class \code{nlg_ssm}.
#' @export
nlg_ssm <- function(y, Z, H, T, R, Z_gn, T_gn, a1, P1, theta,
  known_params = NA, known_tv_params = matrix(NA), n_states, n_etas,
  log_prior_pdf, time_varying = rep(TRUE, 4), state_names = paste0("state",1:n_states)) {
  
  if (is.null(dim(y))) {
    dim(y) <- c(length(y), 1)
  }
 
  if(missing(n_etas)) {
    n_etas <- n_states
  }
  structure(list(y = as.ts(y), Z = Z, H = H, T = T,
    R = R, Z_gn = Z_gn, T_gn = T_gn, a1 = a1, P1 = P1, theta = theta,
    log_prior_pdf = log_prior_pdf, known_params = known_params,
    known_tv_params = known_tv_params,
    n_states = n_states, n_etas = n_etas,
    time_varying = time_varying,
    state_names = state_names), class = "nlg_ssm")
}



#'
#' Univariate state space model with continuous SDE dynamics
#'
#' Constructs an object of class \code{sde_ssm} by defining the functions for
#' the drift, diffusion and derivative of diffusion terms of univariate SDE,
#' as well as the log-density of observation equation. We assume that the
#' observations are measured at integer times (missing values are allowed).
#'
#' As in case of \code{nlg_ssm} models, these general models need a bit more effort from
#' the user, as you must provide the several small C++ snippets which define the
#' model structure. See SDE vignette for an example.
#'
#' @param y Observations as univariate time series (or vector) of length \eqn{n}.
#' @param drift,diffusion,ddiffusion An external pointers for the C++ functions which
#' define the drift, diffusion and derivative of diffusion functions of SDE.
#' @param obs_pdf An external pointer for the C++ function which
#' computes the observational log-density given the the states and parameter vector theta.
#' @param prior_pdf An external pointer for the C++ function which
#' computes the prior log-density given the parameter vector theta.
#' @param theta Parameter vector passed to all model functions.
#' @param x0 Fixed initial value for SDE at time 0.
#' @param positive If \code{TRUE}, positivity constraint is
#'   forced by \code{abs} in Millstein scheme.
#' @return Object of class \code{sde_ssm}.
#' @export
sde_ssm <- function(y, drift, diffusion, ddiffusion, obs_pdf,
  prior_pdf, theta, x0, positive) {
  
  check_y(y)
  n <- length(y)
  
  structure(list(y = as.ts(y), drift = drift,
    diffusion = diffusion,
    ddiffusion = ddiffusion, obs_pdf = obs_pdf,
    prior_pdf = prior_pdf, theta = theta, x0 = x0,
    positive = positive, state_names = "x"), class = "sde_ssm")
}


#'
#' General multivariate linear-Gaussian state space models
#'
#' Construct an object of class \code{gssm} by defining the corresponding terms
#' of the observation and state equation:
#'
#' \deqn{y_t = D_t + Z_t \alpha_t + H_t \epsilon_t, (\textrm{observation equation})}
#' \deqn{\alpha_{t+1} = C_t + T_t \alpha_t + R_t \eta_t, (\textrm{transition equation})}
#'
#' where \eqn{\epsilon_t \sim N(0, I_p)}, \eqn{\eta_t \sim N(0, I_k)} and
#' \eqn{\alpha_1 \sim N(a_1, P_1)} independently of each other.
#'
#' @param y Observations as multivariate time series (or matrix) of length \eqn{n}.
#' @param Z System matrix Z of the observation equation. Either a p x m matrix or
#' a p x m x n array, or an object which can be coerced to such.
#' @param H Covarianc matrix for observational level noise.
#' @param T System matrix T of the state equation. Either a m x m matrix or a
#' m x m x n array, or object which can be coerced to such.
#' @param R Lower triangular matrix R the state equation. Either a m x k matrix or a
#' m x k x n array, or object which can be coerced to such.
#' @param a1 Prior mean for the initial state as a vector of length m.
#' @param P1 Prior covariance matrix for the initial state as m x m matrix.
#' @param xreg An array containing p covariate matrices with dimensions n x h.
#' @param beta matrix of regression coefficients with n columns. Used as an initial
#' value in MCMC. Defaults to matrix of zeros.
#' @param state_names Names for the states.
#' @param H_prior,Z_prior,T_prior,R_prior Priors for the NA values in system matrices.
#' @param obs_intercept,state_intercept Intercept terms for observation and
#' state equations, given as a p times n and m times n matrices.
#' @return Object of class \code{mv_gssm}.
#' @export
mv_gssm <- function(y, Z, H, T, R, a1, P1, xreg = NULL, beta, state_names,
  H_prior, Z_prior, T_prior, R_prior, obs_intercept, state_intercept) {
  
  #check_y(y)
  n <- nrow(y)
  p <- ncol(y)
 
  if (is.null(xreg)) {
    xreg <- array(0, c(0, 0, 0))
    coefs <- matrix(0, 1, p)
    beta <- NULL
  } else {
    if (missing(beta) || is.null(beta)) {
      stop("No prior defined for beta. ")
    }
    if(!is_prior(beta) && !is_prior_list(beta)) {
      stop("Prior for beta must be of class 'bssm_prior' or 'bssm_prior_list.")
    }
    
    if (is.null(dim(xreg)) && length(xreg) == n) {
      xreg <- matrix(xreg, n, 1)
    }
    
    check_xreg(xreg, n)
    if((nx <- ncol(xreg)) > 1) {
      coefs <- sapply(beta, "[[", "init")
    } else {
      coefs <- beta$init
    }
    check_beta(coefs, nx)
    if (is.null(colnames(xreg))) {
      colnames(xreg) <- paste0("coef_",1:ncol(xreg))
    }
    names(coefs) <- colnames(xreg)
    
  }
  if (dim(Z)[1] != p || !(dim(Z)[3] %in% c(1, NA, n)))
    stop("Argument Z must be a (p x m) matrix or (p x m x n) array
      where p is the number of series, m is the number of states, and n is the length of the series. ")
  m <- dim(Z)[2]
  dim(Z) <- c(p, m, (n - 1) * (max(dim(Z)[3], 0, na.rm = TRUE) > 1) + 1)
  
  if (length(T) == 1 && m == 1) {
    dim(T) <- c(1, 1, 1)
  } else {
    if ((length(T) == 1) || any(dim(T)[1:2] != m) || !(dim(T)[3] %in% c(1, NA, n)))
      stop("Argument T must be a (m x m) matrix, (m x m x 1) or (m x m x n) array, where m is the number of states. ")
    dim(T) <- c(m, m, (n - 1) * (max(dim(T)[3], 0, na.rm = TRUE) > 1) + 1)
  }
  
  if (length(R) == m) {
    dim(R) <- c(m, 1, 1)
    k <- 1
  } else {
    if (!(dim(R)[1] == m) || dim(R)[2] > m || !dim(R)[3] %in% c(1, NA, n))
      stop("Argument R must be a (m x k) matrix, (m x k x 1) or (m x k x n) array, where k<=m is the number of disturbances eta, and m is the number of states. ")
    k <- dim(R)[2]
    dim(R) <- c(m, k, (n - 1) * (max(dim(R)[3], 0, na.rm = TRUE) > 1) + 1)
  }
  
  if (missing(a1)) {
    a1 <- rep(0, m)
  } else {
    if (length(a1) <= m) {
      a1 <- rep(a1, length.out = m)
    } else stop("Misspecified a1, argument a1 must be a vector of length m, where m is the number of state_names and 1<=t<=m.")
  }
  if (missing(P1)) {
    P1 <- matrix(0, m, m)
  } else {
    if (length(P1) == 1 && m == 1) {
      dim(P1) <- c(1, 1)
    } else {
      if (any(dim(P1)[1:2] != m))
        stop("Argument P1 must be (m x m) matrix, where m is the number of states. ")
    }
  }
  if (any(dim(H)[1:2] != p) || !(dim(H)[3] %in% c(1, n, NA)))
    stop("Argument H must be a p x p matrix or a p x p x n array.")
  dim(H) <- c(p, p, (n - 1) * (max(dim(H)[3], 0, na.rm = TRUE) > 1) + 1)
  
  if (missing(state_names)) {
    state_names <- paste("State", 1:m)
  }
  colnames(Z) <- colnames(T) <- rownames(T) <- rownames(R) <- names(a1) <-
    rownames(P1) <- colnames(P1) <- state_names
  
  H_ind <- which(is.na(H)) - 1L
  H_n <- length(H_ind)
  Z_ind <- which(is.na(Z)) - 1L
  Z_n <- length(Z_ind)
  T_ind <- which(is.na(T)) - 1L
  T_n <- length(T_ind)
  R_ind <- which(is.na(R)) - 1L
  R_n <- length(R_ind)
  
  if (H_n > 0) {
    check_prior(H_prior, "H_prior")
    if (H_n == 1) {
      H[is.na(H)] <- H_prior$init
    } else {
      H[is.na(H)] <-  sapply(H_prior, "[[", "init")
    }
  } else H_prior <- NULL
  
  if (Z_n > 0) {
    check_prior(Z_prior, "Z_prior")
    if (Z_n == 1) {
      Z[is.na(Z)] <- Z_prior$init
    } else {
      Z[is.na(Z)] <-  sapply(Z_prior, "[[", "init")
    }
  } else Z_prior <- NULL
  
  if (T_n > 0) {
    check_prior(T_prior, "T_prior")
    if (T_n == 1) {
      T[is.na(T)] <- T_prior$init
    } else {
      T[is.na(T)] <-  sapply(T_prior, "[[", "init")
    }
  } else T_prior <- NULL
  
  if (R_n > 0) {
    check_prior(R_prior, "R_prior")
    if (R_n == 1) {
      R[is.na(R)] <- R_prior$init
    } else {
      R[is.na(R)] <-  sapply(R_prior, "[[", "init")
    }
  } else R_prior <- NULL
  
  priors <- c(if(H_n > 1) Z_prior else list(H_prior),
    if(Z_n > 1) Z_prior else list(Z_prior),
    if(T_n > 1) T_prior else list(T_prior),
    if(R_n > 1) R_prior else list(R_prior),
    if(ncol(xreg) > 1) beta else list(beta))
  
  names(priors) <- c(if(H_n > 0) paste0("H_",1:H_n),
    if(Z_n > 0) paste0("Z_",1:Z_n),
    if(T_n > 0) paste0("T_",1:T_n),
    if(R_n > 0) paste0("R_",1:R_n), names(coefs))
  priors <- priors[sapply(priors, is_prior)]
  
  if (!missing(obs_intercept)) {
    check_obs_intercept(obs_intercept, p, n)
  } else {
    obs_intercept <- matrix(0, p, 1)
  }
  if (!missing(state_intercept)) {
    check_state_intercept(state_intercept, m, n)
  } else {
    state_intercept <- matrix(0, m, 1)
  }
  
  theta <- if (length(priors) > 0) sapply(priors, "[[", "init") else numeric(0)
  priors <- combine_priors(priors)
  
  structure(list(y = as.ts(y), Z = Z, H = H, T = T, R = R, a1 = a1, P1 = P1,
    xreg = xreg, coefs = coefs, obs_intercept = obs_intercept,
    state_intercept = state_intercept, Z_ind = Z_ind,
    H_ind = H_ind, T_ind = T_ind, R_ind = R_ind, 
    prior_distributions = priors$prior_distribution, prior_parameters = priors$parameters,
    theta = theta), class = "mv_gssm")
}
