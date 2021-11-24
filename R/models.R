#' @srrstats {G2.3, G2.3a, G2.3b} match.arg and tolower used where applicable.
#' @srrstats {G2.7, G2.8, G2.9} Only matrix/mts/arrays as tabular data are 
#' supported, not data.frame or similar objects.
#' @srrstats {G2.14, G2.14a, G2.14b, G2.14c, BS3.0} Missing observations are 
#' handled automatically as per SSM theory, whereas missing values are not 
#' allowed elsewhere.
#' @srrstats {BS1.0, BS1.1, BS1.2, BS1.2c} Examples and definitions of priors.
NULL

## placeholder functions for fixed models
default_prior_fn <- function(theta) {
  0
}
default_update_fn <- function(theta) {
  
}
#'
#' General univariate linear-Gaussian state space models
#'
#' Construct an object of class \code{ssm_ulg} by directly defining the 
#' corresponding terms of the model.
#' 
#' The general univariate linear-Gaussian model is defined using the following 
#' observational and state equations:
#'
#' \deqn{y_t = D_t + Z_t \alpha_t + H_t \epsilon_t, 
#' (\textrm{observation equation})}
#' \deqn{\alpha_{t+1} = C_t + T_t \alpha_t + R_t \eta_t, 
#' (\textrm{transition equation})}
#'
#' where \eqn{\epsilon_t \sim N(0, 1)}, \eqn{\eta_t \sim N(0, I_k)} and
#' \eqn{\alpha_1 \sim N(a_1, P_1)} independently of each other.
#' Here k is the number of disturbance terms which can be less than m, the 
#' number of states.
#'
#' The \code{update_fn} function should take only one 
#' vector argument which is used to create list with elements named as
#' \code{Z}, \code{H} \code{T}, \code{R}, \code{a1}, \code{P1}, \code{D}, 
#' and \code{C},
#' where each element matches the dimensions of the original model.
#' If any of these components is missing, it is assumed to be constant wrt. 
#' theta.
#' Note that while you can input say R as m x k matrix for \code{ssm_ulg}, 
#' \code{update_fn} should return R as m x k x 1 in this case. 
#' It might be useful to first construct the model without updating function 
#' and then check the expected structure of the model components from the 
#' output.
#' 
#' @inheritParams ssm_ung
#' @param H A vector of standard deviations. Either a scalar or a vector of 
#' length n. 
#' @param update_fn A function which returns list of updated model 
#' components given input vector theta. This function should take only one 
#' vector argument which is used to create list with elements named as
#' \code{Z}, \code{H}, \code{T}, \code{R}, \code{a1}, \code{P1}, \code{D}, and
#' \code{C}, where each element matches the dimensions of the original model 
#' It's best to check the internal dimensions with \code{str(model_object)} as 
#' the dimensions of input arguments can differ from the final dimensions.
#' If any of these components is missing, it is assumed to be constant wrt. 
#' theta.
#' @return An object of class \code{ssm_ulg}.
#' @export
#' @examples 
#' 
#' # Regression model with time-varying coefficients
#' set.seed(1)
#' n <- 100
#' x1 <- rnorm(n)
#' x2 <- rnorm(n)
#' b1 <- 1 + cumsum(rnorm(n, sd = 0.5))
#' b2 <- 2 + cumsum(rnorm(n, sd = 0.1))
#' y <- 1 + b1 * x1 + b2 * x2 + rnorm(n, sd = 0.1)
#' 
#' Z <- rbind(1, x1, x2)
#' H <- 0.1
#' T <- diag(3)
#' R <- diag(c(0, 1, 0.1))
#' a1 <- rep(0, 3)
#' P1 <- diag(10, 3)
#' 
#' # updates the model given the current values of the parameters
#' update_fn <- function(theta) {
#'   R <- diag(c(0, theta[1], theta[2]))
#'   dim(R) <- c(3, 3, 1)
#'   list(R = R, H = theta[3])
#' }
#' # prior for standard deviations as half-normal(1)
#' prior_fn <- function(theta) {
#'   if(any(theta < 0)) {
#'     log_p <- -Inf 
#'   } else {
#'     log_p <- sum(dnorm(theta, 0, 1, log = TRUE))
#'   }
#'   log_p
#' }
#' 
#' model <- ssm_ulg(y, Z, H, T, R, a1, P1, 
#'   init_theta = c(1, 0.1, 0.1), 
#'   update_fn = update_fn, prior_fn = prior_fn, 
#'   state_names = c("level", "b1", "b2"),
#'   # using default values, but being explicit for testing purposes
#'   C = matrix(0, 3, 1), D = numeric(1))
#' 
#' out <- run_mcmc(model, iter = 5000)
#' out
#' sumr <- summary(out, variable = "state", times = 1:n)
#' sumr$true <- c(b1, b2, rep(1, n))
#' library(ggplot2)
#' ggplot(sumr, aes(x = time, y = Mean)) +
#' geom_ribbon(aes(ymin = `2.5%`, ymax = `97.5%`), alpha = 0.5) +
#' geom_line() + 
#' geom_line(aes(y = true), colour = "red") + 
#' facet_wrap(~ variable, scales = "free") +
#' theme_bw()
#' 
#' # Perhaps easiest way to construct a general SSM for bssm is to use the 
#' # model building functionality of KFAS:
#' library("KFAS")
#' 
#' model_kfas <- SSModel(log(drivers) ~ SSMtrend(1, Q = 5e-4)+
#'   SSMseasonal(period = 12, sea.type = "trigonometric", Q = 0) +
#'  log(PetrolPrice) + law, data = Seatbelts, H = 0.005)
#' 
#' # use as_bssm function for conversion, kappa defines the 
#' # prior variance for diffuse states
#' model_bssm <- as_bssm(model_kfas, kappa = 100)
#' 
#' # define updating function for parameter estimation
#' # we can use SSModel and as_bssm functions here as well
#' # (for large model it is more efficient to do this 
#' # "manually" by constructing only necessary matrices,
#' # i.e., in this case  a list with H and Q)
#'
#' prior_fn <- function(theta) {
#'   if(any(theta < 0)) -Inf else sum(dnorm(theta, 0, 0.1, log = TRUE))
#' }
#'  
#' update_fn <- function(theta) {
#'   
#'   model_kfas <- SSModel(log(drivers) ~ SSMtrend(1, Q = theta[1]^2)+
#'     SSMseasonal(period = 12, 
#'       sea.type = "trigonometric", Q = theta[2]^2) +
#'     log(PetrolPrice) + law, data = Seatbelts, H = theta[3]^2)
#'   
#'   # the bssm_model object is essentially list so this is fine
#'   as_bssm(model_kfas, kappa = 100, init_theta = init_theta,
#'     update_fn = update_fn, prior_fn = prior_fn) 
#' }
#' 
#' init_theta <- rep(1e-2, 3)
#' names(init_theta) <- c("sd_level", "sd_seasonal", "sd_y")
#' 
#' model_bssm <- update_fn(init_theta)
#' 
#' \donttest{
#' out <- run_mcmc(model_bssm, iter = 10000, burnin = 5000) 
#' out
#' }
#' # Above the regression coefficients are modelled as 
#' # time-invariant latent states. 
#' # Here is an alternative way where we use variable D so that the
#' # coefficients are part of parameter vector theta. Note however that the 
#' # first option often preferable in order to keep the dimension of theta low.
#' 
#' updatefn2 <- function(theta) {
#'   # note no PetrolPrice or law variables here
#'   model_kfas2 <- SSModel(log(drivers) ~ SSMtrend(1, Q = theta[1]^2)+
#'     SSMseasonal(period = 12, sea.type = "trigonometric", Q = theta[2]^2), 
#'     data = Seatbelts, H = theta[3]^2)
#'   
#'   X <- model.matrix(~ -1 + law + log(PetrolPrice), data = Seatbelts)
#'   D <- t(X %*% theta[4:5])
#'   as_bssm(model_kfas2, D = D, kappa = 100)
#' }
#' prior2 <- function(theta) {
#'  if(any(theta[1:3] < 0)) {
#'   -Inf
#'  } else {
#'    sum(dnorm(theta[1:3], 0, 0.1, log = TRUE)) +
#'    sum(dnorm(theta[4:5], 0, 10, log = TRUE))
#'  }
#' }
#' init_theta <- c(rep(1e-2, 3), 0, 0)
#' names(init_theta) <- c("sd_level", "sd_seasonal", "sd_y", "law", "Petrol")
#' model_bssm2 <- updatefn2(init_theta)
#' model_bssm2$theta <- init_theta
#' model_bssm2$prior_fn <- prior2
#' model_bssm2$update_fn <- updatefn2
#' \donttest{
#' out2 <- run_mcmc(model_bssm2, iter = 10000, burnin = 5000) 
#' out2
#' }
ssm_ulg <- function(y, Z, H, T, R, a1 = NULL, P1 = NULL, 
  init_theta = numeric(0),
  D = NULL, C = NULL, state_names, update_fn = default_update_fn, 
  prior_fn = default_prior_fn) {
  
  y <- check_y(y)
  n <- length(y)
  
  # create Z
  Z <- check_Z(Z, 1L, n)
  m <- dim(Z)[1]
  
  # create T
  T <- check_T(T, m, n)
  
  # create R
  R <- check_R(R, m, n)
  
  a1 <- check_a1(a1, m)
  
  P1 <- check_P1(P1, m)
  
  D <- check_D(D, 1L, n)
  C <- check_C(C, m, n)
  
  H <- check_H(H, 1L, n)
  
  if (missing(state_names)) {
    state_names <- paste("state", 1:m)
  }
  rownames(Z) <- colnames(T) <- rownames(T) <- rownames(R) <- names(a1) <-
    rownames(P1) <- colnames(P1) <- state_names
  
  if (is.null(names(init_theta)) && length(init_theta) > 0) 
    names(init_theta) <- paste0("theta_", seq_along(init_theta))
  
  
  # xreg and beta are need in C++ side in order to combine constructors 
  structure(list(y = as.ts(y), Z = Z, H = H, T = T, R = R, a1 = a1, P1 = P1,
    D = D, C = C, update_fn = update_fn,
    prior_fn = prior_fn, theta = init_theta,
    xreg = matrix(0, 0, 0), beta = numeric(0)), 
    class = c("ssm_ulg", "lineargaussian", "bssm_model"))
}
#' General univariate non-Gaussian state space model
#'
#' Construct an object of class \code{ssm_ung} by directly defining the 
#' corresponding terms of the model.
#' 
#' The general univariate non-Gaussian model is defined using the following 
#' observational and state equations:
#'
#' \deqn{p(y_t | D_t + Z_t \alpha_t), (\textrm{observation equation})}
#' \deqn{\alpha_{t+1} = C_t + T_t \alpha_t + R_t \eta_t, 
#' (\textrm{transition equation})}
#'
#' where \eqn{\eta_t \sim N(0, I_k)} and
#' \eqn{\alpha_1 \sim N(a_1, P_1)} independently of each other,
#' and \eqn{p(y_t | .)} is either Poisson, binomial, gamma, or 
#' negative binomial distribution.
#' Here k is the number of disturbance terms which can be less than m, 
#' the number of states.
#'
#' The \code{update_fn} function should take only one 
#' vector argument which is used to create list with elements named as
#' \code{Z}, \code{phi} \code{T}, \code{R}, \code{a1}, \code{P1}, \code{D},
#'  and \code{C},
#' where each element matches the dimensions of the original model.
#' If any of these components is missing, it is assumed to be constant 
#' wrt. theta.
#' Note that while you can input say R as m x k matrix for \code{ssm_ung}, 
#' \code{update_fn} should return R as m x k x 1 in this case. 
#' It might be useful to first construct the model without updating function 
#' and then check the expected structure of the model components from 
#' the output.
#' 
#' @inheritParams bsm_ng
#' @param y Observations as time series (or vector) of length \eqn{n}.
#' @param Z System matrix Z of the observation equation. Either a 
#' vector of length m,
#' a m x n matrix, or object which can be coerced to such.
#' @param T System matrix T of the state equation. Either a m x m matrix or a
#' m x m x n array, or object which can be coerced to such.
#' @param R Lower triangular matrix R the state equation. Either 
#' a m x k matrix or a m x k x n array, or object which can be coerced to such.
#' @param a1 Prior mean for the initial state as a vector of length m.
#' @param P1 Prior covariance matrix for the initial state as m x m matrix.
#' @param state_names A character vector defining the names of the states.
#' @param C Intercept terms \eqn{C_t} for the state equation, given as a
#'  m times 1 or m times n matrix.
#' @param D Intercept terms \eqn{D_t} for the observations equation, given as a
#' scalar or vector of length n.
#' @param init_theta Initial values for the unknown hyperparameters theta 
#' (i.e. unknown variables excluding latent state variables).
#' @param update_fn A function which returns list of updated model 
#' components given input vector theta. This function should take only one 
#' vector argument which is used to create list with elements named as
#' \code{Z}, \code{T}, \code{R}, \code{a1}, \code{P1}, \code{D}, \code{C}, and
#' \code{phi}, where each element matches the dimensions of the original model.
#' If any of these components is missing, it is assumed to be constant wrt. 
#' theta. It's best to check the internal dimensions with 
#' \code{str(model_object)} as the dimensions of input arguments can differ 
#' from the final dimensions.
#' @param prior_fn A function which returns log of prior density 
#' given input vector theta.
#' @return An object of class \code{ssm_ung}.
#' @export
#' @examples 
#' 
#' data("drownings", package = "bssm")
#' model <- ssm_ung(drownings[, "deaths"], Z = 1, T = 1, R = 0.2, 
#'   a1 = 0, P1 = 10, distribution = "poisson", u = drownings[, "population"])
#' 
#' # approximate results based on Gaussian approximation
#' out <- smoother(model)
#' ts.plot(cbind(model$y / model$u, exp(out$alphahat)), col = 1:2)
ssm_ung <- function(y, Z, T, R, a1 = NULL, P1 = NULL, distribution, phi = 1, 
  u, init_theta = numeric(0), D = NULL, C = NULL, state_names, 
  update_fn = default_update_fn,
  prior_fn = default_prior_fn) {
  
  
  distribution <- match.arg(tolower(distribution), 
    c("poisson", "binomial", "negative binomial", "gamma"))
  
  y <- check_y(y, distribution = distribution)
  n <- length(y)
  if (missing(u)) u <- rep(1, n)
  u <- check_u(u, y)
  Z <- check_Z(Z, 1L, n)
  m <- dim(Z)[1]
  
  T <- check_T(T, m, n)
  
  # create R
  R <- check_R(R, m, n)
  
  a1 <- check_a1(a1, m)
  
  P1 <- check_P1(P1, m)
  
  D <- check_D(D, 1L, n)
  C <- check_C(C, m, n)
  
  check_phi(phi)
  
  initial_mode <- matrix(init_mode(y, u, distribution), ncol = 1)
  
  if (missing(state_names)) {
    state_names <- paste("state", 1:m)
  }
  rownames(Z) <- colnames(T) <- rownames(T) <- rownames(R) <- names(a1) <-
    rownames(P1) <- colnames(P1) <- state_names
  
  if(is.null(names(init_theta)) && length(init_theta) > 0) 
    names(init_theta) <- paste0("theta_", seq_along(init_theta))
  
  # xreg and beta are need in C++ side in order to combine constructors 
  structure(list(y = as.ts(y), Z = Z, T = T, R = R, a1 = a1, P1 = P1, 
    phi = phi, u = u, D = D, C = C, distribution = distribution,
    initial_mode = initial_mode, update_fn = update_fn,
    prior_fn = prior_fn, theta = init_theta,
    max_iter = 100, conv_tol = 1e-8, local_approx = TRUE,
    xreg = matrix(0, 0, 0), beta = numeric(0)),
    class = c("ssm_ung", "nongaussian", "bssm_model"))
}

#' General multivariate linear Gaussian state space models
#' 
#' Construct an object of class \code{ssm_mlg} by directly defining the 
#' corresponding terms of the model.
#' 
#' The general multivariate linear-Gaussian model is defined using the 
#' following observational and state equations:
#'
#' \deqn{y_t = D_t + Z_t \alpha_t + H_t \epsilon_t, 
#' (\textrm{observation equation})}
#' \deqn{\alpha_{t+1} = C_t + T_t \alpha_t + R_t \eta_t, 
#' (\textrm{transition equation})}
#'
#' where \eqn{\epsilon_t \sim N(0, I_p)}, \eqn{\eta_t \sim N(0, I_k)} and
#' \eqn{\alpha_1 \sim N(a_1, P_1)} independently of each other. 
#' Here p is the number of time series and k is the number of disturbance terms 
#' (which can be less than m, the number of states).
#'
#' The \code{update_fn} function should take only one 
#' vector argument which is used to create list with elements named as
#' \code{Z}, \code{H} \code{T}, \code{R}, \code{a1}, \code{P1}, \code{D}, 
#' and \code{C},
#' where each element matches the dimensions of the original model.
#' If any of these components is missing, it is assumed to be 
#' constant wrt. theta.
#' Note that while you can input say R as m x k matrix for \code{ssm_mlg}, 
#' \code{update_fn} should return R as m x k x 1 in this case. 
#' It might be useful to first construct the model without updating function
#' 
#' @inheritParams ssm_ulg
#' @param y Observations as multivariate time series or matrix with 
#' dimensions n x p.
#' @param Z System matrix Z of the observation equation as p x m matrix or 
#' p x m x n array.
#' @param H Lower triangular matrix H of the observation. Either a scalar or 
#' a vector of length n.
#' @param T System matrix T of the state equation. Either a m x m matrix or a
#' m x m x n array.
#' @param R Lower triangular matrix R the state equation. Either a m x k matrix 
#' or a m x k x n array.
#' @param D Intercept terms for observation equation, given as a p x n matrix.
#' @param C Intercept terms for state equation, given as m x n matrix.
#' @return An object of class \code{ssm_mlg}.
#' @export
#' @examples
#' 
#' data("GlobalTemp", package = "KFAS")
#' model_temp <- ssm_mlg(GlobalTemp, H = matrix(c(0.15,0.05,0, 0.05), 2, 2), 
#'   R = 0.05, Z = matrix(1, 2, 1), T = 1, P1 = 10,
#'   state_names = "temperature",
#'   # using default values, but being explicit for testing purposes
#'   D = matrix(0, 2, 1), C = matrix(0, 1, 1))
#' ts.plot(cbind(model_temp$y, smoother(model_temp)$alphahat), col = 1:3)
#'
ssm_mlg <- function(y, Z, H, T, R, a1 = NULL, P1 = NULL, 
  init_theta = numeric(0), D = NULL, C = NULL, state_names, 
  update_fn = default_update_fn, prior_fn = default_prior_fn) {
  
  # create y
  y <- check_y(y, multivariate = TRUE)
  n <- nrow(y)
  p <- ncol(y)

  # create Z
  Z <- check_Z(Z, p, n, multivariate = TRUE)
  m <- dim(Z)[2]
  
  T <- check_T(T, m, n)
  
  # create R
  R <- check_R(R, m, n)
  
  a1 <- check_a1(a1, m)
  P1 <- check_P1(P1, m)
  
  D <- check_D(D, p, n)
  D <- as.matrix(D) # p = 1
  C <- check_C(C, m, n)
  
  H <- check_H(H, p, n, multivariate = TRUE)
  
  if (missing(state_names)) {
    state_names <- paste("state", 1:m)
  }
  colnames(Z) <- colnames(T) <- rownames(T) <- rownames(R) <- names(a1) <-
    rownames(P1) <- colnames(P1) <- state_names
  
  if(is.null(names(init_theta)) && length(init_theta) > 0)
    names(init_theta) <- paste0("theta_", seq_along(init_theta))
  
  structure(list(y = as.ts(y), Z = Z, H = H, T = T, R = R, a1 = a1, 
    P1 = P1, D = D, C = C, update_fn = update_fn,
    prior_fn = prior_fn, theta = init_theta, 
    state_names = state_names), class = c("ssm_mlg", "lineargaussian", 
      "bssm_model"))
}

#' General Non-Gaussian State Space Model
#' 
#' Construct an object of class \code{ssm_mng} by directly defining the 
#' corresponding terms of the model.
#' 
#' The general multivariate non-Gaussian model is defined using the following 
#' observational and state equations:
#'
#' \deqn{p^i(y^i_t | D_t + Z_t \alpha_t), (\textrm{observation equation})}
#' \deqn{\alpha_{t+1} = C_t + T_t \alpha_t + R_t \eta_t, 
#' (\textrm{transition equation})}
#'
#' where \eqn{\eta_t \sim N(0, I_k)} and
#' \eqn{\alpha_1 \sim N(a_1, P_1)} independently of each other, and 
#' \eqn{p^i(y_t | .)} is either Poisson, binomial, gamma, Gaussian, or 
#' negative binomial distribution for each observation series \eqn{i=1,...,p}. 
#' Here k is the number of disturbance terms (which can be less than m, 
#' the number of states).
#' @inheritParams ssm_ung
#' @param y Observations as multivariate time series or matrix with dimensions 
#' n x p.
#' @param Z System matrix Z of the observation equation as p x m matrix or 
#' p x m x n array.
#' @param T System matrix T of the state equation. Either a m x m matrix or a
#' m x m x n array.
#' @param R Lower triangular matrix R the state equation. Either a m x k 
#' matrix or a
#' m x k x n array.
#' @param distribution A vector of distributions of the observed series. 
#' Possible choices are
#' \code{"poisson"}, \code{"binomial"}, \code{"negative binomial"}, 
#' \code{"gamma"}, and \code{"gaussian"}.
#' @param phi Additional parameters relating to the non-Gaussian distributions.
#' For negative binomial distribution this is the dispersion term, for 
#' gamma distribution this is the shape parameter, for Gaussian this is 
#' standard deviation, and for other distributions this is ignored.
#' @param u A matrix of positive constants for non-Gaussian models 
#' (of same dimensions as y). For Poisson,  gamma, and negative binomial 
#' distribution, this corresponds to the offset term. For binomial, this is the 
#' number of trials (and as such should be integer(ish)).
#' @param D Intercept terms for observation equation, given as p x n matrix.
#' @param C Intercept terms for state equation, given as m x n matrix.
#' @return An object of class \code{ssm_mng}.
#' @export
#' @examples
#'  
#' set.seed(1)
#' n <- 20
#' x <- cumsum(rnorm(n, sd = 0.5))
#' phi <- 2
#' y <- cbind(
#'   rgamma(n, shape = phi, scale = exp(x) / phi),
#'   rbinom(n, 10, plogis(x)))
#' 
#' Z <- matrix(1, 2, 1)
#' T <- 1
#' R <- 0.5
#' a1 <- 0
#' P1 <- 1
#' 
#' update_fn <- function(theta) {
#'   list(R = array(theta[1], c(1, 1, 1)), phi = c(theta[2], 1))
#' }
#' 
#' prior_fn <- function(theta) {
#'   ifelse(all(theta > 0), sum(dnorm(theta, 0, 1, log = TRUE)), -Inf)
#' }
#' 
#' model <- ssm_mng(y, Z, T, R, a1, P1, phi = c(2, 1), 
#'   init_theta = c(0.5, 2), 
#'   distribution = c("gamma", "binomial"),
#'   u = cbind(1, rep(10, n)),
#'   update_fn = update_fn, prior_fn = prior_fn,
#'   state_names = "random_walk",
#'   # using default values, but being explicit for testing purposes
#'   D = matrix(0, 2, 1), C = matrix(0, 1, 1))
#'
#' # smoothing based on approximating gaussian model
#' ts.plot(cbind(y, fast_smoother(model)), 
#'   col = 1:3, lty = c(1, 1, 2))
#' 
ssm_mng <- function(y, Z, T, R, a1 = NULL, P1 = NULL, distribution, 
  phi = 1, u, init_theta = numeric(0), D = NULL, C = NULL, state_names, 
  update_fn = default_update_fn, prior_fn = default_prior_fn) {
  
  # create y
  
  y <- check_y(y, multivariate = TRUE)
  n <- nrow(y)
  p <- ncol(y)
  
  if (missing(u)) u <- matrix(1, n, p)
  u <- check_u(u, y, multivariate = TRUE)
  
  if(length(distribution) == 1) distribution <- rep(distribution, p)
  check_distribution(y, distribution)
  if(length(phi) == 1) phi <- rep(phi, p)
  for(i in 1:p) {
    distribution[i] <- match.arg(tolower(distribution[i]), 
      c("poisson", "binomial", "negative binomial", "gamma", "gaussian"))
    check_phi(phi[i])
  }
  
  # create Z
  Z <- check_Z(Z, p, n, multivariate = TRUE)
  m <- dim(Z)[2]
  
  T <- check_T(T, m, n)
  
  # create R
  R <- check_R(R, m, n)
  
  a1 <- check_a1(a1, m)
  P1 <- check_P1(P1, m)
  
  D <- check_D(D, p, n)
  if (p == 1) D <- as.matrix(D)
  C <- check_C(C, m, n)
  
  initial_mode <- y
  for(i in 1:p) {
    initial_mode[, i] <- init_mode(y[, i], u[, i], distribution[i])
  }
  
  if (missing(state_names)) {
    state_names <- paste("state", 1:m)
  }
  colnames(Z) <- colnames(T) <- rownames(T) <- rownames(R) <- names(a1) <-
    rownames(P1) <- colnames(P1) <- state_names
  
  if(is.null(names(init_theta)) && length(init_theta) > 0)
    names(init_theta) <- paste0("theta_", seq_along(init_theta))
  
  
  structure(list(y = as.ts(y), Z = Z, T = T, R = R, a1 = a1, P1 = P1, 
    phi = phi, u = u, D = D, C = C, distribution = distribution,
    initial_mode = initial_mode, update_fn = update_fn,
    prior_fn = prior_fn, theta = init_theta,
    max_iter = 100, conv_tol = 1e-8, local_approx = TRUE), 
    class = c("ssm_mng", "nongaussian", "bssm_model"))
}
#' Basic Structural (Time Series) Model
#'
#' Constructs a basic structural model with local level or local trend 
#' component and seasonal component.
#' 
#' @inheritParams bsm_ng
#' @param sd_y Standard deviation of the noise of observation equation.
#' Should be an object of class \code{bssm_prior} or scalar 
#' value defining a known value such as 0.  
#' @param D Intercept terms for observation equation, given as a length n 
#' numeric vector or a scalar in case of time-invariant intercept.
#' @param C Intercept terms for state equation, given as a m times n matrix 
#' or m times 1 matrix in case of time-invariant intercept.
#' @return An object of class \code{bsm_lg}.
#' @export
#' @examples
#'
#' set.seed(1)
#' n <- 50
#' x <- rnorm(n)
#' level <- numeric(n)
#' level[1] <- rnorm(1)
#' for (i in 2:n) level[i] <- rnorm(1, -0.2 + level[i-1], sd = 0.1)
#' y <- rnorm(n, 2.1 + x + level)
#' model <- bsm_lg(y, sd_y = halfnormal(1, 5), sd_level = 0.1, a1 = level[1], 
#'   P1 = matrix(0, 1, 1), xreg = x, beta = normal(1, 0, 1),
#'   D = 2.1, C = matrix(-0.2, 1, 1))
#'   
#' ts.plot(cbind(fast_smoother(model), level), col = 1:2)
#' 
#' prior <- uniform(0.1 * sd(log10(UKgas)), 0, 1)
#' # period here is redundant as frequency(UKgas) = 4
#' model_UKgas <- bsm_lg(log10(UKgas), sd_y = prior, sd_level =  prior,
#'   sd_slope =  prior, sd_seasonal =  prior, period = 4)
#'
#' # Note small number of iterations for CRAN checks
#' mcmc_out <- run_mcmc(model_UKgas, iter = 5000)
#' summary(mcmc_out, return_se = TRUE)
#' # Use the summary method from coda:
#' summary(expand_sample(mcmc_out, "theta"))$stat
#' mcmc_out$theta[which.max(mcmc_out$posterior), ]
#' sqrt((fit <- StructTS(log10(UKgas), type = "BSM"))$coef)[c(4, 1:3)]
#'
 
bsm_lg <- function(y, sd_y, sd_level, sd_slope, sd_seasonal,
  beta, xreg = NULL, period, a1 = NULL, P1 = NULL, D = NULL, 
  C = NULL) {
  
  y <- check_y(y)
  n <- length(y)
  
  regression_part <- create_regression(beta, xreg, n)
  
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
    period <- 1L
  } else {
    if (missing(period)) period <- frequency(y)
    period <- check_period(period, n)
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
  
  a1 <- check_a1(a1, m)
  
  if (missing(P1)) {
    P1 <- diag(100, m)
  } else {
    P1 <- check_P1(P1, m)
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
  
  
  if(ncol(regression_part$xreg) > 1) {
    priors <- c(list(sd_y, sd_level, sd_slope, sd_seasonal), 
      regression_part$beta)
  } else {
    priors <- list(sd_y, sd_level, sd_slope, sd_seasonal, 
      regression_part$beta)
  }
  names(priors) <- c("sd_y", "sd_level", "sd_slope", "sd_seasonal", 
    names(regression_part$coefs))
  priors <- priors[vapply(priors, is_prior, TRUE)]
  
  D <- check_D(D, 1L, n)
  C <- check_C(C, m, n)
  
  theta <- if (length(priors) > 0) {
    vapply(priors, "[[", "init", FUN.VALUE = 1) 
  } else numeric(0)
  priors <- combine_priors(priors)
  
  structure(list(y = as.ts(y), Z = Z, H = H, T = T, R = R,
    a1 = a1, P1 = P1, xreg = regression_part$xreg, 
    beta = regression_part$coefs, D = D, C = C,
    slope = slope, seasonal = seasonal, period = period, 
    fixed = as.integer(!notfixed), 
    prior_distributions = priors$prior_distribution, 
    prior_parameters = priors$parameters,
    theta = theta), class = c("bsm_lg", "ssm_ulg", "lineargaussian", 
      "bssm_model"))
}

#' Non-Gaussian Basic Structural (Time Series) Model
#'
#' Constructs a non-Gaussian basic structural model with local level or
#' local trend component, a seasonal component, and regression component
#' (or subset of these components).
#'
#' @param y A vector or a \code{ts} object of observations.
#' @param sd_level Standard deviation of the noise of level equation.
#' Should be an object of class \code{bssm_prior} or scalar 
#' value defining a known value such as 0. 
#' @param sd_slope Standard deviation of the noise of slope equation.
#' Should be an object of class \code{bssm_prior}, scalar 
#' value defining a known value such as 0, or missing, in which case the slope 
#' term is omitted from the model.
#' @param sd_seasonal Standard deviation of the noise of seasonal equation.
#' Should be an object of class \code{bssm_prior}, scalar 
#' value defining a known value such as 0, or missing, in which case the 
#' seasonal term is omitted from the model.
#' @param sd_noise A prior for the standard deviation of the additional noise 
#' term to be added to linear predictor, defined as an object of class 
#' \code{bssm_prior}. If missing, no additional noise term is used.
#' @param distribution Distribution of the observed time series. Possible 
#' choices are \code{"poisson"}, \code{"binomial"}, \code{"gamma"}, and 
#' \code{"negative binomial"}.
#' @param phi Additional parameter relating to the non-Gaussian distribution.
#' For negative binomial distribution this is the dispersion term, for gamma 
#' distribution this is the shape parameter, and for other distributions this 
#' is ignored. Should an object of class \code{bssm_prior} or 
#' a positive scalar.
#' @param u A vector of positive constants for non-Gaussian models. For Poisson, 
#' gamma, and negative binomial distribution, this corresponds to the offset 
#' term. For binomial, this is the number of trials.
#' @param beta A prior for the regression coefficients. 
#' Should be an object of class \code{bssm_prior} or \code{bssm_prior_list} 
#' (in case of multiple coefficients) or missing in case of no covariates.
#' @param xreg A matrix containing covariates with number of rows matching the 
#' length of \code{y}. Can also be \code{ts}, \code{mts} or similar object 
#' convertible to matrix.
#' @param period Length of the seasonal pattern. 
#' Must be a positive value greater than 2 and less than the length of the 
#' input time series. Default is \code{frequency(y)}, 
#' which can also return non-integer value (in which case error is given).
#' @param a1 Prior means for the initial states (level, slope, seasonals).
#' Defaults to vector of zeros.
#' @param P1 Prior covariance matrix for the initial states (level, slope, seasonals).
#' Default is diagonal matrix with 1000 on the diagonal.
#' @param C Intercept terms for state equation, given as a m x n or m x 1 
#' matrix.
#' @return An object of class \code{bsm_ng}.
#' @export
#' @examples
#' # Same data as in Vihola, Helske, Franks (2020)
#' data(poisson_series)
#' s <- sd(log(pmax(0.1, poisson_series)))
#' model <- bsm_ng(poisson_series, sd_level = uniform(0.115, 0, 2 * s),
#'  sd_slope = uniform(0.004, 0, 2 * s), P1 = diag(0.1, 2), 
#'  distribution = "poisson")
#' 
#' \donttest{
#' out <- run_mcmc(model, iter = 1e5, particles = 10)
#' summary(out, variable = "theta", return_se = TRUE)
#' # should be about 0.093 and 0.016
#' summary(out, variable = "states", return_se = TRUE, 
#'  states = 1, times = c(1, 100))
#' # should be about -0.075, 2.618
#' }
#' 
#' model <- bsm_ng(Seatbelts[, "VanKilled"], distribution = "poisson",
#'   sd_level = halfnormal(0.01, 1),
#'   sd_seasonal = halfnormal(0.01, 1),
#'   beta = normal(0, 0, 10),
#'   xreg = Seatbelts[, "law"],
#'   # default values, just for illustration
#'   period = 12L,
#'   a1 = rep(0, 1 + 11), # level + period - 1 seasonal states
#'   P1 = diag(1, 12),
#'   C = matrix(0, 12, 1),
#'   u = rep(1, nrow(Seatbelts)))
#'
#' \donttest{
#' set.seed(123)
#' mcmc_out <- run_mcmc(model, iter = 5000, particles = 10, mcmc_type = "da")
#' mcmc_out$acceptance_rate
#' theta <- expand_sample(mcmc_out, "theta")
#' plot(theta)
#' summary(theta)
#' 
#' library("ggplot2")
#' ggplot(as.data.frame(theta[,1:2]), aes(x = sd_level, y = sd_seasonal)) +
#'   geom_point() + stat_density2d(aes(fill = ..level.., alpha = ..level..),
#'   geom = "polygon") + scale_fill_continuous(low = "green", high = "blue") +
#'   guides(alpha = "none")
#'
#' # Traceplot using as.data.frame method for MCMC output
#' library("dplyr")
#' as.data.frame(mcmc_out) %>% 
#'   filter(variable == "sd_level") %>% 
#'   ggplot(aes(y = value, x = iter)) + geom_line()
#'   
#' }
#' # Model with slope term and additional noise to linear predictor to capture 
#' # excess variation   
#' model2 <- bsm_ng(Seatbelts[, "VanKilled"], distribution = "poisson",
#'   sd_level = halfnormal(0.01, 1),
#'   sd_seasonal = halfnormal(0.01, 1),
#'   beta = normal(0, 0, 10),
#'   xreg = Seatbelts[, "law"],
#'   sd_slope = halfnormal(0.01, 0.1),
#'   sd_noise = halfnormal(0.01, 1))
#'
#' # instead of extra noise term, model using negative binomial distribution:
#' model3 <- bsm_ng(Seatbelts[, "VanKilled"], 
#'   distribution = "negative binomial",
#'   sd_level = halfnormal(0.01, 1),
#'   sd_seasonal = halfnormal(0.01, 1),
#'   beta = normal(0, 0, 10),
#'   xreg = Seatbelts[, "law"],
#'   sd_slope = halfnormal(0.01, 0.1),
#'   phi = gamma_prior(1, 5, 5)) 
#' 
bsm_ng <- function(y, sd_level, sd_slope, sd_seasonal, sd_noise,
  distribution, phi, u, beta, xreg = NULL, period, 
  a1 = NULL, P1 = NULL, C = NULL) {
  
  distribution <- match.arg(tolower(distribution), 
    c("poisson", "binomial", "negative binomial", "gamma"))
  
  y <- check_y(y, multivariate = FALSE, distribution)
  n <- length(y)
  
  if (missing(u)) u <- rep(1, n)
  u <- check_u(u, y)
  
  regression_part <- create_regression(beta, xreg, n)
  
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
    period <- 1L
  } else {
    if (missing(period)) period <- frequency(y)
    period <- check_period(period, n)
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
  
  m <- as.integer(1L + as.integer(slope) + 
      as.integer(seasonal) * (period - 1) + as.integer(noise))
  
  
  a1 <- check_a1(a1, m)
  if (missing(P1)) {
    P1 <- diag(100, m)
  } else {
    P1 <- check_P1(P1, m)
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
  
  phi_est <- FALSE
  use_phi <- distribution %in% c("negative binomial", "gamma")
  if (use_phi) {
    if (is_prior(phi)) {
      check_phi(phi$init)
      phi_est <- TRUE
    } else {
      check_phi(phi)
    }
  } else {
    phi <- 1
  }
  
  initial_mode <- matrix(init_mode(y, u, distribution), ncol = 1)
  
  dim(T) <- c(m, m, 1)
  dim(R) <- c(m, ncol(R), 1)
  
  names(a1) <- rownames(P1) <- colnames(P1) <- rownames(Z) <-
    rownames(T) <- colnames(T) <- rownames(R) <- state_names
  
  if(ncol(regression_part$xreg) > 1) {
    priors <- c(list(sd_level, sd_slope, sd_seasonal, sd_noise, phi), 
      regression_part$beta)
  } else {
    priors <- list(sd_level, sd_slope, sd_seasonal, sd_noise, phi, 
      regression_part$beta)
  }
  names(priors) <- c("sd_level", "sd_slope", "sd_seasonal", "sd_noise", "phi",
    names(regression_part$coefs))
  priors <- priors[vapply(priors, is_prior, TRUE)]
  
  if (phi_est) {
    phi <- phi$init
  }
  
  D <- numeric(1)
  C <- check_C(C, m, n)
  
  theta <- if (length(priors) > 0) {
    vapply(priors, "[[", "init", FUN.VALUE = 1) 
  } else numeric(0)
  priors <- combine_priors(priors)
  
  structure(list(y = as.ts(y), Z = Z, T = T, R = R,
    a1 = a1, P1 = P1, phi = phi, u = u, xreg = regression_part$xreg, 
    beta = regression_part$coefs, D = D, C = C,
    slope = slope, seasonal = seasonal, noise = noise,
    period = period, fixed = as.integer(!notfixed),
    distribution = distribution, initial_mode = initial_mode, 
    prior_distributions = priors$prior_distribution, 
    prior_parameters = priors$parameters,
    theta = theta, phi_est = phi_est,
    max_iter = 100, conv_tol = 1e-8, local_approx = TRUE), 
    class = c("bsm_ng", "ssm_ung", "nongaussian", "bssm_model"))
}

#' Stochastic Volatility Model
#'
#' Constructs a simple stochastic volatility model with Gaussian errors and
#' first order autoregressive signal. See the main vignette for details.
#' 
#' @param y A numeric vector or a \code{\link{ts}} object of observations.
#' @param mu A prior for mu parameter of transition equation. 
#' Should be an object of class \code{bssm_prior}.
#' @param rho A prior for autoregressive coefficient. 
#' Should be an object of class \code{bssm_prior}.
#' @param sd_ar A prior for the standard deviation of noise of the AR-process.
#' Should be an object of class \code{bssm_prior}.
#' @param sigma A prior for sigma parameter of observation equation, internally 
#' denoted as phi. Should be an object of class \code{bssm_prior}. 
#' Ignored if \code{mu} is provided. Note that typically 
#' parametrization using mu is preferred due to better numerical properties and 
#' availability of better Gaussian approximation. 
#' Most notably the global approximation approach does not work with sigma 
#' parameterization as sigma is not a parameter of the resulting approximate 
#' model.
#' @return An object of class \code{svm}.
#' @export
#' @rdname svm
#' @examples
#'
#' data("exchange")
#' y <- exchange[1:100] # for faster CRAN check
#' model <- svm(y, rho = uniform(0.98, -0.999, 0.999),
#'  sd_ar = halfnormal(0.15, 5), sigma = halfnormal(0.6, 2))
#'
#' obj <- function(pars) {
#'    -logLik(svm(y, 
#'      rho = uniform(pars[1], -0.999, 0.999),
#'      sd_ar = halfnormal(pars[2], 5),
#'      sigma = halfnormal(pars[3], 2)), particles = 0)
#' }
#' opt <- optim(c(0.98, 0.15, 0.6), obj, 
#'   lower = c(-0.999, 1e-4, 1e-4),
#'   upper = c(0.999, 10, 10), method = "L-BFGS-B")
#' pars <- opt$par
#' model <- svm(y, 
#'   rho = uniform(pars[1],-0.999,0.999),
#'   sd_ar = halfnormal(pars[2], 5),
#'   sigma = halfnormal(pars[3], 2))
#' 
#' # alternative parameterization  
#' model2 <- svm(y, rho = uniform(0.98,-0.999, 0.999),
#'  sd_ar = halfnormal(0.15, 5), mu = normal(0, 0, 1))
#'
#' obj2 <- function(pars) {
#'    -logLik(svm(y, 
#'      rho = uniform(pars[1], -0.999, 0.999),
#'      sd_ar = halfnormal(pars[2], 5),
#'      mu = normal(pars[3], 0, 1)), particles = 0)
#' }
#' opt2 <- optim(c(0.98, 0.15, 0), obj2, lower = c(-0.999, 1e-4, -Inf),
#'   upper = c(0.999, 10, Inf), method = "L-BFGS-B")
#' pars2 <- opt2$par
#' model2 <- svm(y, 
#'   rho = uniform(pars2[1],-0.999,0.999),
#'   sd_ar = halfnormal(pars2[2], 5),
#'   mu = normal(pars2[3], 0, 1))
#'
#' # sigma is internally stored in phi
#' ts.plot(cbind(model$phi * exp(0.5 * fast_smoother(model)), 
#'   exp(0.5 * fast_smoother(model2))), col = 1:2)
#'
svm <- function(y, mu, rho, sd_ar, sigma) {
  
  if(!missing(sigma) && !missing(mu)) {
    stop("Define either sigma or mu, but not both.")
  }
  
  y <- check_y(y)
  
  check_rho(rho$init)
  check_sd(sd_ar$init, "rho")
  if(!missing(mu)) {
    svm_type <- 1L
    check_mu(mu$init)
    initial_mode <- matrix(log(pmax(1e-4, y^2)), ncol = 1)
  } else {
    svm_type <- 0L
    check_sd(sigma$init, "sigma", FALSE)
    initial_mode <- 
      matrix(log(pmax(1e-4, y^2)) - 2 * log(sigma$init), ncol = 1)
  }
  a1 <- if(svm_type) mu$init else 0
  P1 <- matrix(sd_ar$init^2 / (1 - rho$init^2))
  
  Z <- matrix(1)
  T <- array(rho$init, c(1, 1, 1))
  R <- array(sd_ar$init, c(1, 1, 1))
  
  names(a1) <- rownames(P1) <- colnames(P1) <- rownames(Z) <-
    rownames(T) <- colnames(T) <- rownames(R) <- "signal"
  
  priors <- list(rho, sd_ar, if(svm_type==0) sigma else mu)
  priors <- priors[!vapply(priors, is.null, TRUE)]
  names(priors) <-
    c("rho", "sd_ar", if(svm_type==0) "sigma" else "mu")
  
  C <- if (svm_type) matrix(mu$init * (1 - T[1])) else matrix(0)
  D <- matrix(0)
  
  theta <- if (length(priors) > 0) {
    vapply(priors, "[[", "init", FUN.VALUE = 1) 
  } else numeric(0)
  priors <- combine_priors(priors)
  
  structure(list(y = as.ts(y), Z = Z, T = T, R = R,
    a1 = a1, P1 = P1, phi = if (svm_type == 0) sigma$init else 1, 
    xreg = matrix(0, 0, 0), 
    beta = numeric(0), D = D, C = C, 
    initial_mode = initial_mode, 
    svm_type = svm_type, distribution = "svm", u = 1, 
    phi_est = !as.logical(svm_type),
    prior_distributions = priors$prior_distribution, 
    prior_parameters = priors$parameters,
    theta = theta,
    max_iter = 100, conv_tol = 1e-8, local_approx = TRUE),
    class = c("svm", "ssm_ung", "nongaussian", "bssm_model"))
}
#' Non-Gaussian model with AR(1) latent process
#'
#' Constructs a simple non-Gaussian model where the state dynamics follow an 
#' AR(1) process.
#' 
#' @inheritParams bsm_ng
#' @param rho A prior for autoregressive coefficient. 
#' Should be an object of class \code{bssm_prior}.
#' @param mu A fixed value or a prior for the stationary mean of the latent 
#' AR(1) process. Should be an object of class \code{bssm_prior} or scalar 
#' value defining a fixed mean such as 0.
#' @param sigma A prior for the standard deviation of noise of the AR-process. 
#' Should be an object of class \code{bssm_prior}
#' @return An object of class \code{ar1_ng}.
#' @export
#' @rdname ar1_ng
#' @examples 
#' model <- ar1_ng(discoveries, rho = uniform(0.5,-1,1), 
#'   sigma = halfnormal(0.1, 1), mu = normal(0, 0, 1), 
#'   distribution = "poisson")
#' out <- run_mcmc(model, iter = 1e4, mcmc_type = "approx",
#'   output_type = "summary")
#'   
#' ts.plot(cbind(discoveries, exp(out$alphahat)), col = 1:2)
#' 
#' set.seed(1)
#' n <- 30
#' phi <- 2
#' rho <- 0.9
#' sigma <- 0.1
#' beta <- 0.5
#' u <- rexp(n, 0.1)
#' x <- rnorm(n)
#' z <- y <- numeric(n)
#' z[1] <- rnorm(1, 0, sigma / sqrt(1 - rho^2))
#' y[1] <- rnbinom(1, mu = u * exp(beta * x[1] + z[1]), size = phi)
#' for(i in 2:n) {
#'   z[i] <- rnorm(1, rho * z[i - 1], sigma)
#'   y[i] <- rnbinom(1, mu = u * exp(beta * x[i] + z[i]), size = phi)
#' }
#' 
#' model <- ar1_ng(y, rho = uniform_prior(0.9, 0, 1), 
#'   sigma = gamma_prior(0.1, 2, 10), mu = 0., 
#'   phi = gamma_prior(2, 2, 1), distribution = "negative binomial",
#'   xreg = x, beta = normal_prior(0.5, 0, 1), u = u)
#' 
ar1_ng <- function(y, rho, sigma, mu, distribution, phi, u, beta, 
  xreg = NULL) {
  
  distribution <- match.arg(tolower(distribution), 
    c("poisson", "binomial", "negative binomial", "gamma"))
  
  y <- check_y(y, multivariate = FALSE, distribution)
  n <- length(y)
  
  if (missing(u)) u <- rep(1, n)
  u <- check_u(u, y)
  
  regression_part <- create_regression(beta, xreg, n)
  
  check_rho(rho$init)
  check_sd(sigma$init, "rho")
  
  if (is_prior(mu)) {
    check_mu(mu$init)
    mu_est <- TRUE
    a1 <- mu$init
    C <- matrix(mu$init * (1 - rho$init))
  } else {
    mu_est <- FALSE
    check_mu(mu)
    a1 <- mu
    C <- matrix(mu * (1 - rho$init))
  }
  distribution <- match.arg(tolower(distribution), c("poisson", "binomial",
    "negative binomial", "gamma"))
  
  use_phi <- distribution %in% c("negative binomial", "gamma")
  phi_est <- FALSE
  if (use_phi) {
    if (is_prior(phi)) {
      check_phi(phi$init)
      phi_est <- TRUE
    } else {
      check_phi(phi)
    }
  } else {
    phi <- 1
  }
  
  initial_mode <- matrix(init_mode(y, u, distribution), ncol = 1)
  P1 <- matrix(sigma$init^2 / (1 - rho$init^2))
  
  Z <- matrix(1)
  T <- array(rho$init, c(1, 1, 1))
  R <- array(sigma$init, c(1, 1, 1))
  
  names(a1) <- rownames(P1) <- colnames(P1) <- rownames(Z) <-
    rownames(T) <- colnames(T) <- rownames(R) <- "signal"
  
  
  if(ncol(regression_part$xreg) > 1) {
    priors <- c(list(rho, sigma, mu, phi), regression_part$beta)
  } else {
    priors <- list(rho, sigma, mu, phi, regression_part$beta)
  }
  names(priors) <-
    c("rho", "sigma", "mu", "phi", names(regression_part$coefs))
  priors <- priors[vapply(priors, is_prior, TRUE)]
  
  if (phi_est) {
    phi <- phi$init
  }
  D <- matrix(0)
  
  theta <- if (length(priors) > 0) {
    vapply(priors, "[[", "init", FUN.VALUE = 1) 
  } else numeric(0)
  priors <- combine_priors(priors)
  
  structure(list(y = as.ts(y), Z = Z, T = T, R = R,
    a1 = a1, P1 = P1, phi = phi, u = u, 
    xreg = regression_part$xreg, beta = regression_part$coefs,
    D = D, C = C,
    initial_mode = initial_mode,
    distribution = distribution, mu_est = mu_est, phi_est = phi_est,
    prior_distributions = priors$prior_distribution, 
    prior_parameters = priors$parameters, theta = theta, 
    max_iter = 100, conv_tol = 1e-8, local_approx = TRUE),
    class = c("ar1_ng", "ssm_ung", "nongaussian", "bssm_model"))
}
#' Univariate Gaussian model with AR(1) latent process
#'
#' Constructs a simple Gaussian model where the state dynamics 
#' follow an AR(1) process.
#' 
#' @inheritParams ar1_ng
#' @param sd_y A prior for the standard deviation of observation equation.
#' @return An object of class \code{ar1_lg}.
#' @export
#' @rdname ar1_lg
#' @examples 
#' set.seed(1)
#' mu <- 2
#' rho <- 0.7
#' sd_y <- 0.1
#' sigma <- 0.5
#' beta <- -1
#' x <- rnorm(30)
#' z <- y <- numeric(30)
#' z[1] <- rnorm(1, mu, sigma / sqrt(1 - rho^2))
#' y[1] <- rnorm(1, beta * x[1] + z[1], sd_y)
#' for(i in 2:30) {
#'   z[i] <- rnorm(1, mu * (1 - rho) + rho * z[i - 1], sigma)
#'   y[i] <- rnorm(1, beta * x[i] + z[i], sd_y)
#' }
#' model <- ar1_lg(y, rho = uniform(0.5, -1, 1), 
#'   sigma = halfnormal(1, 10), mu = normal(0, 0, 1), 
#'   sd_y = halfnormal(1, 10), 
#'   xreg = x,  beta = normal(0, 0, 1))
#' out <- run_mcmc(model, iter = 2e4)
#' summary(out, return_se = TRUE)
#' 
ar1_lg <- function(y, rho, sigma, mu, sd_y, beta, xreg = NULL) {
  
  y <- check_y(y)
  n <- length(y)
  regression_part <- create_regression(beta, xreg, n)
  
  check_rho(rho$init)
  check_sd(sigma$init, "rho")
  
  if (is_prior(mu)) {
    check_mu(mu$init)
    mu_est <- TRUE
    a1 <- mu$init
    C <- matrix(mu$init * (1 - rho$init))
  } else {
    mu_est <- FALSE
    check_mu(mu)
    a1 <- mu
    C <- matrix(mu * (1 - rho$init))
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
  
  
  if(ncol(regression_part$xreg) > 1) {
    priors <- c(list(rho, sigma, mu, sd_y), regression_part$beta)
  } else {
    priors <- list(rho, sigma, mu, sd_y, regression_part$beta)
  }
  names(priors) <-
    c("rho", "sigma", "mu", "sd_y", names(regression_part$coefs))
  priors <- priors[vapply(priors, is_prior, TRUE)]
  
  D <- matrix(0)
  
  theta <- if (length(priors) > 0) {
    vapply(priors, "[[", "init", FUN.VALUE = 1) 
  } else numeric(0)
  priors <- combine_priors(priors)
  
  structure(list(y = as.ts(y), Z = Z, H = H, T = T, R = R,
    a1 = a1, P1 = P1, 
    xreg = regression_part$xreg, beta = regression_part$coefs,
    D = D, C = C,
    mu_est = mu_est, sd_y_est = sd_y_est,
    prior_distributions = priors$prior_distribution, 
    prior_parameters = priors$parameters, theta = theta,
    max_iter = 100, conv_tol = 1e-8),
    class = c("ar1_lg", "ssm_ulg", "lineargaussian", "bssm_model"))
}

#'
#' General multivariate nonlinear Gaussian state space models
#'
#' Constructs an object of class \code{ssm_nlg} by defining the corresponding 
#' terms of the observation and state equation.
#' 
#' The nonlinear Gaussian model is defined as
#'
#' \deqn{y_t = Z(t, \alpha_t, \theta) + H(t, \theta) \epsilon_t, 
#' (\textrm{observation equation})}
#' \deqn{\alpha_{t+1} = T(t, \alpha_t, \theta) + R(t, \theta)\eta_t, 
#' (\textrm{transition equation})}
#'
#' where \eqn{\epsilon_t \sim N(0, I_p)}, \eqn{\eta_t \sim N(0, I_m)} and
#' \eqn{\alpha_1 \sim N(a_1, P_1)} independently of each other, and functions
#' \eqn{Z, H, T, R} can depend on \eqn{\alpha_t} and parameter vector 
#' \eqn{\theta}.
#'
#' Compared to other models, these general models need a bit more effort from
#' the user, as you must provide the several small C++ snippets which define the
#' model structure. See examples in the vignette and \code{cpp_example_model}.
#' 
#' @param y Observations as multivariate time series (or matrix) of length 
#' \eqn{n}.
#' @param Z,H,T,R  An external pointers (object of class \code{externalptr}) 
#' for the C++ functions which define the corresponding model functions.
#' @param Z_gn,T_gn An external pointers (object of class \code{externalptr}) 
#' for the C++ functions which define the gradients of the corresponding model 
#' functions.
#' @param a1 Prior mean for the initial state as object of class 
#' \code{externalptr}
#' @param P1 Prior covariance matrix for the initial state as object of class 
#' \code{externalptr}
#' @param theta Parameter vector passed to all model functions.
#' @param known_params A vector of known parameters passed to all model 
#' functions.
#' @param known_tv_params A matrix of known parameters passed to all model 
#' functions.
#' @param n_states Number of states in the model (positive integer).
#' @param n_etas Dimension of the noise term of the transition equation 
#' (positive integer).
#' @param log_prior_pdf An external pointer (object of class 
#' \code{externalptr}) for the C++ function which
#' computes the log-prior density given theta.
#' @param time_varying Optional logical vector of length 4, denoting whether 
#' the values of
#' Z, H, T, and R vary with respect to time variable (given identical states).
#' If used, this can speed up some computations.
#' @param state_names A character vector containing names for the states.
#' @return An object of class \code{ssm_nlg}.
#' @export
#' @examples
#' \donttest{ # Takes a while on CRAN
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
#' out <- ekf(model_nlg, iekf_iter = 100)
#' ts.plot(cbind(x, out$at[1:n], out$att[1:n]), col = 1:3)
#' }
ssm_nlg <- function(y, Z, H, T, R, Z_gn, T_gn, a1, P1, theta,
  known_params = NA, known_tv_params = matrix(NA), n_states, n_etas,
  log_prior_pdf, time_varying = rep(TRUE, 4), 
  state_names = paste0("state", 1:n_states)) {
  
  if (is.null(dim(y))) {
    dim(y) <- c(length(y), 1)
  }
  
  if(missing(n_etas)) {
    n_etas <- n_states
  }
  n_states <- as.integer(n_states)
  n_etas <- as.integer(n_etas)
  
  theta <- check_theta(theta)
  
  structure(list(y = as.ts(y), Z = Z, H = H, T = T,
    R = R, Z_gn = Z_gn, T_gn = T_gn, a1 = a1, P1 = P1, theta = theta,
    log_prior_pdf = log_prior_pdf, known_params = known_params,
    known_tv_params = known_tv_params,
    n_states = n_states, n_etas = n_etas,
    time_varying = time_varying,
    state_names = state_names,
    max_iter = 100, conv_tol = 1e-8), 
    class = c("ssm_nlg", "bssm_model"))
}

#'
#' Univariate state space model with continuous SDE dynamics
#'
#' Constructs an object of class \code{ssm_sde} by defining the functions for
#' the drift, diffusion and derivative of diffusion terms of univariate SDE,
#' as well as the log-density of observation equation. We assume that the
#' observations are measured at integer times (missing values are allowed).
#'
#' As in case of \code{ssm_nlg} models, these general models need a bit more 
#' effort from the user, as you must provide the several small C++ snippets 
#' which define the model structure. See vignettes for an example and 
#' \code{cpp_example_model}.
#'
#' @param y Observations as univariate time series (or vector) of length 
#' \eqn{n}.
#' @param drift,diffusion,ddiffusion An external pointers for the C++ functions 
#' which
#' define the drift, diffusion and derivative of diffusion functions of SDE.
#' @param obs_pdf An external pointer for the C++ function which
#' computes the observational log-density given the the states and parameter 
#' vector theta.
#' @param prior_pdf An external pointer for the C++ function which
#' computes the prior log-density given the parameter vector theta.
#' @param theta Parameter vector passed to all model functions.
#' @param x0 Fixed initial value for SDE at time 0.
#' @param positive If \code{TRUE}, positivity constraint is
#'   forced by \code{abs} in Milstein scheme.
#' @return An object of class \code{ssm_sde}.
#' @export
#' @examples
#' 
#' \donttest{ # Takes a while on CRAN
#' library("sde")
#' set.seed(1)
#' # theta_0 = rho = 0.5
#' # theta_1 = nu = 2
#' # theta_2 = sigma = 0.3
#' x <- sde.sim(t0 = 0, T = 50, X0 = 1, N = 50,
#'        drift = expression(0.5 * (2 - x)),
#'        sigma = expression(0.3),
#'        sigma.x = expression(0))
#' y <- rpois(50, exp(x[-1]))
#'
#' # source c++ snippets
#' pntrs <- cpp_example_model("sde_poisson_OU")
#' 
#' sde_model <- ssm_sde(y, pntrs$drift, pntrs$diffusion,
#'  pntrs$ddiffusion, pntrs$obs_density, pntrs$prior,
#'  c(rho = 0.5, nu = 2, sigma = 0.3), 1, positive = FALSE)
#' 
#' est <- particle_smoother(sde_model, L = 12, particles = 500)
#' 
#' ts.plot(cbind(x, est$alphahat, 
#'   est$alphahat - 2*sqrt(c(est$Vt)), 
#'   est$alphahat + 2*sqrt(c(est$Vt))), 
#'   col = c(2, 1, 1, 1), lty = c(1, 1, 2, 2))
#' 
#' 
#' # Takes time with finer mesh, parallelization with IS-MCMC helps a lot
#' out <- run_mcmc(sde_model, L_c = 4, L_f = 8, 
#'   particles = 50, iter = 2e4,
#'   threads = 4L)
#' }
#'
ssm_sde <- function(y, drift, diffusion, ddiffusion, obs_pdf,
  prior_pdf, theta, x0, positive) {
  
  y <- check_y(y)
  theta <- check_theta(theta)
  structure(list(y = as.ts(y), drift = drift,
    diffusion = diffusion,
    ddiffusion = ddiffusion, obs_pdf = obs_pdf,
    prior_pdf = prior_pdf, theta = theta, x0 = x0,
    positive = positive, state_names = "x"), 
    class = c("ssm_sde", "bssm_model"))
}


