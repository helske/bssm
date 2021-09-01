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
#' @param y Observations as time series (or vector) of length \eqn{n}.
#' @param Z System matrix Z of the observation equation as m x 1 or 
#' m x n matrix.
#' @param H Vector of standard deviations. Either a scalar or a vector of 
#' length n.
#' @param T System matrix T of the state equation. Either a m x m matrix or a
#' m x m x n array.
#' @param R Lower triangular matrix R the state equation. Either a 
#' m x k matrix or a m x k x n array.
#' @param a1 Prior mean for the initial state as a vector of length m.
#' @param P1 Prior covariance matrix for the initial state as m x m matrix.
#' @param init_theta Initial values for the unknown hyperparameters theta.
#' @param D Intercept terms for observation equation, given as a 
#' length n vector.
#' @param C Intercept terms for state equation, given as m x n matrix.
#' @param update_fn Function which returns list of updated model 
#' components given input vector theta. See details.
#' @param prior_fn Function which returns log of prior density 
#' given input vector theta.
#' @param state_names Names for the states.
#' @return Object of class \code{ssm_ulg}.
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
#'   update_fn = update_fn, prior_fn = prior_fn)
#' 
#' out <- run_mcmc(model, iter = 10000)
#' out
#' sumr <- summary(out, variable = "state")
#' ts.plot(sumr$Mean, col = 1:3)
#' lines(b1, col= 2, lty = 2)
#' lines(b2, col= 3, lty = 2)
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
#' updatefn <- function(theta) {
#'   
#'   model_kfas <- SSModel(log(drivers) ~ SSMtrend(1, Q = theta[1]^2)+
#'     SSMseasonal(period = 12, 
#'       sea.type = "trigonometric", Q = theta[2]^2) +
#'     log(PetrolPrice) + law, data = Seatbelts, H = theta[3]^2)
#'   
#'   as_bssm(model_kfas, kappa = 100)
#' }
#' 
#' prior <- function(theta) {
#'   if(any(theta < 0)) -Inf else sum(dnorm(theta, 0, 0.1, log = TRUE))
#' }
#' init_theta <- rep(1e-2, 3)
#' c("sd_level", "sd_seasonal", "sd_y")
#' model_bssm <- as_bssm(model_kfas, kappa = 100, 
#'   init_theta = init_theta, 
#'   prior_fn = prior, update_fn = updatefn)
#' 
#' \dontrun{
#' out <- run_mcmc(model_bssm, iter = 10000, burnin = 5000) 
#' out
#' 
#' # Above the regression coefficients are modelled as 
#' # time-invariant latent states. 
#' # Here is an alternative way where we use variable D so that the
#' # coefficients are part of parameter vector theta:
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
#' 
#' out2 <- run_mcmc(model_bssm2, iter = 10000, burnin = 5000) 
#' out2
#' }
ssm_ulg <- function(y, Z, H, T, R, a1, P1, init_theta = numeric(0),
  D, C, state_names, update_fn = default_update_fn, 
  prior_fn = default_prior_fn) {
  
  check_y(y)
  n <- length(y)
  
  # create Z
  Z <- check_Z(Z, 1L, n)
  m <- dim(Z)[1]
  
  # create T
  T <- check_T(T, m, n)
  
  # create R
  R <- check_R(R)
  
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
    xreg = matrix(0, 0, 0), beta = numeric(0)), class = c("ssm_ulg", "gaussian"))
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
#' @param distribution Distribution of the observed time series. Possible 
#' choices are
#' \code{"poisson"}, \code{"binomial"}, \code{"gamma"}, and 
#' \code{"negative binomial"}.
#' @param phi Additional parameter relating to the non-Gaussian distribution.
#' For negative binomial distribution this is the dispersion term, for 
#' gamma distribution this is the shape parameter, and for other 
#' distributions this is ignored.
#' @param u Constant parameter vector for non-Gaussian models. For Poisson, 
#' gamma, and negative binomial distribution, this corresponds to the offset 
#' term. For binomial, this is the number of trials.
#' @param state_names Names for the states.
#' @param C Intercept terms \eqn{C_t} for the state equation, given as a
#'  m times 1 or m times n matrix.
#' @param D Intercept terms \eqn{D_t} for the observations equation, given as a
#' scalar or vector of length n.
#' @param init_theta Initial values for the unknown hyperparameters theta.
#' @param update_fn Function which returns list of updated model 
#' components given input vector theta. See details.
#' @param prior_fn Function which returns log of prior density 
#' given input vector theta.
#' @return Object of class \code{ssm_ung}.
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
ssm_ung <- function(y, Z, T, R, a1, P1, distribution, phi = 1, u = 1, 
  init_theta = numeric(0), D, C, state_names, update_fn = default_update_fn,
  prior_fn = default_prior_fn) {
  
  
  distribution <- match.arg(distribution, 
    c("poisson", "binomial", "negative binomial", "gamma"))
  
  check_y(y, distribution = distribution)
  n <- length(y)
  Z <- check_Z(Z, 1L, n)
  m <- dim(Z)[1]
  
  T <- check_T(T, m, n)
  
  # create R
  R <- check_R(R)
  
  a1 <- check_a1(a1, m)
  
  P1 <- check_P1(P1, m)
  
  D <- check_D(D, 1L, n)
  C <- check_C(C, m, n)
  
  check_phi(phi)
  
  if (length(u) == 1) {
    u <- rep(u, length.out = n)
  }
  check_u(u)
  
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
    class = c("ssm_ung", "nongaussian"))
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
#' @param a1 Prior mean for the initial state as a vector of length m.
#' @param P1 Prior covariance matrix for the initial state as m x m matrix.
#' @param init_theta Initial values for the unknown hyperparameters theta.
#' @param D Intercept terms for observation equation, given as a p x n matrix.
#' @param C Intercept terms for state equation, given as m x n matrix.
#' @param update_fn Function which returns list of updated model 
#' components given input vector theta. This function should take only one 
#' vector argument which is used to create list with elements named as
#' \code{Z}, \code{H} \code{T}, \code{R}, \code{a1}, \code{P1}, \code{D}, 
#' and \code{C},
#' where each element matches the dimensions of the original model.
#' If any of these components is missing, it is assumed to be 
#' onstant wrt. theta.
#' @param prior_fn Function which returns log of prior density 
#' given input vector theta.
#' @param state_names Names for the states.
#' @return Object of class \code{ssm_mlg}.
#' @export
#' @examples
#' 
#' data("GlobalTemp", package = "KFAS")
#' model_temp <- ssm_mlg(GlobalTemp, H = matrix(c(0.15,0.05,0, 0.05), 2, 2), 
#'   R = 0.05, Z = matrix(1, 2, 1), T = 1, P1 = 10)
#' ts.plot(cbind(model_temp$y, smoother(model_temp)$alphahat),col=1:3)
#' 
ssm_mlg <- function(y, Z, H, T, R, a1, P1, init_theta = numeric(0),
  D, C, state_names, update_fn = default_update_fn, 
  prior_fn = default_prior_fn) {
  
  # create y
  check_y(y, multivariate = TRUE)
  n <- nrow(y)
  p <- ncol(y)
  
  # create Z
  Z <- check_Z(Z, p, n)
  m <- dim(Z)[2]
  
  T <- check_T(T, m, n)
  
  # create R
  R <- check_R(R)
  
  a1 <- check_a1(a1, m)
  P1 <- check_P1(P1, m)
  
  D <- check_D(D, p, n)
  C <- check_C(C, m, n)
  
  H <- check_H(H, p, n)
  
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
    state_names = state_names), class = c("ssm_mlg", "gaussian"))
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
#' 
#' @param y Observations as multivariate time series or matrix with dimensions 
#' n x p.
#' @param Z System matrix Z of the observation equation as p x m matrix or 
#' p x m x n array.
#' @param T System matrix T of the state equation. Either a m x m matrix or a
#' m x m x n array.
#' @param R Lower triangular matrix R the state equation. Either a m x k 
#' matrix or a
#' m x k x n array.
#' @param a1 Prior mean for the initial state as a vector of length m.
#' @param P1 Prior covariance matrix for the initial state as m x m matrix.
#' @param distribution vector of distributions of the observed series. 
#' Possible choices are
#' \code{"poisson"}, \code{"binomial"}, \code{"negative binomial"}, 
#' \code{"gamma"}, and \code{"gaussian"}.
#' @param phi Additional parameters relating to the non-Gaussian distributions.
#' For negative binomial distribution this is the dispersion term, for 
#' gamma distribution this is the shape parameter, for Gaussian this is 
#' standard deviation, 
#' and for other distributions this is ignored.
#' @param u Constant parameter for non-Gaussian models. For Poisson, gamma, 
#' and negative binomial distribution, this corresponds to the offset term. 
#' For binomial, this is the number of trials.
#' @param init_theta Initial values for the unknown hyperparameters theta.
#' @param D Intercept terms for observation equation, given as p x n matrix.
#' @param C Intercept terms for state equation, given as m x n matrix.
#' @param update_fn Function which returns list of updated model 
#' components given input vector theta. This function should take only one 
#' vector argument which is used to create list with elements named as
#' \code{Z}, \code{T}, \code{R}, \code{a1}, \code{P1}, \code{D}, \code{C}, and
#' \code{phi},
#' where each element matches the dimensions of the original model.
#' If any of these components is missing, it is assumed to be constant wrt. 
#' theta.
#' @param prior_fn Function which returns log of prior density 
#' given input vector theta.
#' @param state_names Names for the states.
#' @return Object of class \code{ssm_mng}.
#' @export
ssm_mng <- function(y, Z, T, R, a1, P1, distribution, phi = 1, u = 1, 
  init_theta = numeric(0), D, C, state_names, update_fn = default_update_fn,
  prior_fn = default_prior_fn) {
  
  # create y
  
  check_y(y, multivariate = TRUE)
  n <- nrow(y)
  p <- ncol(y)
  if(length(distribution) == 1) distribution <- rep(distribution, p)
  check_distribution(y, distribution)
  if(length(phi) == 1) phi <- rep(phi, p)
  for(i in 1:p) {
    distribution[i] <- match.arg(distribution[i], 
      c("poisson", "binomial", "negative binomial", "gamma", "gaussian"))
    check_phi(phi[i])
  }
  
  # create Z
  Z <- check_Z(Z, p, n)
  m <- dim(Z)[2]
  
  T <- check_T(T, m, n)
  
  # create R
  R <- check_R(R)
  
  a1 <- check_a1(a1, m)
  P1 <- check_P1(P1, m)
  
  D <- check_D(D, p, n)
  C <- check_C(C, m, n)
  
  if (length(u) == 1) {
    u <- matrix(u, n, p)
  }
  check_u(u) 
  if(!identical(dim(y), dim(u))) 
    stop("Dimensions of 'y' and 'u' do not match. ")
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
    class = c("ssm_mng", "nongaussian"))
}
#' Basic Structural (Time Series) Model
#'
#' Constructs a basic structural model with local level or local trend 
#' component and seasonal component.
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
#' @param D,C Intercept terms for observation and
#' state equations, given as a length n vector and m times n matrix 
#' respectively (or scalar and m times 1 matrix).
#' @return Object of class \code{bsm_lg}.
#' @export
#' @examples
#'
#' prior <- uniform(0.1 * sd(log10(UKgas)), 0, 1)
#' model <- bsm_lg(log10(UKgas), sd_y = prior, sd_level =  prior,
#'   sd_slope =  prior, sd_seasonal =  prior)
#'
#' mcmc_out <- run_mcmc(model, iter = 5000)
#' summary(expand_sample(mcmc_out, "theta"))$stat
#' mcmc_out$theta[which.max(mcmc_out$posterior), ]
#' sqrt((fit <- StructTS(log10(UKgas), type = "BSM"))$coef)[c(4, 1:3)]
#'
bsm_lg <- function(y, sd_y, sd_level, sd_slope, sd_seasonal,
  beta, xreg = NULL, period = frequency(y), a1, P1, D, C) {
  
  check_y(y)
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
  
  
  if(ncol(xreg) > 1) {
    priors <- c(list(sd_y, sd_level, sd_slope, sd_seasonal), 
      regression_part$beta)
  } else {
    priors <- list(sd_y, sd_level, sd_slope, sd_seasonal, 
      regresssion_part$beta)
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
    theta = theta), class = c("bsm_lg", "ssm_ulg", "gaussian"))
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
#' See \link[=uniform]{priors} for details. If missing, no additional noise 
#' term is used.
#' @param distribution Distribution of the observed time series. Possible 
#' choices are
#' \code{"poisson"}, \code{"binomial"}, \code{"gamma"}, and 
#' \code{"negative binomial"}.
#' @param phi Additional parameter relating to the non-Gaussian distribution.
#' For negative binomial distribution this is the dispersion term, for 
#' gamma distribution this is the shape parameter, and for other distributions 
#' this is ignored.
#' @param u Constant parameter vector for non-Gaussian models. For Poisson, 
#' gamma, and negative binomial distribution, this corresponds to the offset 
#' term. For binomial, this is the number of trials.
#' @param beta Prior for the regression coefficients.
#' @param xreg Matrix containing covariates.
#' @param period Length of the seasonal component i.e. the number of
#' observations per season. Default is \code{frequency(y)}.
#' @param a1 Prior means for the initial states (level, slope, seasonals).
#' Defaults to vector of zeros.
#' @param P1 Prior covariance for the initial states (level, slope, seasonals).
#' Default is diagonal matrix with 1e5 on the diagonal.
#' @param C Intercept terms for state equation, given as a
#'  m times n matrix.
#' @return Object of class \code{bsm_ng}.
#' @export
#' @examples
#' model <- bsm_ng(Seatbelts[, "VanKilled"], distribution = "poisson",
#'   sd_level = halfnormal(0.01, 1),
#'   sd_seasonal = halfnormal(0.01, 1),
#'   beta = normal(0, 0, 10),
#'   xreg = Seatbelts[, "law"])
#' \dontrun{
#' set.seed(123)
#' mcmc_out <- run_mcmc(model, iter = 5000, particles = 10)
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
#' # Traceplot using as.data.frame method for MCMC output:
#' library("dplyr")
#' as.data.frame(mcmc_out) %>% 
#'   filter(variable == "sd_level") %>% 
#'   ggplot(aes(y = value, x = iter)) + geom_line()
#' }
bsm_ng <- function(y, sd_level, sd_slope, sd_seasonal, sd_noise,
  distribution, phi, u = 1, beta, xreg = NULL, period = frequency(y), a1, P1,
  C) {
  
  
  distribution <- match.arg(distribution, 
    c("poisson", "binomial", "negative binomial", "gamma"))
  
  check_y(y, multivariate = FALSE, distribution)
  n <- length(y)
  
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
      check_phi(phi$init, distribution)
      phi_est <- TRUE
    } else {
      check_phi(phi, distribution)
    }
  } else {
    phi <- 1
  }
  
  check_u(u)
  if (length(u) != n) {
    u <- rep(u, length.out = n)
  }
  
  
  initial_mode <- matrix(init_mode(y, u, distribution), ncol = 1)
  
  dim(T) <- c(m, m, 1)
  dim(R) <- c(m, ncol(R), 1)
  
  names(a1) <- rownames(P1) <- colnames(P1) <- rownames(Z) <-
    rownames(T) <- colnames(T) <- rownames(R) <- state_names
  
  if(ncol(xreg) > 1) {
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
    class = c("bsm_ng", "ssm_ung", "nongaussian"))
}

#' Stochastic Volatility Model
#'
#' Constructs a simple stochastic volatility model with Gaussian errors and
#' first order autoregressive signal.
#'
#' @param y Vector or a \code{\link{ts}} object of observations.
#' @param mu Prior for mu parameter of transition equation.
#' @param rho prior for autoregressive coefficient.
#' @param sd_ar Prior for the standard deviation of noise of the AR-process.
#' @param sigma Prior for sigma parameter of observation equation, internally 
#' denoted as phi. Ignored if \code{mu} is provided. Note that typically 
#' parametrization using mu is preferred due to better numerical properties and 
#' availability of better Gaussian approximation. 
#' Most notably the global approximation approach does not work with sigma 
#' parameterization as sigma is not a parameter of the resulting approximate 
#' model.
#' @return Object of class \code{svm}.
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
#'    sigma = halfnormal(pars[3],sd=2)), particles = 0)
#' }
#' opt <- nlminb(c(0.98, 0.15, 0.6), obj, lower = c(-0.999, 1e-4, 1e-4),
#'   upper = c(0.999,10,10))
#' pars <- opt$par
#' model <- svm(exchange, rho = uniform(pars[1],-0.999,0.999),
#'   sd_ar = halfnormal(pars[2],sd=5),
#'   sigma = halfnormal(pars[3],sd=2))
#'
svm <- function(y, mu, rho, sd_ar, sigma) {
  
  if(!missing(sigma) && !missing(mu)) {
    stop("Define either sigma or mu, but not both.")
  }
  
  check_y(y)
  
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
    class = c("svm", "ssm_ung", "nongaussian"))
}
#' Non-Gaussian model with AR(1) latent process
#'
#' Constructs a simple non-Gaussian model where the state dynamics follow an 
#' AR(1) process.
#'
#' @param y Vector or a \code{\link{ts}} object of observations.
#' @param rho prior for autoregressive coefficient.
#' @param mu A fixed value or a prior for the stationary mean of the latent 
#' AR(1) process. Parameter is omitted if this is set to 0.
#' @param sigma Prior for the standard deviation of noise of the AR-process.
#' @param beta Prior for the regression coefficients.
#' @param xreg Matrix containing covariates.
#' @param distribution Distribution of the observed time series. Possible 
#' choices are \code{"poisson"}, \code{"binomial"}, \code{"gamma"}, and 
#' \code{"negative binomial"}.
#' @param phi Additional parameter relating to the non-Gaussian distribution.
#' For negative binomial distribution this is the dispersion term, for gamma 
#' distribution this is the shape parameter, and for other distributions this 
#' is ignored.
#' @param u Constant parameter vector for non-Gaussian models. For Poisson, 
#' gamma, and negative binomial distribution, this corresponds to the offset 
#' term. For binomial, this is the number of trials.
#' @return Object of class \code{ar1_ng}.
#' @export
#' @rdname ar1_ng
ar1_ng <- function(y, rho, sigma, mu, distribution, phi, u = 1, beta, 
  xreg = NULL) {
  
  distribution <- match.arg(distribution, 
    c("poisson", "binomial", "negative binomial", "gamma"))
  
  check_y(y, multivariate = FALSE, distribution)
  
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
  distribution <- match.arg(distribution, c("poisson", "binomial",
    "negative binomial", "gamma"))
  
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
  
  check_u(u)
  if (length(u) != n) {
    u <- rep(u, length.out = n)
  }
  
  initial_mode <- matrix(init_mode(y, u, distribution), ncol = 1)
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
    class = c("ar1_ng", "ssm_ung", "nongaussian"))
}
#' Univariate Gaussian model with AR(1) latent process
#'
#' Constructs a simple Gaussian model where the state dynamics 
#' follow an AR(1) process.
#'
#' @param y Vector or a \code{\link{ts}} object of observations.
#' @param rho prior for autoregressive coefficient.
#' @param mu A fixed value or a prior for the stationary mean of the latent 
#' AR(1) process. Parameter is omitted if this is set to 0.
#' @param sigma Prior for the standard deviation of noise of the AR-process.
#' @param sd_y Prior for the standard deviation of observation equation.
#' @param beta Prior for the regression coefficients.
#' @param xreg Matrix containing covariates.
#' @return Object of class \code{ar1_lg}.
#' @export
#' @rdname ar1_lg
#' @examples 
#' model <- ar1_lg(BJsales, rho = uniform(0.5,-1,1), 
#'   sigma = halfnormal(1, 10), mu = normal(200, 200, 100), 
#'   sd_y = halfnormal(1, 10))
#' out <- run_mcmc(model, iter = 2e4)
#' summary(out, return_se = TRUE)
ar1_lg <- function(y, rho, sigma, mu, sd_y, beta, xreg = NULL) {
  
  check_y(y)
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
  
  
  if(ncol(xreg) > 1) {
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
    class = c("ar1_lg", "ssm_ulg", "gaussian"))
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
#' model structure. See examples in the vignette.
#' 
#' @param y Observations as multivariate time series (or matrix) of length 
#' \eqn{n}.
#' @param Z,H,T,R  An external pointers for the C++ functions which
#' define the corresponding model functions.
#' @param Z_gn,T_gn An external pointers for the C++ functions which
#' define the gradients of the corresponding model functions.
#' @param a1 Prior mean for the initial state as a vector of length m.
#' @param P1 Prior covariance matrix for the initial state as m x m matrix.
#' @param theta Parameter vector passed to all model functions.
#' @param known_params Vector of known parameters passed to all model 
#' functions.
#' @param known_tv_params Matrix of known parameters passed to all model 
#' functions.
#' @param n_states Number of states in the model.
#' @param n_etas Dimension of the noise term of the transition equation.
#' @param log_prior_pdf An external pointer for the C++ function which
#' computes the log-prior density given theta.
#' @param time_varying Optional logical vector of length 4, denoting whether 
#' the values of
#' Z, H, T, and R vary with respect to time variable (given identical states).
#' If used, this can speed up some computations.
#' @param state_names Names for the states.
#' @return Object of class \code{ssm_nlg}.
#' @export
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
  structure(list(y = as.ts(y), Z = Z, H = H, T = T,
    R = R, Z_gn = Z_gn, T_gn = T_gn, a1 = a1, P1 = P1, theta = theta,
    log_prior_pdf = log_prior_pdf, known_params = known_params,
    known_tv_params = known_tv_params,
    n_states = n_states, n_etas = n_etas,
    time_varying = time_varying,
    state_names = state_names,
    max_iter = 100, conv_tol = 1e-8), 
    class = "ssm_nlg")
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
#' which define the model structure. See vignettes for an example.
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
#' @return Object of class \code{ssm_sde}.
#' @export
#' @examples
#' \dontrun{
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
#' # Template can be found in the vignette
#' Rcpp::sourceCpp("ssm_sde_template.cpp")
#' pntrs <- create_xptrs()
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
#' # Takes time with finer mesh, parallelization with IS-MCMC helps a lot
#' out <- run_mcmc(sde_model, L_c = 4, L_f = 8, 
#'   particles = 50, iter = 2e4,
#'   threads = 4L)
#' 
#' }
ssm_sde <- function(y, drift, diffusion, ddiffusion, obs_pdf,
  prior_pdf, theta, x0, positive) {
  
  check_y(y)
  
  structure(list(y = as.ts(y), drift = drift,
    diffusion = diffusion,
    ddiffusion = ddiffusion, obs_pdf = obs_pdf,
    prior_pdf = prior_pdf, theta = theta, x0 = x0,
    positive = positive, state_names = "x"), class = "ssm_sde")
}


