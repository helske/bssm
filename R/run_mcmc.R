#' Bayesian Inference of State Space Models
#'
#' Adaptive Markov chain Monte Carlo simulation for SSMs using
#' Robust Adaptive Metropolis algorithm by Vihola (2012). Several different
#' MCMC sampling schemes are implemented, see parameter
#' arguments, package vignette, Vihola, Helske, Franks (2020) and Helske and
#' Vihola (2021) for details.
#'
#' @details
#'
#' For linear-Gaussian models, option \code{"summary"} does not simulate
#' states directly but computes the posterior means and variances of states
#' using fast Kalman smoothing. This is slightly faster,
#' more memory efficient and more accurate than calculations based on
#' simulation smoother. In other cases, the means and
#' covariances are computed using the full output of particle filter
#' instead of subsampling one of these as in case of
#' \code{output_type = "full"}. The states are sampled up to the time point n+1
#' where n is the length of the input time series i.e. the last values are
#' one-step-ahead predictions. (for predicting further, see
#' \code{?predict.mcmc_output}).
#'
#' Initial values for the sampling are taken from the model object
#' (\code{model$theta}). If you want to continue from previous run, you can
#' reconstruct your original model by plugging in the previously obtained
#' parameters to \code{model$theta}, providing the S matrix for the RAM
#' algorithm and setting \code{burnin = 0}. See example. Note however, that
#' this is not identical as running all the iterations once, due to the
#' RNG "discontinuity" and because even without burnin bssm does include
#' "theta_0" i.e. the initial theta in the final chain (even with
#' \code{burnin=0}).
#'
#' @importFrom stats tsp
#' @importFrom rlang is_interactive
#' @param model Model of class \code{bssm_model}.
#' @param iter A positive integer defining the total number of MCMC iterations.
#' Suitable value depends on the model, data, and the choice of specific
#' algorithms (\code{mcmc_type} and \code{sampling_method}). As increasing
#' \code{iter} also increases run time, it is is generally good idea to first
#' test the performance with a small values, e.g., less than 10000.
#' @param output_type Either \code{"full"}
#' (default, returns posterior samples from the posterior
#' \eqn{p(\alpha, \theta | y)}), \code{"theta"} (for marginal posterior of
#' theta), or \code{"summary"} (return the mean and variance estimates of the
#' states and posterior samples of theta). See details.
#' @param burnin A positive integer defining the length of the burn-in period
#' which is disregarded from the results. Defaults to \code{iter / 2}.
#' Note that all MCMC algorithms of \code{bssm} use adaptive MCMC during the
#' burn-in period in order to find good proposal distribution.
#' @param thin A positive integer defining the thinning rate. All the MCMC
#' algorithms in \code{bssm} use the jump chain representation (see refs),
#' and the thinning is applied to these blocks. Defaults to 1.
#' For IS-corrected methods, larger value can also be
#' statistically more effective. Note: With \code{output_type = "summary"},
#' the thinning does not affect the computations of the summary statistics in
#' case of pseudo-marginal methods.
#' @param gamma Tuning parameter for the adaptation of RAM algorithm. Must be
#' between 0 and 1.
#' @param target_acceptance Target acceptance rate for MCMC. Defaults to 0.234.
#' Must be between 0 and 1.
#' @param S Matrix defining the initial value for the lower triangular matrix
#' of the RAM algorithm, so that the covariance matrix of the Gaussian proposal
#' distribution is \eqn{SS'}. Note that for some parameters
#' (currently the standard deviation, dispersion, and autoregressive parameters
#' of the BSM and AR(1) models) the sampling is done in unconstrained parameter
#' space, i.e. internal_theta = log(theta) (and logit(rho) or AR coefficient).
#' @param end_adaptive_phase Logical, if \code{TRUE}, S is held fixed after the
#' burnin period. Default is \code{FALSE}.
#' @param threads Number of threads for state simulation. Positive integer
#' (default is 1).
#' Note that parallel computing is only used in the post-correction phase of
#' IS-MCMC and when sampling the states in case of (approximate) Gaussian
#' models.
#' @param seed Seed for the C++ RNG (positive integer).
#' @param local_approx If \code{TRUE} (default), Gaussian approximation
#' needed for some of the methods is performed at each iteration.
#' If \code{FALSE}, approximation is updated only once at the start of the
#' MCMC using the initial model.
#' @param max_iter Maximum number of iterations used in Gaussian approximation,
#' as a positive integer.
#' Default is 100 (although typically only few iterations are needed).
#' @param conv_tol Positive tolerance parameter used in Gaussian approximation.
#' @param particles A positive integer defining the number of state samples per
#' MCMC iteration for models other than linear-Gaussian models.
#' Ignored if \code{mcmc_type} is \code{"approx"} or \code{"ekf"}. Suitable
#' values depend on the model, the data, \code{mcmc_type} and
#' \code{sampling_method}. While large values provide more
#' accurate estimates, the run time also increases with respect to to the
#' number of particles, so it is generally a good idea to test the run time
#' firstwith a small number of particles, e.g., less than 100.
#' @param mcmc_type What type of MCMC algorithm should be used for models other
#' than linear-Gaussian models? Possible choices are
#' \code{"pm"} for pseudo-marginal MCMC,
#' \code{"da"} for delayed acceptance version of PMCMC ,
#' \code{"approx"} for approximate inference based on the Gaussian
#' approximation of the model,
#' \code{"ekf"} for approximate inference using extended Kalman filter
#' (for \code{ssm_nlg}),
#' or one of the three importance sampling type weighting schemes:
#' \code{"is3"} for simple importance sampling (weight is computed for each
#' MCMC iteration independently),
#' \code{"is2"} for jump chain importance sampling type weighting (default), or
#' \code{"is1"} for importance sampling type weighting where the number of
#' particles used for
#' weight computations is proportional to the length of the jump chain block.
#' @param sampling_method Method for state sampling when for models other than
#' linear-Gaussian models. If \code{"psi"}, \eqn{\psi}-APF is used (default).
#' If \code{"spdk"}, non-sequential importance sampling
#' based on Gaussian approximation is used. If \code{"bsf"}, bootstrap filter
#' is used. If \code{"ekf"}, particle filter based on EKF-proposals are used
#' (only for \code{ssm_nlg} models).
#' @param iekf_iter Non-negative integer. The default zero corresponds to
#' normal EKF, whereas \code{iekf_iter > 0} corresponds to iterated EKF
#' with \code{iekf_iter} iterations. Used only for models of class
#' \code{ssm_nlg}.
#' @param L_c,L_f For \code{ssm_sde} models, Positive integer values defining
#' the discretization levels for first and second stages (defined as 2^L).
#' For pseudo-marginal methods (\code{"pm"}), maximum of these is used.
#' @param verbose If \code{TRUE}, prints a progress bar to the console. If
#' missing, defined by \code{rlang::is_interactive}.
#' Set to \code{FALSE} if number of iterations is less than 50.
#' @param ... Ignored.
#' @return An object of class \code{mcmc_output}.
#' @export
#' @srrstats {G2.3, G2.3a, G2.3b} match.arg and tolower used where applicable.
#' @srrstats {BS1.0, BS1.1, BS1.2, BS1.2a, BS1.2b}
#' @srrstats {BS2.6}
#' @srrstats {BS2.7} Illustrated in the examples.
#' @srrstats {BS2.7, BS1.3, BS1.3a, BS1.3b, BS2.8} Explained in docs.
#' @srrstats {BS2.9} The argument 'seed' is set to random value if not
#' specified by the user.
#' @srrstats {BS5.0, BS5.1, BS5.2} Starting values are integrated into the
#' input model, whereas some metadata (like the class of input model and seed)
#' is returned by run_mcmc.
#' @srrstats {BS2.12, BS2.13} There is a progress bar which can be switched off
#' with \code{verbose = FALSE}.
#' @srrstats {BS1.2c} Examples on defining priors.
#' @srrstats {BS2.14} No warnings are issues during MCMC.
#' @rdname run_mcmc
#' @references
#' Vihola M (2012). Robust adaptive Metropolis algorithm with
#' coerced acceptance rate. Statistics and Computing, 22(5), p 997-1008.
#' https://doi.org/10.1007/s11222-011-9269-5
#'
#' Vihola, M, Helske, J, Franks, J (2020). Importance sampling type
#' estimators based on approximate marginal Markov chain Monte Carlo.
#' Scand J Statist. 1-38. https://doi.org/10.1111/sjos.12492
#'
#' Helske J, Vihola M (2021). bssm: Bayesian Inference of Non-linear and
#' Non-Gaussian State Space Models in R. The R Journal (2021) 13:2, 578-589.
#' https://doi.org/10.32614/RJ-2021-103
#'
run_mcmc <- function(model, ...) {
  UseMethod("run_mcmc", model)
}
#' @method run_mcmc lineargaussian
#' @rdname run_mcmc
#' @export
#' @examples
#' model <- ar1_lg(LakeHuron, rho = uniform(0.5,-1,1),
#'   sigma = halfnormal(1, 10), mu = normal(500, 500, 500),
#'   sd_y = halfnormal(1, 10))
#'
#' mcmc_results <- run_mcmc(model, iter = 2e4)
#' summary(mcmc_results, return_se = TRUE)
#'
#' sumr <- summary(mcmc_results, variable = "states")
#' library("ggplot2")
#' ggplot(sumr, aes(time, Mean)) +
#'   geom_ribbon(aes(ymin = `2.5%`, ymax = `97.5%`), alpha = 0.25) +
#'   geom_line() + theme_bw() +
#'   geom_point(data = data.frame(Mean = LakeHuron, time = time(LakeHuron)),
#'     col = 2)
#'
#' # Continue from the previous run
#' model$theta[] <- mcmc_results$theta[nrow(mcmc_results$theta), ]
#' run_more <- run_mcmc(model, S = mcmc_results$S, iter = 1000, burnin = 0)
#'
run_mcmc.lineargaussian <- function(model, iter, output_type = "full",
  burnin = floor(iter / 2), thin = 1, gamma = 2 / 3,
  target_acceptance = 0.234, S, end_adaptive_phase = FALSE, threads = 1,
  seed = sample(.Machine$integer.max, size = 1),
  verbose, ...) {

  check_missingness(model)

  if (!test_flag(end_adaptive_phase))
    stop("Argument 'end_adaptive_phase' should be TRUE or FALSE. ")
  seed <- check_intmax(seed, "seed", FALSE, max = .Machine$integer.max)

  threads <- check_intmax(threads, "threads")
  thin <- check_intmax(thin, "thin", max = 100)
  iter <- check_intmax(iter, "iter", max = 1e12)
  burnin <- check_intmax(burnin, "burnin", positive = FALSE, max = 1e12)
  if(burnin > iter) stop("Argument 'burnin' should be smaller than 'iter'.")

  if (missing(verbose)) {
    verbose <- is_interactive()
  } else {
  if (!test_flag(verbose))
    stop("Argument 'verbose' should be TRUE or FALSE. ")
  }
  if (iter < 50) verbose <- FALSE

  if (length(model$theta) == 0)
    stop("No unknown parameters ('model$theta' has length of zero).")
  a <- proc.time()

  check_prop(target_acceptance)
  check_prop(gamma, "gamma")
  output_type <- pmatch(tolower(output_type), c("full", "summary", "theta"))

  if (inherits(model, "bsm_lg")) {
    names_ind <- !model$fixed & c(TRUE, TRUE, model$slope, model$seasonal)
    transformed <-
      c("sd_y", "sd_level", "sd_slope", "sd_seasonal")[names_ind]
    model$theta[transformed] <- log(pmax(0.001, model$theta[transformed]))
  } else {
    if (inherits(model, "ar1_lg")) {
      transformed <- c("sigma", "sd_y")
      model$theta[transformed] <- log(pmax(0.001, model$theta[transformed]))
    }
  }

  if (missing(S)) {
    S <- diag(0.1 * pmax(0.1, abs(model$theta)), length(model$theta))
  }
  if(output_type == "full") {
    nsamples <-
      ifelse(!is.null(nrow(model$y)), nrow(model$y), length(model$y)) *
      length(model$a1) * (iter - burnin) / thin * target_acceptance
    if (nsamples > 1e12) {
      warning(paste("Number of state samples to be stored is approximately",
        nsamples, "you might run out of memory."))
    }
  }
  out <- gaussian_mcmc(model, output_type,
    iter, burnin, thin, gamma, target_acceptance, S, seed,
    end_adaptive_phase, threads, model_type(model), verbose)

  if (output_type == 1) {
    colnames(out$alpha) <- names(model$a1)
  } else {
    if (output_type == 2) {
      colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <-
        names(model$a1)
      out$alphahat <- ts(out$alphahat, start = start(model$y),
        frequency = frequency(model$y))
    }
  }

  colnames(out$theta) <- rownames(out$S) <- colnames(out$S) <-
    names(model$theta)

  if (inherits(model, "bsm_lg")) {
    out$theta[, transformed] <- exp(out$theta[, transformed])
  } else {
    if (inherits(model, "ar1_lg")) {
      out$theta[, transformed] <- exp(out$theta[, transformed])
    }
  }
  out$call <- match.call()
  out$seed <- seed
  out$iter <- iter
  out$burnin <- burnin
  out$thin <- thin
  out$mcmc_type <- "gaussian_mcmc"
  out$output_type <- output_type
  dim(out$counts) <- NULL
  out$time <- proc.time() - a
  class(out) <- "mcmc_output"
  attr(out, "model_type") <- class(model)[1]
  attr(out, "ts") <-
    list(start = start(model$y), end = end(model$y),
      frequency = frequency(model$y))
  out
}
#' @method run_mcmc nongaussian
#' @rdname run_mcmc
#' @export
#' @examples
#' set.seed(1)
#' n <- 50
#' slope <- cumsum(c(0, rnorm(n - 1, sd = 0.001)))
#' level <- cumsum(slope + c(0, rnorm(n - 1, sd = 0.2)))
#' y <- rpois(n, exp(level))
#' poisson_model <- bsm_ng(y,
#'   sd_level = halfnormal(0.01, 1),
#'   sd_slope = halfnormal(0.01, 0.1),
#'   P1 = diag(c(10, 0.1)), distribution = "poisson")
#'
#' # Note small number of iterations for CRAN checks
#' mcmc_out <- run_mcmc(poisson_model, iter = 1000, particles = 10,
#'   mcmc_type = "da")
#' summary(mcmc_out, what = "theta", return_se = TRUE)
#'
#' set.seed(123)
#' n <- 50
#' sd_level <- 0.1
#' drift <- 0.01
#' beta <- -0.9
#' phi <- 5
#'
#' level <- cumsum(c(5, drift + rnorm(n - 1, sd = sd_level)))
#' x <- 3 + (1:n) * drift + sin(1:n + runif(n, -1, 1))
#' y <- rnbinom(n, size = phi, mu = exp(beta * x + level))
#'
#' model <- bsm_ng(y, xreg = x,
#'   beta = normal(0, 0, 10),
#'   phi = halfnormal(1, 10),
#'   sd_level = halfnormal(0.1, 1),
#'   sd_slope = halfnormal(0.01, 0.1),
#'   a1 = c(0, 0), P1 = diag(c(10, 0.1)^2),
#'   distribution = "negative binomial")
#'
#' # run IS-MCMC
#' # Note small number of iterations for CRAN checks
#' fit <- run_mcmc(model, iter = 4000,
#'   particles = 10, mcmc_type = "is2", seed = 1)
#'
#' # extract states
#' d_states <- as.data.frame(fit, variable = "states", time = 1:n)
#'
#' library("dplyr")
#' library("ggplot2")
#'
#'  # compute summary statistics
#' level_sumr <- d_states %>%
#'   filter(variable == "level") %>%
#'   group_by(time) %>%
#'   summarise(mean = diagis::weighted_mean(value, weight),
#'     lwr = diagis::weighted_quantile(value, weight,
#'       0.025),
#'     upr = diagis::weighted_quantile(value, weight,
#'       0.975))
#'
#' # visualize
#' level_sumr %>% ggplot(aes(x = time, y = mean)) +
#'   geom_line() +
#'   geom_line(aes(y = lwr), linetype = "dashed", na.rm = TRUE) +
#'   geom_line(aes(y = upr), linetype = "dashed", na.rm = TRUE) +
#'   theme_bw() +
#'   theme(legend.title = element_blank()) +
#'   xlab("Time") + ylab("Level")
#'
#' # theta
#' d_theta <- as.data.frame(fit, variable = "theta")
#' ggplot(d_theta, aes(x = value)) +
#'  geom_density(aes(weight = weight), adjust = 2, fill = "#92f0a8") +
#'  facet_wrap(~ variable, scales = "free") +
#'  theme_bw()
#'
#'
#' # Bivariate Poisson model:
#'
#' set.seed(1)
#' x <- cumsum(c(3, rnorm(19, sd = 0.5)))
#' y <- cbind(
#'   rpois(20, exp(x)),
#'   rpois(20, exp(x)))
#'
#' prior_fn <- function(theta) {
#'   # half-normal prior using transformation
#'   dnorm(exp(theta), 0, 1, log = TRUE) + theta # plus jacobian term
#' }
#'
#' update_fn <- function(theta) {
#'   list(R = array(exp(theta), c(1, 1, 1)))
#' }
#'
#' model <- ssm_mng(y = y, Z = matrix(1,2,1), T = 1,
#'   R = 0.1, P1 = 1, distribution = "poisson",
#'   init_theta = log(0.1),
#'   prior_fn = prior_fn, update_fn = update_fn)
#'
#' # Note small number of iterations for CRAN checks
#' out <- run_mcmc(model, iter = 4000, mcmc_type = "approx")
#'
#' sumr <- as.data.frame(out, variable = "states") %>%
#'   group_by(time) %>% mutate(value = exp(value)) %>%
#'   summarise(mean = mean(value),
#'     ymin = quantile(value, 0.05), ymax = quantile(value, 0.95))
#' ggplot(sumr, aes(time, mean)) +
#' geom_ribbon(aes(ymin = ymin, ymax = ymax),alpha = 0.25) +
#' geom_line() +
#' geom_line(data = data.frame(mean = y[, 1], time = 1:20),
#'   colour = "tomato") +
#' geom_line(data = data.frame(mean = y[, 2], time = 1:20),
#'   colour = "tomato") +
#' theme_bw()
#'
run_mcmc.nongaussian <- function(model, iter, particles, output_type = "full",
  mcmc_type = "is2", sampling_method = "psi", burnin = floor(iter / 2),
  thin = 1, gamma = 2 / 3, target_acceptance = 0.234, S,
  end_adaptive_phase = FALSE, local_approx  = TRUE, threads = 1,
  seed = sample(.Machine$integer.max, size = 1), max_iter = 100,
  conv_tol = 1e-8, verbose, ...) {

  check_missingness(model)

  if (!test_flag(end_adaptive_phase))
    stop("Argument 'end_adaptive_phase' should be TRUE or FALSE. ")

  seed <- check_intmax(seed, "seed", FALSE, max = .Machine$integer.max)

  if (missing(particles)) {
    nsim <- eval(match.call(expand.dots = TRUE)$nsim)
    if (!is.null(nsim)) {
      warning(paste("Argument `nsim` is deprecated. Use argument `particles`",
        "instead.", sep = " "))
      particles <- nsim
      particles <- check_intmax(particles, "particles")
    }
  } else {
    particles <- check_intmax(particles, "particles")
  }

  threads <- check_intmax(threads, "threads")
  model$max_iter <- check_intmax(max_iter, "max_iter", positive = FALSE)
  model$conv_tol <- check_positive_real(conv_tol, "conv_tol")
  thin <- check_intmax(thin, "thin", max = 100)
  iter <- check_intmax(iter, "iter", max = 1e12)
  burnin <- check_intmax(burnin, "burnin", positive = FALSE, max = 1e12)
  if(burnin > iter) stop("Argument 'burnin' should be smaller than 'iter'.")

  if (missing(verbose)) {
    verbose <- is_interactive()
  } else {
    if (!test_flag(verbose))
      stop("Argument 'verbose' should be TRUE or FALSE. ")
  }
  if (iter < 50) verbose <- FALSE

  if (!test_flag(local_approx))  {
    stop("Argument 'local_approx' should be TRUE or FALSE. ")
  } else model$local_approx <- local_approx

  if (length(model$theta) == 0)
    stop("No unknown parameters ('model$theta' has length of zero).")
  a <- proc.time()
  check_prop(target_acceptance)
  check_prop(gamma, "gamma")

  output_type <- pmatch(tolower(output_type), c("full", "summary", "theta"))
  mcmc_type <- match.arg(tolower(mcmc_type),
    c("pm", "da", paste0("is", 1:3), "approx"))
  if (mcmc_type == "approx") particles <- 0
  if (particles < 2 && mcmc_type != "approx")
    stop(paste("Number of state samples less than 2, use 'mcmc_type' 'approx'",
      "instead.", sep = " "))

  sampling_method <-
    pmatch(match.arg(tolower(sampling_method), c("psi", "bsf", "spdk")),
      c("psi", "bsf", "spdk"))

  dists <-
    c("svm", "poisson", "binomial", "negative binomial", "gamma", "gaussian")
  model$distribution <-
    pmatch(model$distribution, dists, duplicates.ok = TRUE) - 1

  if(output_type == "full") {
    nsamples <-
      ifelse(!is.null(nrow(model$y)), nrow(model$y), length(model$y)) *
      length(model$a1) * (iter - burnin) / thin * target_acceptance
    if (nsamples > 1e12) {
      warning(paste("Number of state samples to be stored is approximately",
        nsamples, "you might run out of memory."))
    }
  }

  if (inherits(model, "bsm_ng")) {

    names_ind <-
      c(!model$fixed & c(TRUE, model$slope, model$seasonal), model$noise)

    transformed <- c(
      c("sd_level", "sd_slope", "sd_seasonal", "sd_noise")[names_ind],
      if (dists[model$distribution + 1] %in% dists[4:5]) "phi")

    model$theta[transformed] <- log(pmax(0.001, model$theta[transformed]))
  } else {
    if (inherits(model, "ar1_ng")) {

      transformed <- c("sigma",
        if (dists[model$distribution + 1] %in% dists[4:5]) "phi")

      model$theta[transformed] <- log(pmax(0.001, model$theta[transformed]))
    }
  }

  if (missing(S)) {
    S <- diag(0.1 * pmax(0.1, abs(model$theta)), length(model$theta))
  }

  switch(mcmc_type,
    "da" = {
      out <- nongaussian_da_mcmc(model,
        output_type, particles, iter, burnin, thin, gamma, target_acceptance, S,
        seed, end_adaptive_phase, threads,
        sampling_method, model_type(model), verbose)
    },
    "pm" = {
      out <- nongaussian_pm_mcmc(model, output_type,
        particles, iter, burnin, thin, gamma, target_acceptance, S,
        seed, end_adaptive_phase, threads,
        sampling_method, model_type(model), verbose)
    },
    "is1" =,
    "is2" =,
    "is3" = {
      out <- nongaussian_is_mcmc(model, output_type,
        particles, iter, burnin, thin, gamma, target_acceptance, S,
        seed, end_adaptive_phase, threads,
        sampling_method,
        pmatch(mcmc_type, paste0("is", 1:3)), model_type(model), FALSE,
        verbose)
    },
    "approx" = {
      out <- nongaussian_is_mcmc(model, output_type,
        particles, iter, burnin, thin, gamma, target_acceptance, S,
        seed, end_adaptive_phase, threads,
        sampling_method, 2, model_type(model), TRUE, verbose)
    })
  if (output_type == 1) {
    colnames(out$alpha) <- names(model$a1)
  } else {
    if (output_type == 2) {
      colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <-
        names(model$a1)
      out$alphahat <- ts(out$alphahat, start = start(model$y),
        frequency = frequency(model$y))
    }
  }

  colnames(out$theta) <- rownames(out$S) <- colnames(out$S) <-
    names(model$theta)
  if (inherits(model, "bsm_ng")) {
    out$theta[, transformed] <- exp(out$theta[, transformed])
  } else {
    if (inherits(model, "ar1_ng")) {
      out$theta[, transformed] <- exp(out$theta[, transformed])
    }
  }
  out$iter <- iter
  out$burnin <- burnin
  out$thin <- thin
  out$mcmc_type <- mcmc_type
  out$output_type <- output_type
  out$call <- match.call()
  out$seed <- seed
  dim(out$counts) <- NULL
  out$time <- proc.time() - a
  class(out) <- "mcmc_output"
  attr(out, "model_type") <- class(model)[1]
  attr(out, "ts") <-
    list(start = start(model$y), end = end(model$y),
      frequency = frequency(model$y))
  out
}
#' @method run_mcmc ssm_nlg
#' @rdname run_mcmc
#' @export
run_mcmc.ssm_nlg <-  function(model, iter, particles, output_type = "full",
  mcmc_type = "is2", sampling_method = "bsf",
  burnin = floor(iter / 2), thin = 1,
  gamma = 2 / 3, target_acceptance = 0.234, S, end_adaptive_phase = FALSE,
  threads = 1, seed = sample(.Machine$integer.max, size = 1), max_iter = 100,
  conv_tol = 1e-8, iekf_iter = 0, verbose, ...) {

  check_missingness(model)

  if (!test_flag(end_adaptive_phase))
    stop("Argument 'end_adaptive_phase' should be TRUE or FALSE. ")

  seed <- check_intmax(seed, "seed", FALSE, max = .Machine$integer.max)

  if (missing(particles)) {
    nsim <- eval(match.call(expand.dots = TRUE)$nsim)
    if (!is.null(nsim)) {
      warning(paste("Argument `nsim` is deprecated. Use argument `particles`",
        "instead.", sep = " "))
      particles <- nsim
      particles <- check_intmax(particles, "particles")
    }
  } else {
    particles <- check_intmax(particles, "particles")
  }

  threads <- check_intmax(threads, "threads")
  max_iter <- check_intmax(max_iter, "max_iter", positive = FALSE)
  conv_tol <- check_positive_real(conv_tol, "conv_tol")
  thin <- check_intmax(thin, "thin", max = 100)
  iter <- check_intmax(iter, "iter", max = 1e12)
  burnin <- check_intmax(burnin, "burnin", positive = FALSE, max = 1e12)
  iekf_iter <- check_intmax(iekf_iter, "iekf_iter", positive = FALSE)
  if(burnin > iter) stop("Argument 'burnin' should be smaller than 'iter'.")

  if (missing(verbose)) {
    verbose <- is_interactive()
  } else {
    if (!test_flag(verbose))
      stop("Argument 'verbose' should be TRUE or FALSE. ")
  }
  if (iter < 50) verbose <- FALSE

  if (length(model$theta) == 0)
    stop("No unknown parameters ('model$theta' has length of zero).")
  a <- proc.time()

  check_prop(target_acceptance)
  check_prop(gamma, "gamma")

  output_type <- pmatch(tolower(output_type), c("full", "summary", "theta"))
  mcmc_type <- match.arg(tolower(mcmc_type), c("pm", "da", paste0("is", 1:3),
    "ekf", "approx"))
  if (mcmc_type %in% c("ekf", "approx")) particles <- 0
  sampling_method <- pmatch(match.arg(tolower(sampling_method),
    c("psi", "bsf", "ekf")), c("psi", "bsf", NA, "ekf"))

  if (missing(S)) {
    S <- diag(0.1 * pmax(0.1, abs(model$theta)), length(model$theta))
  }

  if (particles < 2 && !(mcmc_type %in% c("ekf", "approx")))
    stop(paste("Number of state samples less than 2, use 'mcmc_type'",
      "'approx' or 'ekf' instead.", sep = " "))

  if(output_type == "full") {
    nsamples <-
      ifelse(!is.null(nrow(model$y)), nrow(model$y), length(model$y)) *
      model$n_states * (iter - burnin) / thin * target_acceptance
    if (nsamples > 1e12) {
      warning(paste("Number of state samples to be stored is approximately",
        nsamples, "you might run out of memory."))
    }
  }

  out <- switch(mcmc_type,
    "da" = {
      nonlinear_da_mcmc(t(model$y), model$Z, model$H, model$T,
        model$R, model$Z_gn, model$T_gn, model$a1, model$P1,
        model$theta, model$log_prior_pdf, model$known_params,
        model$known_tv_params, as.integer(model$time_varying),
        model$n_states, model$n_etas, seed,
        particles, iter, burnin, thin, gamma, target_acceptance, S,
        end_adaptive_phase, threads, max_iter, conv_tol,
        sampling_method, iekf_iter, output_type, verbose)
    },
    "pm" = {
      nonlinear_pm_mcmc(t(model$y), model$Z, model$H, model$T,
        model$R, model$Z_gn, model$T_gn, model$a1, model$P1,
        model$theta, model$log_prior_pdf, model$known_params,
        model$known_tv_params, as.integer(model$time_varying),
        model$n_states, model$n_etas, seed,
        particles, iter, burnin, thin, gamma, target_acceptance, S,
        end_adaptive_phase, threads, max_iter, conv_tol,
        sampling_method, iekf_iter, output_type, verbose)
    },
    "is1" =,
    "is2" =,
    "is3" = {
      if (sampling_method == 4)
        stop("IS-MCMC with extended particle filter is (not yet) supported.")
      nonlinear_is_mcmc(t(model$y), model$Z, model$H, model$T,
        model$R, model$Z_gn, model$T_gn, model$a1, model$P1,
        model$theta, model$log_prior_pdf, model$known_params,
        model$known_tv_params, as.integer(model$time_varying),
        model$n_states, model$n_etas, seed,
        particles, iter, burnin, thin, gamma, target_acceptance, S,
        end_adaptive_phase, threads, pmatch(mcmc_type, paste0("is", 1:3)),
        sampling_method, max_iter, conv_tol, iekf_iter,
        output_type, FALSE, verbose)
    },
    "ekf" = {
      nonlinear_ekf_mcmc(t(model$y), model$Z, model$H, model$T,
        model$R, model$Z_gn, model$T_gn, model$a1, model$P1,
        model$theta, model$log_prior_pdf, model$known_params,
        model$known_tv_params, as.integer(model$time_varying),
        model$n_states, model$n_etas, seed,
        iter, burnin, thin, gamma, target_acceptance, S,
        end_adaptive_phase,  threads, iekf_iter, output_type, verbose)
    },
    "approx" = {
      nonlinear_is_mcmc(t(model$y), model$Z, model$H, model$T,
        model$R, model$Z_gn, model$T_gn, model$a1, model$P1,
        model$theta, model$log_prior_pdf, model$known_params,
        model$known_tv_params, as.integer(model$time_varying),
        model$n_states, model$n_etas, seed,
        particles, iter, burnin, thin, gamma, target_acceptance, S,
        end_adaptive_phase, threads, 2,
        sampling_method, max_iter, conv_tol,
        iekf_iter, output_type, TRUE, verbose)
    }
  )
  if (output_type == 1) {
    colnames(out$alpha) <- model$state_names
  } else {
    if (output_type == 2) {
      colnames(out$alphahat) <- colnames(out$Vt) <- rownames(out$Vt) <-
        model$state_names
      out$alphahat <- ts(out$alphahat, start = start(model$y),
        frequency = frequency(model$y))
    }
  }


  colnames(out$theta) <- rownames(out$S) <- colnames(out$S) <-
    names(model$theta)

  out$iter <- iter
  out$burnin <- burnin
  out$thin <- thin
  out$mcmc_type <- mcmc_type
  out$output_type <- output_type
  out$call <- match.call()
  out$seed <- seed
  dim(out$counts) <- NULL
  out$time <- proc.time() - a
  class(out) <- "mcmc_output"
  attr(out, "model_type") <- "ssm_nlg"
  attr(out, "ts") <-
    list(start = start(model$y), end = end(model$y),
      frequency = frequency(model$y))
  out
}
#' @method run_mcmc ssm_sde
#' @rdname run_mcmc
#' @export
#' @references
#' Vihola, M, Helske, J, Franks, J. Importance sampling type estimators based
#' on approximate marginal Markov chain Monte Carlo.
#' Scand J Statist. 2020; 1-38. https://doi.org/10.1111/sjos.12492
run_mcmc.ssm_sde <-  function(model, iter, particles, output_type = "full",
  mcmc_type = "is2", L_c, L_f,
  burnin = floor(iter/2), thin = 1,
  gamma = 2/3, target_acceptance = 0.234, S, end_adaptive_phase = FALSE,
  threads = 1, seed = sample(.Machine$integer.max, size = 1), verbose,
  ...) {

  check_missingness(model)

  if (any(c(model$drift, model$diffusion, model$ddiffusion, model$prior_pdf,
    model$obs_pdf) %in% c("<pointer: (nil)>", "<pointer: 0x0>"))) {
    stop(paste("NULL pointer detected, please recompile the pointer file",
      "and reconstruct the model.", sep = " "))
  }

  if (!test_flag(end_adaptive_phase))
    stop("Argument 'end_adaptive_phase' should be TRUE or FALSE. ")

  seed <- check_intmax(seed, "seed", FALSE, max = .Machine$integer.max)

  if (missing(particles)) {
    nsim <- eval(match.call(expand.dots = TRUE)$nsim)
    if (!is.null(nsim)) {
      warning(paste("Argument `nsim` is deprecated. Use argument `particles`",
        "instead.", sep = " "))
      particles <- nsim
    }
  }
  particles <- check_intmax(particles, "particles")
  threads <- check_intmax(threads, "threads")
  thin <- check_intmax(thin, "thin", max = 100)
  iter <- check_intmax(iter, "iter", max = 1e12)
  burnin <- check_intmax(burnin, "burnin", positive = FALSE, max = 1e12)
  if(burnin > iter) stop("Argument 'burnin' should be smaller than 'iter'.")

  if (missing(verbose)) {
    verbose <- is_interactive()
  } else {
    if (!test_flag(verbose))
      stop("Argument 'verbose' should be TRUE or FALSE. ")
  }
  if (iter < 50) verbose <- FALSE

  if (length(model$theta) == 0)
    stop("No unknown parameters ('model$theta' has length of zero).")
  a <- proc.time()

  check_prop(target_acceptance)
  check_prop(gamma, "gamma")

  output_type <- pmatch(tolower(output_type), c("full", "summary", "theta"))
  mcmc_type <- match.arg(tolower(mcmc_type), c("pm", "da", paste0("is", 1:3)))

  if (missing(S)) {
    S <- diag(0.1 * pmax(0.1, abs(model$theta)), length(model$theta))
  }

  if (mcmc_type != "pm") {
    if (L_f <= L_c) stop("L_f should be larger than L_c.")
    if (L_c < 1) stop("L_c should be at least 1")
  } else {
    if (missing(L_c)) L_c <- 0
    if (missing(L_f)) L_f <- 0
    L <- max(L_c, L_f)
    if (L <= 0) stop("L should be positive.")
  }
  if(output_type == "full") {
    nsamples <- length(model$y) * (iter - burnin) / thin * target_acceptance
    if (nsamples > 1e12) {
      warning(paste("Number of state samples to be stored is approximately",
        nsamples, "you might run out of memory."))
    }
  }

  out <- switch(mcmc_type,
    "da" = {
      out <- sde_da_mcmc(model$y, model$x0, model$positive,
        model$drift, model$diffusion, model$ddiffusion,
        model$prior_pdf, model$obs_pdf, model$theta,
        particles, L_c, L_f, seed,
        iter, burnin, thin, gamma, target_acceptance, S,
        end_adaptive_phase, output_type, verbose)
    },
    "pm" = {

      out <- sde_pm_mcmc(model$y, model$x0, model$positive,
        model$drift, model$diffusion, model$ddiffusion,
        model$prior_pdf, model$obs_pdf, model$theta,
        particles, L, seed,
        iter, burnin, thin, gamma, target_acceptance, S,
        end_adaptive_phase, output_type, verbose)
    },
    "is1" =,
    "is2" =,
    "is3" = {
      out <- sde_is_mcmc(model$y, model$x0, model$positive,
        model$drift, model$diffusion, model$ddiffusion,
        model$prior_pdf, model$obs_pdf, model$theta,
        particles, L_c, L_f, seed,
        iter, burnin, thin, gamma, target_acceptance, S,
        end_adaptive_phase, pmatch(mcmc_type, paste0("is", 1:3)),
        threads, output_type, verbose)
    })

  colnames(out$alpha) <- model$state_names

  colnames(out$theta) <- rownames(out$S) <- colnames(out$S) <-
    names(model$theta)

  out$iter <- iter
  out$burnin <- burnin
  out$thin <- thin
  out$mcmc_type <- mcmc_type
  out$output_type <- output_type
  out$call <- match.call()
  out$seed <- seed
  dim(out$counts) <- NULL
  out$time <- proc.time() - a
  class(out) <- "mcmc_output"
  attr(out, "model_type") <- "ssm_sde"
  attr(out, "ts") <-
    list(start = start(model$y), end = end(model$y),
      frequency = frequency(model$y))
  out
}
