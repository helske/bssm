#' Integrated Autocorrelation Time
#'
#' Estimates the integrated autocorrelation time (IACT) based on Sokal (1997).
#' Note that the estimator is not particularly good for very short series x 
#' (say < 100), but that is not very practical for MCMC applications anyway.
#'
#' @param x A numeric vector.
#' @return A single numeric value of IACT estimate.
#' @references
#' Sokal A. (1997) Monte Carlo Methods in Statistical Mechanics: Foundations 
#' and New Algorithms. 
#' In: DeWitt-Morette C., Cartier P., Folacci A. (eds) Functional Integration. 
#' NATO ASI Series (Series B: Physics), vol 361. Springer, Boston, MA. 
#' https://doi.org/10.1007/978-1-4899-0319-8_6
#' @export
#' @srrstats {BS5.3, BS5.5}
#' @examples
#' set.seed(1)
#' n <- 1000
#' x <- numeric(n)
#' phi <- 0.8
#' for(t in 2:n) x[t] <- phi * x[t-1] + rnorm(1)
#' iact(x)
iact <- function(x) {
  
  if (!test_numeric(x))
    stop("Argument 'x' should be a numeric vector. ")
  
  IACT((x - mean(x)) / sd(x))
}

#' Asymptotic Variance of IS-type Estimators
#'
#' The asymptotic variance MCMCSE^2 is based on Corollary 1 
#' of Vihola et al. (2020) from weighted samples from IS-MCMC. The default 
#' method is based on the integrated autocorrelation time (IACT) by Sokal 
#' (1997) which seem to work well for reasonable problems, but it is also 
#' possible to use the Geyer's method as implemented in \code{ess_mean} of the 
#' \code{posterior} package. 
#' 
#' @importFrom posterior ess_mean
#' @importFrom checkmate test_numeric
#' @param x A numeric vector of samples.
#' @param w A numeric vector of weights. If missing, set to 1 (i.e. no 
#' weighting is assumed).
#' @param method Method for computing IACT. Default is \code{"sokal"},
#' other option \code{"geyer"}.
#' @return A single numeric value of asymptotic variance estimate.
#' @references
#' Vihola M, Helske J, Franks J. (2020). Importance sampling type estimators 
#' based on approximate marginal Markov chain Monte Carlo. 
#' Scand J Statist. 1-38. https://doi.org/10.1111/sjos.12492
#' 
#' Sokal A. (1997). Monte Carlo Methods in Statistical Mechanics: Foundations 
#' and New Algorithms. 
#' In: DeWitt-Morette C, Cartier P, Folacci A (eds) Functional Integration. 
#' NATO ASI Series (Series B: Physics), vol 361. Springer, Boston, MA. 
#' https://doi.org/10.1007/978-1-4899-0319-8_6
#' 
#' Gelman, A, Carlin J B, Stern H S, Dunson, D B, Vehtari A, Rubin D B. (2013). 
#' Bayesian Data Analysis, Third Edition. Chapman and Hall/CRC.
#' 
#' Vehtari A, Gelman A, Simpson D, Carpenter B, Bürkner P-C. (2021). 
#' Rank-normalization, folding, and localization: An improved Rhat for 
#' assessing convergence of MCMC. Bayesian analysis, 16(2):667-718. 
#' https://doi.org/10.1214/20-BA1221
#' @export
#' @srrstats {BS5.3, BS5.5}
#' @examples
#' set.seed(1)
#' n <- 1e4 
#' x <- numeric(n)
#' phi <- 0.7
#' for(t in 2:n) x[t] <- phi * x[t-1] + rnorm(1)
#' w <- rexp(n, 0.5 * exp(0.001 * x^2))
#' # different methods:
#' asymptotic_var(x, w, method = "sokal")
#' asymptotic_var(x, w, method = "geyer")
#' 
#' data("negbin_model")
#' # can be obtained directly with summary method
#' d <- suppressWarnings(as_draws(negbin_model))
#' sqrt(asymptotic_var(d$sd_level, d$weight))
#' 
asymptotic_var <- function(x, w, method = "sokal") {
  
  method <- match.arg(tolower(method), c("sokal", "geyer"))
  if (!test_numeric(x) & !is.null(class(x)))
    stop("Argument 'x' should be a numeric vector. ")
  if (missing(w)) {
    w <- rep(1, length(x))
  } else {
    if (!test_numeric(w))
      stop("Argument 'w' should be a numeric vector. ")
    if(any(w < 0) | any(!is.finite(w)))
      stop("Nonfinite or negative weights in 'w'.")
    if (!any(w > 0)) {
      stop("No positive weights in 'w'.")
    }
  }
  estimate_c <- mean(w)
  estimate_mean <- weighted_mean(x, w)
  z <- w * (x - estimate_mean)
  switch(method,
    sokal = (var(z) * iact(z) / estimate_c^2) / length(z),
    # ESS(z) = n / IACT(z)
    geyer = var(z) / ess_mean(z) / estimate_c^2)
}

#' Effective Sample Size for IS-type Estimators
#'
#' Computes the effective sample size (ESS) based on weighted posterior 
#' samples.
#' 
#' The asymptotic variance MCMCSE^2 is based on Corollary 1 of 
#' Vihola et al. (2020) which is used to compute an estimate for the ESS
#' using the identity ESS(x) = var(x) / MCMCSE^2 where var(x) is the 
#' posterior variance of x assuming independent samples. 
#' 
#' @param x A numeric vector of samples.
#' @param w A numeric vector of weights. If missing, set to 1 (i.e. no 
#' weighting is assumed).
#' @param method Method for computing the ESS. Default is \code{"sokal"}, other 
#' option are \code{"geyer"} (see also \code{asymptotic_var}).
#' @references
#' Vihola, M, Helske, J, Franks, J. (2020). Importance sampling type estimators 
#' based on approximate marginal Markov chain Monte Carlo. 
#' Scand J Statist. 1-38. https://doi.org/10.1111/sjos.12492
#' 
#' Sokal A. (1997). Monte Carlo Methods in Statistical Mechanics: Foundations 
#' and New Algorithms. 
#' In: DeWitt-Morette C, Cartier P, Folacci A (eds) Functional Integration. 
#' NATO ASI Series (Series B: Physics), vol 361. Springer, Boston, MA. 
#' https://doi.org/10.1007/978-1-4899-0319-8_6
#' 
#' Gelman, A, Carlin J B, Stern H S, Dunson, D B, Vehtari A, Rubin D B. (2013). 
#' Bayesian Data Analysis, Third Edition. Chapman and Hall/CRC.
#' @export
#' @srrstats {BS5.3, BS5.5}
#' @return A single numeric value of effective sample size estimate.
#' @examples
#' set.seed(1)
#' n <- 1e4 
#' x <- numeric(n)
#' phi <- 0.7
#' for(t in 2:n) x[t] <- phi * x[t-1] + rnorm(1)
#' w <- rexp(n, 0.5 * exp(0.001 * x^2))
#' # different methods:
#' estimate_ess(x, w, method = "sokal")
#' estimate_ess(x, w, method = "geyer")
#' 
estimate_ess <- function(x, w, method = "sokal") {
  
  method <- match.arg(tolower(method), c("sokal", "geyer"))
  
  if (!test_numeric(x))
    stop("Argument 'x' should be a numeric vector. ")
  
  if (missing(w)) {
    w <- rep(1, length(x))
  } else {
    if (!test_numeric(w))
      stop("Argument 'w' should be a numeric vector. ")
    if(any(w < 0) | any(!is.finite(w)))
      stop("Nonfinite or negative weights in 'w'.")
    if (!any(w > 0)) {
      stop("No positive weights in 'w'.")
    }
  }
  weighted_var(x, w) / asymptotic_var(x, w, method = method)
}
