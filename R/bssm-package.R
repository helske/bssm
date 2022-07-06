#'
#' Bayesian Inference of State Space Models
#'
#' This package contains functions for efficient Bayesian inference of state
#' space models (SSMs). For details, see the package vignette and the R Journal
#' paper.
#'
#' @details
#' The model is assumed to be either
#'
#' * Exponential family state space model, where the state equation is linear
#'   Gaussian, and the conditional observation density is either Gaussian,
#'   Poisson, binomial, negative binomial or Gamma density.
#'
#' * Basic stochastic volatility model.
#'
#' * General non-linear model with Gaussian noise terms.
#'
#' * Model with continuous SDE dynamics.
#'
#' Missing values in response series are allowed as per SSM theory and can be
#' automatically predicted, but there can be no missing values in the system
#' matrices of the model.
#'
#' The package contains multiple functions for building the model:
#'
#' * `bsm_lg` for basic univariate structural time series model (BSM),
#'   `ar1` for univariate noisy AR(1) process, and `ssm_ulg` and `ssm_mlg` for
#'   arbitrary linear gaussian model with univariate/multivariate
#'   observations.
#' * The non-Gaussian versions (where observations are non-Gaussian) of the
#'   above models can be constructed using the functions `bsm_ng`, `ar1_ng`,
#'   `ssm_ung` and `ssm_mng`.
#' * An univariate stochastic volatility model can be defined using a function
#'   `svm`.
#' * For non-linear models, user must define the model using C++ snippets and
#'   the the function `ssm_nlg`. See details in the `growth_model` vignette.
#' * Diffusion models can be defined with the function `ssm_sde`, again using
#'   the C++ snippets. See `sde_model` vignette for details.
#'
#' See the corresponding functions for some examples and details.
#'
#' After building the model, the model can be estimated via `run_mcmc`
#' function. The documentation of this function gives some examples. The
#' \code{bssm} package includes several MCMC sampling and sequential Monte
#' Carlo methods for models outside classic linear-Gaussian framework. For
#' definitions of the currently supported models and methods, usage of the
#' package as well as some theory behind the novel IS-MCMC and
#' \eqn{\psi}{psi}-APF algorithms, see Helske and Vihola (2021), Vihola,
#' Helske, Franks (2020), and the package vignettes.
#'
#' The output of the `run_mcmc` can be analysed by extracting the posterior
#' samples of the latent states and hyperparameters using `as.data.frame`,
#' `as_draws`, `expand_sample`, and `summary` methods, as well as `fitted` and
#' `predict` methods. Some MCMC diagnostics checks are available via
#' `check_diagnostics` function, some of which are also provided via the print
#' method of the `run_mcmc` output. Functionality of the `ggplot2` and
#' `bayesplot`, can be used to visualize the posterior draws or their summary
#' statistics, and further diagnostics checks can be performed with the help of
#' the `posterior` and `coda` packages.
#'
#' @references
#' Helske J, Vihola M (2021). bssm: Bayesian Inference of Non-linear and
#' Non-Gaussian State Space Models in R. The R Journal (2021) 13:2, 578-589.
#' https://doi.org/10.32614/RJ-2021-103
#'
#' Vihola, M, Helske, J, Franks, J. (2020). Importance sampling type estimators
#' based on approximate marginal Markov chain Monte Carlo.
#' Scand J Statist. 1-38. https://doi.org/10.1111/sjos.12492
#'
#' H. Wickham. ggplot2: Elegant Graphics for Data Analysis. Springer-Verlag
#' New York, 2016.
#'
#' Gabry J, Mahr T (2022). “bayesplot: Plotting for Bayesian Models.” R package
#' version 1.9.0, https://mc-stan.org/bayesplot.
#'
#' Bürkner P, Gabry J, Kay M, Vehtari A (2022). “posterior: Tools for Working
#' with Posterior Distributions.” R package version 1.2.1,
#' https://mc-stan.org/posterior.
#'
#' Martyn Plummer, Nicky Best, Kate Cowles and Karen Vines (2006). CODA:
#' Convergence Diagnosis and Output Analysis for MCMC, R News, vol 6, 7-11.
#'
#' @docType package
#' @name bssm
#' @aliases bssm
#' @importFrom Rcpp evalCpp
#' @importFrom stats as.ts dnorm  end frequency is.ts logLik quantile start
#' time ts ts.union tsp tsp<- sd na.omit
#' @useDynLib bssm
#' @examples
#' # Create a local level model (latent random walk + noise) to the Nile
#' # dataset using the bsm_lg function:
#' model <- bsm_lg(Nile,
#'   sd_y = tnormal(init = 100, mean = 100, sd = 100, min = 0),
#'   sd_level = tnormal(init = 50, mean = 50, sd = 100, min = 0),
#'   a1 = 1000, P1 = 500^2)
#'
#' # the priors for the unknown paramters sd_y and sd_level were defined
#' # as trunctated normal distributions, see ?bssm_prior for details
#'
#' # Run the MCMC for 2000 iterations (notice the small number of iterations to
#' # comply with the CRAN's check requirements)
#' fit <- run_mcmc(model, iter = 2000)
#'
#' # Some diagnostics checks:
#' check_diagnostics(fit)
#'
#' # print some summary information:
#' fit
#'
#' # traceplots:
#' plot(fit)
#'
#' # extract the summary statistics for state variable
#' sumr <- summary(fit,variable = "states")
#'
#' # visualize
#' library("ggplot2")
#' ggplot(sumr, aes(time, Mean)) +
#'     geom_ribbon(aes(ymin = `2.5%`, ymax = `97.5%`),alpha = 0.25) +
#'     geom_line() +
#'     theme_bw()
#'
NULL
#' Deaths by drowning in Finland in 1969-2019
#'
#' Dataset containing number of deaths by drowning in Finland in 1969-2019,
#' corresponding population sizes (in hundreds of thousands), and
#' yearly average summer temperatures (June to August), based on simple
#' unweighted average of three weather stations: Helsinki (Southern Finland),
#' Jyvaskyla (Central Finland), and Sodankyla (Northern Finland).
#'
#' @name drownings
#' @docType data
#' @format A time series object containing 51 observations.
#' @source Statistics Finland
#' \url{https://pxnet2.stat.fi/PXWeb/pxweb/en/StatFin/}.
#' @keywords datasets
#' @examples
#' data("drownings")
#' model <- bsm_ng(drownings[, "deaths"], u = drownings[, "population"],
#'   xreg = drownings[, "summer_temp"], distribution = "poisson",
#'   beta = normal(0, 0, 1),
#'   sd_level = gamma_prior(0.1,2, 10), sd_slope = gamma_prior(0, 2, 10))
#'
#' fit <- run_mcmc(model, iter = 5000,
#'   output_type = "summary", mcmc_type = "approx")
#' fit
#' ts.plot(model$y/model$u, exp(fit$alphahat[, 1]), col = 1:2)
NULL
#' Pound/Dollar daily exchange rates
#'
#' Dataset containing daily log-returns from 1/10/81-28/6/85 as in Durbin and
#' Koopman (2012).
#'
#' @name exchange
#' @docType data
#' @format A vector of length 945.
#' @source The data used to be available on the www.ssfpack.com/DKbook.html but
#' this page is does not seem to be available anymore.
#' @keywords datasets
#' @references
#' James Durbin, Siem Jan Koopman (2012).
#' Time Series Analysis by State Space Methods. Oxford University Press.
#' https://doi.org/10.1093/acprof:oso/9780199641178.001.0001
#' @examples
#' data("exchange")
#' model <- svm(exchange, rho = uniform(0.97,-0.999,0.999),
#'  sd_ar = halfnormal(0.175, 2), mu = normal(-0.87, 0, 2))
#'
#' out <- particle_smoother(model, particles = 500)
#' plot.ts(cbind(model$y, exp(out$alphahat)))
NULL
#' Simulated Poisson Time Series Data
#'
#' See example for code for reproducing the data. This was used in
#' Vihola, Helske, Franks (2020).
#'
#' @srrstats {G5.0, G5.1, G5.4} used in Vihola, Helske, Franks (2020).
#' @name poisson_series
#' @docType data
#' @format A vector of length 100.
#' @keywords datasets
#' @references
#' Vihola, M, Helske, J, Franks, J (2020). Importance sampling type
#' estimators based on approximate marginal Markov chain Monte Carlo.
#' Scand J Statist. 1-38. https://doi.org/10.1111/sjos.12492
#'
#' @examples
#' # The data was generated as follows:
#' set.seed(321)
#' slope <- cumsum(c(0, rnorm(99, sd = 0.01)))
#' y <- rpois(100, exp(cumsum(slope + c(0, rnorm(99, sd = 0.1)))))
NULL
#'
#' Simulated Negative Binomial Time Series Data
#'
#' See example for code for reproducing the data. This was used in
#' Helske and Vihola (2021).
#'
  #' @srrstats {G5.1} used in Helske and Vihola (2021).
#' @name negbin_series
#' @docType data
#' @format A time series \code{mts} object with 200 time points and two series.
#' @keywords datasets
#' @seealso \code{negbin_model}
#' @references
#' Helske J, Vihola M (2021). bssm: Bayesian Inference of Non-linear and
#' Non-Gaussian State Space Models in R. The R Journal (2021) 13:2, 578-589.
#' https://doi.org/10.32614/RJ-2021-103
#'
#' @examples
#' # The data was generated as follows:
#' set.seed(123)
#' n <- 200
#' sd_level <- 0.1
#' drift <- 0.01
#' beta <- -0.9
#' phi <- 5
#'
#' level <- cumsum(c(5, drift + rnorm(n - 1, sd = sd_level)))
#' x <- 3 + (1:n) * drift + sin(1:n + runif(n, -1, 1))
#' y <- rnbinom(n, size = phi, mu = exp(beta * x + level))
#'
NULL
#' Estimated Negative Binomial Model of Helske and Vihola (2021)
#'
#' This model was used in Helske and Vihola (2021), but with larger number of
#' iterations. Here only 2000 iterations were used in order to reduce the size
#' of the model object in CRAN.
#'
#' @srrstats {G5.0, G5.1, G5.4, BS7.2} used in Helske and Vihola (2021).
#' @name negbin_model
#' @docType data
#' @format A object of class \code{mcmc_output}.
#' @keywords datasets
#' @references
#' Helske J, Vihola M (2021). bssm: Bayesian Inference of Non-linear and
#' Non-Gaussian State Space Models in R. The R Journal (2021) 13:2, 578-589.
#' https://doi.org/10.32614/RJ-2021-103
#'
#' @examples
#' # reproducing the model:
#' data("negbin_series")
#' # Construct model for bssm
#' bssm_model <- bsm_ng(negbin_series[, "y"],
#'   xreg = negbin_series[, "x"],
#'   beta = normal(0, 0, 10),
#'   phi = halfnormal(1, 10),
#'   sd_level = halfnormal(0.1, 1),
#'   sd_slope = halfnormal(0.01, 0.1),
#'   a1 = c(0, 0), P1 = diag(c(10, 0.1)^2),
#'   distribution = "negative binomial")
#'
#' \donttest{
#' # In the paper we used 60000 iterations with first 10000 as burnin
#' fit_bssm <- run_mcmc(bssm_model, iter = 2000, particles = 10, seed = 1)
#' fit_bssm
#' }
NULL
