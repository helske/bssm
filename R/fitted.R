#' Fitted for State Space Model
#' 
#' Returns summary statistics from the posterior predictive 
#' distribution of the mean.
#' 
#' @export
#' @importFrom stats fitted
#' @importFrom magrittr %>%
#' @importFrom dplyr group_by ungroup summarise as_tibble
#' @importFrom diagis weighted_quantile weighted_var weighted_mean weighted_se
#' @name fitted.mcmc_output
#' @param object Results object of class \code{mcmc_output} from 
#' \code{\link{run_mcmc}} based on the input model.
#' @param model A \code{bssm_model} object.
#' @param probs Numeric vector defining the quantiles of interest. Default is 
#' \code{c(0.025, 0.975)}.
#' @param ... Ignored.
#' @examples
#' prior <- uniform(0.1 * sd(log10(UKgas)), 0, 1)
#' model <- bsm_lg(log10(UKgas), sd_y = prior, sd_level =  prior,
#'   sd_slope =  prior, sd_seasonal =  prior, period = 4)
#' fit <- run_mcmc(model, iter = 1e4)
#' res <- fitted(fit, model) 
#' head(res)
#' 
fitted.mcmc_output <- function(object, model, 
  probs = c(0.025, 0.975), ...)  {
  
  if (!inherits(model, "bssm_model")) {
    stop("Argument 'model' should be an object of class 'bssm_model'.")
  }
  if (inherits(model, c("ssm_mng", "ssm_mlg", "ssm_nlg"))) {
    if (!identical(nrow(object$alpha) - 1L, nrow(model$y))) {
      stop("Number of observations of the model and MCMC output do not match.") 
    }
  } else {
    if (!identical(nrow(object$alpha) - 1L, length(model$y))) {
      stop("Number of observations of the model and MCMC output do not match.") 
    }
  }
  
  if (any(probs < 0 | probs > 1)) stop("'probs' outside [0, 1].")
  
  n <- nrow(object$alpha) - 1L
  m <- ncol(object$alpha)
  
  states <- aperm(object$alpha[1:n, , , drop = FALSE], c(2, 1, 3))
  theta <- t(object$theta)
  
  switch(attr(object, "model_type"),
    ssm_mlg =,
    ssm_ulg =,
    bsm_lg =,
    ar1_lg = {
      if (!identical(length(model$a1), m)) {
        stop(paste("Model does not correspond to the MCMC output:",
          "Wrong number of states. ", sep = " "))
      }
      pred <- gaussian_predict_past(model, theta, states,
        2L, 1L, 
        pmatch(attr(object, "model_type"), 
          c("ssm_mlg", "ssm_ulg", "bsm_lg", "ar1_lg")) - 1L)
      
    },
    ssm_mng =, 
    ssm_ung =, 
    bsm_ng =, 
    svm =,
    ar1_ng = {
      if (!identical(length(model$a1), m)) {
        stop(paste("Model does not correspond to the MCMC output:",
          "Wrong number of states. ", sep = " "))
      }
      model$distribution <- pmatch(model$distribution,
        c("svm", "poisson", "binomial", "negative binomial", "gamma", 
          "gaussian"), 
        duplicates.ok = TRUE) - 1
      pred <- nongaussian_predict_past(model, theta, states,
        2L, 1L, 
        pmatch(attr(object, "model_type"), 
          c("ssm_mng", "ssm_ung", "bsm_ng", "svm", "ar1_ng")) - 1L)
      
      if (anyNA(pred)) 
        warning("NA or NaN values in predictions, possible under/overflow?")
    },
    ssm_nlg = {
      if (!identical(model$n_states, m)) {
        stop(paste("Model does not correspond to the MCMC output:",
          "Wrong number of states. ", sep = " "))
      }
      pred <- nonlinear_predict_past(t(model$y), model$Z, 
        model$H, model$T, model$R, model$Z_gn, 
        model$T_gn, model$a1, model$P1, 
        model$log_prior_pdf, model$known_params, 
        model$known_tv_params, as.integer(model$time_varying),
        model$n_states, model$n_etas,
        theta, states, 2L, 1L)
      
    }
    , stop("Not yet implemented for ssm_sde. "))
  
  variables <- colnames(model$y)
  if (is.null(variables)) 
    variables <- paste("Series", 1:max(1, ncol(model$y)))
  w <- object$counts * 
    (if (object$mcmc_type %in% paste0("is", 1:3)) object$weights else 1)
  
  
  d <- data.frame(value = as.numeric(pred),
    Variable = variables,
    Time = rep(time(model$y), each = nrow(pred)))
  
  d %>% dplyr::group_by(.data$Variable, .data$Time) %>%
    dplyr::summarise(
      Mean = weighted_mean(.data$value, w),
      SD = sqrt(weighted_var(.data$value, w)),
      dplyr::as_tibble(as.list(weighted_quantile(.data$value, w, 
        probs = probs))),
      "SE(Mean)" = as.numeric(sqrt(asymptotic_var(.data$value, w)))) %>% 
    dplyr::ungroup()
}

