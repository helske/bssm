#' Predictions for State Space Models
#' 
#' Draw samples from the posterior predictive distribution given the 
#' posterior draws of hyperparameters theta and alpha_{n+1}.
#'
#' @param object mcmc_output object obtained from 
#' \code{\link{run_mcmc}}
#' @param type Return predictions on \code{"mean"} 
#' \code{"response"}, or  \code{"state"} level. 
#' @param future_model Model for future observations. 
#' Should have same structure
#' as the original model which was used in MCMC, in order 
#' to plug the posterior samples of the model parameters to the right places.
#' @param nsim Number of samples to draw.
#' @param seed Seed for RNG.
#' @param ... Ignored.
#' @return Data frame of predicted samples.
#' @method predict mcmc_output
#' @rdname predict
#' @export
#' @examples
#' require("graphics")
#' y <- log10(JohnsonJohnson)
#' prior <- uniform(0.01, 0, 1)
#' model <- bsm_lg(window(y, end = c(1974, 4)), sd_y = prior,
#'   sd_level = prior, sd_slope = prior, sd_seasonal = prior)
#' 
#' mcmc_results <- run_mcmc(model, iter = 5000)
#' future_model <- model
#' future_model$y <- ts(rep(NA, 25), 
#'   start = tsp(model$y)[2] + 2 * deltat(model$y), 
#'   frequency = frequency(model$y))
#' pred <- predict(mcmc_results, future_model, type = "state", 
#'   nsim = 1000)
#' 
#' require("dplyr")
#' sumr_fit <- as.data.frame(mcmc_results, "states") %>%
#'   group_by(time, iter) %>% 
#'   mutate(signal = 
#'       value[variable == "level"] + 
#'       value[variable == "seasonal_1"]) %>%
#'   group_by(time) %>%
#'   summarise(mean = mean(signal), 
#'     lwr = quantile(signal, 0.025), 
#'     upr = quantile(signal, 0.975))
#' 
#' sumr_pred <- pred %>% 
#'   group_by(time, sample) %>%
#'   mutate(signal = 
#'       value[variable == "level"] + 
#'       value[variable == "seasonal_1"]) %>%
#'   group_by(time) %>%
#'   summarise(mean = mean(signal),
#'     lwr = quantile(signal, 0.025), 
#'     upr = quantile(signal, 0.975)) 
#' 
#' require("ggplot2")
#' rbind(sumr_fit, sumr_pred) %>% 
#'   ggplot(aes(x = time, y = mean)) + 
#'   geom_ribbon(aes(ymin = lwr, ymax = upr), 
#'    fill = "#92f0a8", alpha = 0.25) +
#'   geom_line(colour = "#92f0a8") +
#'   theme_bw() + 
#'   geom_point(data = data.frame(
#'     mean = log10(JohnsonJohnson), 
#'     time = time(JohnsonJohnson)))
#' 
predict.mcmc_output <- function(object, future_model, type = "response",
  seed = sample(.Machine$integer.max, size = 1), nsim, ...) {
  
  type <- match.arg(type, c("response", "mean", "state"))
  
  if (object$output_type != 1) stop("MCMC output must contain posterior samples of the states.")
  
  
  if (attr(object, "model_type") %in% c("bsm_lg", "bsm_ng")) {
    object$theta[,1:(ncol(object$theta) - length(future_model$beta))] <- 
      log(object$theta[,1:(ncol(object$theta) - length(future_model$beta))])
  }
  w <- object$counts * (if(object$mcmc_type %in% paste0("is", 1:3)) object$weights else 1)
  idx <- sample(1:nrow(object$theta), size = nsim, prob = w, replace = TRUE)
  theta <- t(object$theta[idx, ])
  alpha <- matrix(object$alpha[nrow(object$alpha),,idx], nrow = ncol(object$alpha))
  
  switch(attr(object, "model_type"),
    ssm_mlg = ,
    ssm_ulg = ,
    bsm_lg = ,
    ar1_lg = {
      pred <- gaussian_predict(future_model, theta, alpha,
        pmatch(type, c("response", "mean", "state")), 
        seed, 
        pmatch(attr(object, "model_type"), 
          c("ssm_mng", "ssm_ulg", "bsm_lg", "ar1_lg")) - 1L)
        
    },
    ssm_mng = , 
    ssm_ung = , 
    bsm_ng = , 
    svm = {
      future_model$distribution <- pmatch(future_model$distribution,
        c("svm", "poisson", "binomial", "negative binomial", "gamma", "gaussian"), 
        duplicates.ok = TRUE) - 1
      pred <- nongaussian_predict(future_model, theta, alpha,
          pmatch(type, c("response", "mean", "state")), seed, 
          pmatch(attr(object, "model_type"), 
            c("ssm_mng", "ssm_ung", "bsm_ng", "svm", "ar1_ng")) - 1L)
      
      if(anyNA(pred)) warning("NA or NaN values in predictions, possible under/overflow?")
    },
    ssm_nlg = {
      
        pred <- nonlinear_predict(t(future_model$y), future_model$Z, 
          future_model$H, future_model$T, future_model$R, future_model$Z_gn, 
          future_model$T_gn, future_model$a1, future_model$P1, 
          future_model$log_prior_pdf, future_model$known_params, 
          future_model$known_tv_params, as.integer(future_model$time_varying),
          future_model$n_states, future_model$n_etas,
          theta, alpha, pmatch(type, c("response", "mean", "state")), seed)
      
      }
   , stop("Not yet implemented for ssm_sde. "))
  if(type == "state") {
    if(attr(object, "model_type") == "ssm_nl") {
      variables <- future_model$state_names
    } else {
      variables <- names(future_model$a1)
    }
  } else {
    variables <- colnames(future_model$y)
    if(is.null(variables)) variables <- "Series 1"
  }
  
  data.frame(value = as.numeric(pred),
    variable = variables,
    time = rep(time(future_model$y), each = nrow(pred)),
      sample = rep(1:nsim, each = nrow(pred) * ncol(pred)))
}
