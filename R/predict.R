#' Predictions for State Space Models
#' 
#' Draw samples from the posterior predictive distribution for future 
#' time points given the posterior draws of hyperparameters \eqn{\theta} and 
#' latent state \eqn{alpha_{n+1}} returned by \code{run_mcmc}. 
#' Function can also be used to draw samples from the posterior predictive 
#' distribution \eqn{p(\tilde y_1, \ldots, \tilde y_n | y_1,\ldots, y_n)}.
#' 
#' @seealso \code{fitted} for in-sample predictions.
#' @param object Results object of class \code{mcmc_output} from 
#' \code{\link{run_mcmc}}.
#' @param model A \code{bssm_model} object.
#' Should have same structure and class as the original model which was used in 
#' \code{run_mcmc}, in order to plug the posterior samples of the model 
#' parameters to the right places. 
#' It is also possible to input the original model for obtaining predictions 
#' for past time points. In this case, set argument 
#' \code{future} to \code{FALSE}.
#' @param type Type of predictions. Possible choices are 
#' \code{"mean"} \code{"response"}, or  \code{"state"} level. 
#' @param nsim Positive integer defining number of samples to draw.
#' @param future Default is \code{TRUE}, in which case predictions are for the 
#' future, using posterior samples of (theta, alpha_T+1) i.e. the 
#' posterior samples of hyperparameters and latest states. 
#' Otherwise it is assumed that \code{model} corresponds to the original model.
#' @param seed Seed for RNG (positive integer).
#' @param ... Ignored.
#' @return A \code{data.frame} consisting of samples from the predictive 
#' posterior distribution.
#' @method predict mcmc_output
#' @aliases predict predict.mcmc_output
#' @export
#' @examples
#' library("graphics")
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
#' # use "state" for illustrative purposes, we could use type = "mean" directly
#' pred <- predict(mcmc_results, future_model, type = "state", 
#'   nsim = 1000)
#' 
#' library("dplyr")
#' sumr_fit <- as.data.frame(mcmc_results, variable = "states") %>%
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
#' # If we used type = "mean", we could do
#' # sumr_pred <- pred %>% 
#' #   group_by(time) %>%
#' #   summarise(mean = mean(value),
#' #     lwr = quantile(value, 0.025), 
#' #     upr = quantile(value, 0.975)) 
#'     
#' library("ggplot2")
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
#' # Posterior predictions for past observations:
#' yrep <- predict(mcmc_results, model, type = "response", 
#'   future = FALSE, nsim = 1000)
#' meanrep <- predict(mcmc_results, model, type = "mean", 
#'   future = FALSE, nsim = 1000)
#'   
#' sumr_yrep <- yrep %>% 
#'   group_by(time) %>%
#'   summarise(earnings = mean(value),
#'     lwr = quantile(value, 0.025), 
#'     upr = quantile(value, 0.975)) %>%
#'   mutate(interval = "Observations")
#'
#' sumr_meanrep <- meanrep %>% 
#'   group_by(time) %>%
#'   summarise(earnings = mean(value),
#'     lwr = quantile(value, 0.025), 
#'     upr = quantile(value, 0.975)) %>%
#'   mutate(interval = "Mean")
#'     
#' rbind(sumr_meanrep, sumr_yrep) %>% 
#'   mutate(interval = factor(interval, levels = c("Observations", "Mean"))) %>%
#'   ggplot(aes(x = time, y = earnings)) + 
#'   geom_ribbon(aes(ymin = lwr, ymax = upr, fill = interval), 
#'    alpha = 0.75) +
#'   theme_bw() + 
#'   geom_point(data = data.frame(
#'     earnings = model$y, 
#'     time = time(model$y)))    
#' 
#' 
predict.mcmc_output <- function(object, model, nsim, type = "response",  
  future = TRUE, seed = sample(.Machine$integer.max, size = 1), ...) {
  
  check_missingness(model)
  
  if (!inherits(model, "bbsm_model")) {
    stop("Argument 'model' should be an object of class 'bssm_model'.")
  }
  nsim <- check_intmax(nsim, "nsim", max = 10 * object$iter)
  seed <- check_intmax(seed, "seed", FALSE, max = .Machine$integer.max)
  
  if (!test_flag(future)) stop("Argument 'future' should be TRUE or FALSE. ")
  
  type <- match.arg(tolower(type), c("response", "mean", "state"))
  
  if (object$output_type != 1) 
    stop("MCMC output must contain posterior samples of the states.")
  
  if (!identical(attr(object, "model_type"), class(model)[1])) {
    stop("Model class does not correspond to the MCMC output. ")
  }
  if (!identical(ncol(object$theta), length(model$theta))) {
    stop(paste("Number of unknown parameters 'theta' does not correspond to",
      "the MCMC output. ", sep = " "))
  }
  if (nsim < 1) stop("Number of samples 'nsim' should be at least one.")
  
  if (attr(object, "model_type") %in% c("bsm_lg", "bsm_ng")) {
    object$theta[, 1:(ncol(object$theta) - length(model$beta))] <- 
      log(object$theta[, 1:(ncol(object$theta) - length(model$beta))])
  } else {
    if (attr(object, "model_type") == "ar1_lg") {
      object$theta[, c("sigma", "sd_y")] <- 
        log(object$theta[, c("sigma", "sd_y")])
    } else {
      if (attr(object, "model_type") == "ar1_ng") {
        disp <- ifelse(
          object$distribution %in% c("negative binomial", "gamma"), 
          "phi", NULL)
        object$theta[, c("sigma", disp)] <- 
          log(object$theta[, c("sigma", disp)])
      }
    }
  }
  
  if (future) {
    
    w <- object$counts * 
      (if (object$mcmc_type %in% paste0("is", 1:3)) object$weights else 1)
    idx <- sample(seq_len(nrow(object$theta)), size = nsim, prob = w, 
      replace = TRUE)
    theta <- t(object$theta[idx, , drop = FALSE])
    alpha <- matrix(object$alpha[nrow(object$alpha), , idx], 
      nrow = ncol(object$alpha))
    
    switch(attr(object, "model_type"),
      ssm_mlg =,
      ssm_ulg =,
      bsm_lg =,
      ar1_lg = {
        if (!identical(length(model$a1), ncol(object$alpha))) {
          stop(paste("Model does not correspond to the MCMC output:",
            "Wrong number of states. ", sep = " "))
        }
        pred <- gaussian_predict(model, theta, alpha,
          pmatch(type, c("response", "mean", "state")), 
          seed, 
          pmatch(attr(object, "model_type"), 
            c("ssm_mlg", "ssm_ulg", "bsm_lg", "ar1_lg")) - 1L)
        
      },
      ssm_mng =, 
      ssm_ung =, 
      bsm_ng =, 
      svm =,
      ar1_ng = {
        if (!identical(length(model$a1), ncol(object$alpha))) {
          stop(paste("Model does not correspond to the MCMC output:",
            "Wrong number of states. ", sep = " "))
        }
        model$distribution <- pmatch(model$distribution,
          c("svm", "poisson", "binomial", "negative binomial", "gamma", 
            "gaussian"), 
          duplicates.ok = TRUE) - 1
        pred <- nongaussian_predict(model, theta, alpha,
          pmatch(type, c("response", "mean", "state")), seed, 
          pmatch(attr(object, "model_type"), 
            c("ssm_mng", "ssm_ung", "bsm_ng", "svm", "ar1_ng")) - 1L)
        
        if (anyNA(pred)) 
          warning("NA or NaN values in predictions, possible under/overflow?")
      },
      ssm_nlg = {
        if (!identical(model$n_states, ncol(object$alpha))) {
          stop(paste("Model does not correspond to the MCMC output:",
            "Wrong number of states. ", sep = " "))
        }
        pred <- nonlinear_predict(t(model$y), model$Z, 
          model$H, model$T, model$R, model$Z_gn, 
          model$T_gn, model$a1, model$P1, 
          model$log_prior_pdf, model$known_params, 
          model$known_tv_params, as.integer(model$time_varying),
          model$n_states, model$n_etas,
          theta, alpha, pmatch(type, c("response", "mean", "state")), seed)
        
      }
      , stop("Not yet implemented for ssm_sde. "))
    if (type == "state") {
      if (attr(object, "model_type") == "ssm_nlg") {
        variables <- model$state_names
      } else {
        variables <- names(model$a1)
      }
    } else {
      variables <- colnames(model$y)
      if (is.null(variables)) 
        variables <- paste("Series", 1:max(1, ncol(model$y)))
    }
    d <- data.frame(value = as.numeric(pred),
      variable = variables,
      time = rep(time(model$y), each = nrow(pred)),
      sample = rep(1:nsim, each = nrow(pred) * ncol(pred)))
    
  } else {
    
    if (inherits(model, c("ssm_mng", "ssm_mlg", "ssm_nlg"))) {
      if (!identical(nrow(object$alpha) - 1L, nrow(model$y))) {
        stop("Number of observations of the model and MCMC output do not match.") 
      }
    } else {
      if (!identical(nrow(object$alpha) - 1L, length(model$y))) {
        stop("Number of observations of the model and MCMC output do not match.") 
      }
    }
    w <- object$counts * 
      (if (object$mcmc_type %in% paste0("is", 1:3)) object$weights else 1)
    idx <- sample(seq_len(nrow(object$theta)), size = nsim, prob = w, 
      replace = TRUE)
    n <- nrow(object$alpha) - 1L
    m <- ncol(object$alpha)
    
    states <- object$alpha[1:n, , idx, drop = FALSE]
    
    if (type == "state") {
      if (attr(object, "model_type") == "ssm_nlg") {
        variables <- model$state_names
      } else {
        variables <- names(model$a1)
      }
      d <- data.frame(value = as.numeric(states),
        variable = rep(variables, each = n),
        time = rep(time(model$y), times = m),
        sample = rep(1:nsim, each = n * m))
    } else {
      
      variables <- colnames(model$y)
      if (is.null(variables)) 
        variables <- paste("Series", 1:max(1, ncol(model$y)))
      
      theta <- t(object$theta[idx, ])
      states <- aperm(states, c(2, 1, 3))
      
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
            pmatch(type, c("response", "mean", "state")), 
            seed, 
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
            pmatch(type, c("response", "mean", "state")), seed, 
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
            theta, states, pmatch(type, c("response", "mean", "state")), seed)
          
        }
        , stop("Not yet implemented for ssm_sde. "))
      
      d <- data.frame(value = as.numeric(pred),
        variable = variables,
        time = rep(time(model$y), each = nrow(pred)),
        sample = rep(1:nsim, each = nrow(pred) * ncol(pred)))
    }
  }
  d
}
