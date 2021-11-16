#' Print Results from MCMC Run
#'
#' Prints some basic summaries from the MCMC run by  \code{\link{run_mcmc}}.
#'  
#' @method print mcmc_output
#' @importFrom diagis weighted_mean weighted_var weighted_se ess
#' @importFrom coda mcmc
#' @importFrom stats var
#' @param x Output from \code{\link{run_mcmc}}.
#' @param ... Ignored.
#' @export
print.mcmc_output <- function(x, ...) {
  
  if (x$mcmc_type %in% paste0("is", 1:3)) {
    theta <- mcmc(x$theta)
    if (x$output_type == 1)
      alpha <- mcmc(matrix(x$alpha[nrow(x$alpha), , ], ncol = ncol(x$alpha), 
        byrow = TRUE, dimnames = list(NULL, colnames(x$alpha))))
    w <- x$counts * x$weights
  } else {
    theta <- expand_sample(x, "theta")
    if (x$output_type == 1)
      alpha <- 
        expand_sample(x, "state", times = nrow(x$alpha), by_states = FALSE)[[1]]
  }
  
  cat("\nCall:\n", paste(deparse(x$call), sep = "\n", collapse = "\n"), 
    "\n", sep = "")
  
  cat("\n", "Iterations = ", x$burnin + 1, ":", x$iter, "\n", sep = "")
  cat("Thinning interval = ", x$thin, "\n", sep = "")
  cat("Length of the final jump chain = ", length(x$counts), "\n", sep = "")
  cat("\nAcceptance rate after the burn-in period: ", 
    paste(round(x$acceptance_rate, 3), "\n", sep = ""))
  
  cat("\nSummary for theta:\n\n")
  if (x$mcmc_type %in% paste0("is", 1:3)) {
    mean_theta <- weighted_mean(theta, w)
    sd_theta <- sqrt(diag(weighted_var(theta, w, method = "moment")))
    se_theta_is <- weighted_se(theta, w)
    se_theta <- sqrt(apply(theta, 2, function(x) asymptotic_var(x, w)))
    
    stats <- matrix(c(mean_theta, sd_theta, se_theta, se_theta_is), ncol = 4, 
      dimnames = list(colnames(x$theta), c("Mean", "SD", "SE", "SE-IS")))
  } else {
    mean_theta <- colMeans(theta)
    sd_theta <- apply(theta, 2, sd)
    se_theta <- sqrt(apply(theta, 2, function(x) iact(x) * var(x) / length(x)))
    #se_theta <-  sqrt(spectrum0.ar(theta)$spec / nrow(theta))
    stats <- matrix(c(mean_theta, sd_theta, se_theta), ncol = 3, 
      dimnames = list(colnames(x$theta), c("Mean", "SD", "SE")))
  }
  
  print(stats)
  
  cat("\nEffective sample sizes for theta:\n\n")
  esss <- matrix((sd_theta / se_theta)^2, ncol = 1, 
    dimnames = list(colnames(x$theta), c("ESS")))
  print(esss)
  if (x$output_type != 3) {
    
    n <- nrow(x$alpha)
    cat(paste0("\nSummary for alpha_", n), ":\n\n", sep = "")
    
    if (is.null(x$alphahat)) {
      if (x$mcmc_type %in% paste0("is", 1:3)) {
        mean_alpha <- weighted_mean(alpha, w)
        sd_alpha <- sqrt(diag(weighted_var(alpha, w, method = "moment")))
        se_alpha_is <- weighted_se(alpha, w)
        se_alpha <- sqrt(apply(alpha, 2, function(x) asymptotic_var(x, w)))
        stats <- matrix(c(mean_alpha, sd_alpha, se_alpha, se_alpha_is),
          ncol = 4, 
          dimnames = list(colnames(x$alpha), c("Mean", "SD", "SE", "SE-IS")))
      } else {
        mean_alpha <- colMeans(alpha)
        sd_alpha <- apply(alpha, 2, sd)
        se_alpha <-  sqrt(apply(alpha, 2, function(x) iact(x) * var(x) / length(x)))
        #sqrt(spectrum0.ar(alpha)$spec / nrow(alpha))
        stats <- matrix(c(mean_alpha, sd_alpha, se_alpha), ncol = 3, 
          dimnames = list(colnames(x$alpha), c("Mean", "SD", "SE")))
      }
      print(stats)
      
      
      cat(paste0("\nEffective sample sizes for alpha_", n), ":\n\n", sep = "")
      esss <- matrix((sd_alpha / se_alpha)^2, ncol = 1, 
        dimnames = list(colnames(x$alpha), c("ESS")))
      
      print(esss)
      
    } else {
      if (ncol(x$alphahat) == 1) {
        print(cbind("Mean" = x$alphahat[n, ], "SD" = sqrt(x$Vt[, , n])))
      } else {
        print(cbind("Mean" = x$alphahat[n, ], "SD" = sqrt(diag(x$Vt[, , n]))))
      }
    }
  } else cat("\nNo posterior samples for states available.\n")
  cat("\nRun time:\n")
  print(x$time)
}

#' Summary of MCMC object
#' 
#' This functions returns a list containing mean, standard deviations, 
#' standard errors, and effective sample size estimates for parameters and 
#' states.
#' 
#' For IS-MCMC two types of standard errors are reported. 
#' SE-IS can be regarded as the square root of independent IS variance,
#' whereas SE corresponds to the square root of total asymptotic variance 
#' (see Remark 3 of Vihola et al. (2020)).
#' 
#' 
#' @param object Output from \code{run_mcmc}
#' @param return_se if \code{FALSE} (default), computation of standard 
#' errors and effective sample sizes is omitted. 
#' @param variable Are the summary statistics computed for either 
#' \code{"theta"} (default), \code{"states"}, or \code{"both"}?
#' @param only_theta Deprecated. If \code{TRUE}, summaries are computed only 
#' for hyperparameters theta, not latent states alpha.
#' @param ... Ignored.
#' @references 
#' Vihola, M, Helske, J, Franks, J. Importance sampling type estimators based 
#' on approximate marginal Markov chain Monte Carlo. 
#' Scand J Statist. 2020; 1-38. https://doi.org/10.1111/sjos.12492
#' @export
summary.mcmc_output <- function(object, return_se = FALSE, variable = "theta", 
  only_theta = FALSE, ...) {
  
  
  if (!test_flag(return_se)) 
    stop("Argument 'return_se' should be TRUE or FALSE. ")
  if (!test_flag(only_theta)) 
    stop("Argument 'only_theta' should be TRUE or FALSE. ")
  
  if (only_theta) {
    variable <- "theta"
    warning(paste("Argument 'only_theta' is deprecated. Use argument", 
    "'variable' instead. ", sep = " "))
  }
  variable <- match.arg(tolower(variable), c("theta", "states", "both"))
  
  if (variable %in% c("theta", "both")) {
    if (object$mcmc_type %in% paste0("is", 1:3)) {
      theta <- mcmc(object$theta)
      w <- object$counts * object$weights
      mean_theta <- weighted_mean(theta, w)
      sd_theta <- sqrt(diag(weighted_var(theta, w, method = "moment")))
      
      if (return_se) {
        se_theta_is <- weighted_se(theta, w)
        se_theta <- sqrt(apply(theta, 2, function(x) asymptotic_var(x, w)))
        ess_theta <- (sd_theta / se_theta)^2
        ess_w <- apply(object$theta, 2, function(x) ess(w, identity, x))
        summary_theta <- matrix(c(mean_theta, sd_theta, se_theta, ess_theta, 
          se_theta_is, ess_w), ncol = 6, 
          dimnames = list(colnames(object$theta), 
            c("Mean", "SD", "SE", "ESS", "SE-IS", "ESS-IS")))
      } else {
        summary_theta <- matrix(c(mean_theta, sd_theta), ncol = 2, 
          dimnames = list(colnames(object$theta), c("Mean", "SD")))
      }
    } else {
      theta <- expand_sample(object, "theta")
      mean_theta <- colMeans(theta)
      sd_theta <- apply(theta, 2, sd)
      
      if (return_se) {
        #sqrt(spectrum0.ar(theta)$spec / nrow(theta))
        se_theta <- sqrt(apply(theta, 2, function(x) iact(x) * var(x) / length(x)))
        ess_theta <- (sd_theta / se_theta)^2
        summary_theta <- matrix(c(mean_theta, sd_theta, se_theta, ess_theta), 
          ncol = 4, 
          dimnames = list(colnames(object$theta), c("Mean", "SD", "SE", "ESS")))
      } else {
        summary_theta <- matrix(c(mean_theta, sd_theta), ncol = 2, 
          dimnames = list(colnames(object$theta), c("Mean", "SD")))
      }
    }
  }
  
  if (variable %in% c("states", "both")) {
    if (object$output_type != 1) 
      stop("Cannot return summary of states as the MCMC type was not 'full'. ")
    
    m <- ncol(object$alpha)
    
    if (object$mcmc_type %in% paste0("is", 1:3)) {
      w <- object$counts * object$weights
      mean_alpha <- ts(weighted_mean(object$alpha, w), 
        start = attr(object, "ts")$start,
        frequency = attr(object, "ts")$frequency, 
        names = colnames(object$alpha))
      sd_alpha <- weighted_var(object$alpha, w, method = "moment")
      sd_alpha <- if (m > 1) {
        sqrt(t(apply(sd_alpha, 3, diag))) 
      } else matrix(sqrt(sd_alpha), ncol = 1)
      
      
      if (return_se) {
        se_alpha_is <- apply(object$alpha, 2, 
          function(x) weighted_se(t(x), w))
        
        se_alpha <- apply(object$alpha, 2, 
          function(z) sqrt(apply(z, 1, function(x) asymptotic_var(x, w))))
        alpha_ess <- (sd_alpha / se_alpha)^2
        ess_w <- apply(object$alpha, 2, 
          function(z) apply(z, 1, function(x) ess(w, identity, x)))
        summary_alpha <- list(
          "Mean" = mean_alpha, "SD" = sd_alpha, 
          "SE" = se_alpha, "ESS" = alpha_ess, 
          "SE-IS" = se_alpha_is, "ESS-IS" = ess_w)
      } else {
        summary_alpha <- list("Mean" = mean_alpha, "SD" = sd_alpha)
      }
      
    } else {
      alpha <- expand_sample(object, "states")
      mean_alpha <- ts(vapply(alpha, colMeans, numeric(nrow(object$alpha))),
        start = attr(object, "ts")$start,
        frequency = attr(object, "ts")$frequency, 
        names = colnames(object$alpha))
      sd_alpha <- vapply(alpha, function(x) apply(x, 2, sd), 
        numeric(nrow(object$alpha)))
      
      if (return_se) {
        # se_alpha <- vapply(alpha, function(x) 
        #   apply(x, 2, function(z) 
        #     sqrt(spectrum0.ar(z)$spec / length(z))), 
        #   numeric(nrow(object$alpha)))
        se_alpha <- vapply(alpha, function(x) 
          apply(x, 2, function(z) 
            sqrt(apply(theta, 2, function(x) iact(z) * var(z) / length(z)))), 
          numeric(nrow(object$alpha)))
        
        ess_alpha <- (sd_alpha / se_alpha)^2
        summary_alpha <- list(
          "Mean" = mean_alpha, "SD" = sd_alpha, 
          "SE" = se_alpha, "ESS" = ess_alpha)
      } else {
        summary_alpha <- list("Mean" = mean_alpha, "SD" = sd_alpha)
      }
    }
  }
  switch(variable,
    "both" = return(list(theta = summary_theta, states = summary_alpha)),
    "theta" = return(summary_theta),
    "states" = return(summary_alpha)
  )
}
