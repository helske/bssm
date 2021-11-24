#' Convert KFAS Model to bssm Model
#'
#' Converts \code{SSModel} object of \code{KFAS} package to general \code{bssm} 
#' model of type \code{ssm_ulg}, \code{ssm_mlg}, \code{ssm_ung} or 
#' \code{ssm_mng}. As \code{KFAS} supports formula syntax for defining 
#' e.g. regression and cyclic components it maybe sometimes easier to define 
#' the model with \code{KFAS::SSModel} and then convert for the bssm style with 
#' \code{as_bssm}. 
#' 
#' @param model Object of class \code{SSModel}.
#' @param kappa For \code{SSModel} object, a prior variance for initial state
#' used to replace exact diffuse elements of the original model.
#' @param ... Additional arguments to model building functions of \code{bssm}
#' (such as prior and updating functions, C, and D).
#' @return An object of class \code{ssm_ulg}, \code{ssm_mlg}, \code{ssm_ung} or 
#' \code{ssm_mng}.
#' @export
#' @examples
#' library("KFAS")
#'   model_KFAS <- SSModel(Nile ~
#'     SSMtrend(1, Q = 2, P1 = 1e4), H = 2)
#'   model_bssm <- as_bssm(model_KFAS)  
#'   logLik(model_KFAS)
#'   logLik(model_bssm)
#' 
as_bssm <- function(model, kappa = 100, ...) {
  
  kappa <- check_positive_real(kappa, "kappa")
  if (!requireNamespace("KFAS", quietly = TRUE)) {
    stop("This function depends on the KFAS package. ", call. = FALSE)
  }
  
  model$P1[model$P1inf > 0] <- kappa
  
  tvr <- dim(model$R)[3] > 1
  tvq <- dim(model$Q)[3] > 1
  tvrq <- max(tvr, tvq)
  
  R <- array(0, c(dim(model$R)[1:2], tvrq * (nrow(model$y) - 1) + 1))
  
  if (dim(model$R)[2] > 1) {
    for (i in 1:dim(R)[3]) {
      L <- KFAS::ldl(model$Q[, , (i - 1) * tvq + 1])
      d <- sqrt(diag(diag(L)))
      diag(L) <- 1
      R[, , i] <- model$R[, , (i - 1) * tvr + 1] %*% L %*% d
    }
  } else {
    R <- model$R * sqrt(c(model$Q))
  }
  if (attr(model, "p") == 1L) {
    Z <- aperm(model$Z, c(2, 3, 1))
    dim(Z) <- dim(Z)[1:2]
  } else {
    Z <- model$Z
  }
  
  if (any(model$distribution != "gaussian")) {
    if (attr(model, "p") == 1L) {
      if (model$distribution == "negative binomial" && 
          length(unique(model$u)) > 1L) {
        stop(paste("Time-varying dispersion parameter for negative binomial",
        "is not (yet) supported in 'bssm'.", sep = " "))
      } 
      if (model$distribution == "gamma" && length(unique(model$u)) > 1L) {
        stop(paste("Time-varying shape parameter for gamma is not (yet)",
        "supported in 'bssm'.", sep = " "))
      }
      
      switch(model$distribution,
        poisson = {
          phi <- 1
          u <- model$u
        },
        binomial = {
          phi <- 1
          u <- model$u
        },
        gamma = {
          phi <- model$u[1]
          u <- rep(1, length(model$u))  # no exposure for Gamma in KFAS
        },
        "negative binomial" = {
          phi <- model$u[1]
          u <- rep(1, length(model$u))  # no exposure for NB in KFAS
        })
      out <- ssm_ung(y = model$y, Z = Z, T = model$T, R = R, a1 = c(model$a1), 
        P1 = model$P1, phi = phi, u = u, 
        distribution = model$distribution, state_names = rownames(model$a1), 
        ...)
    } else {
      phi <- numeric(attr(model, "p"))
      u <- model$u
      for (i in 1:attr(model, "p")) {
        switch(model$distribution[i],
          poisson = {
            phi[i] <- 1
            u[, i] <- model$u[, i]
          },
          binomial = {
            phi[i] <- 1
            u[, i] <- model$u[, i]
          },
          gamma = {
            if (length(unique(model$u[, i])) > 1)
              stop(paste0("Time-varying shape parameter for gamma is not",
              "(yet) supported in 'bssm'.", sep = " "))
            phi[i] <- model$u[1, i]
            u[, i] <- 1 # no exposure for Gamma in KFAS
          },
          "negative binomial" = {
            if (length(unique(model$u[, i])) > 1)
              stop(paste("Time-varying dispersion parameter for negative",
                "binomial is not (yet) supported in 'bssm'.", sep = " "))
            phi[i] <- model$u[1, i]
            u[, i] <- 1 # no exposure for NB in KFAS
          }, 
          gaussian = {
            if (length(unique(model$u[, i])) > 1)
              stop(paste("Time-varying standard deviation for gaussian", 
              "distribution with non-gaussian series is not supported",
              "in 'bssm'.", sep = " "))
            phi[i] <- sqrt(model$u[1, i])
            u[, i] <- 1
          })
      }
      
      out <- ssm_mng(y = model$y, Z = Z, T = model$T, R = R, a1 = c(model$a1), 
        P1 = model$P1, phi = phi, u = u, 
        distribution = model$distribution, state_names = rownames(model$a1), 
        ...)
    }
    
  } else {
    if (attr(model, "p") == 1L) {
      out <- ssm_ulg(y = model$y, Z = Z, H = sqrt(c(model$H)), T = model$T, 
        R = R, 
        a1 = c(model$a1), P1 = model$P1, state_names = rownames(model$a1), ...)
    } else {
      H <- model$H
      for (i in 1:dim(H)[3]) {
        L <- KFAS::ldl(model$H[, , i])
        d <- sqrt(diag(diag(L)))
        diag(L) <- 1
        H[, , i] <- L %*% d
      }
      
      out <- ssm_mlg(y = model$y, Z = Z, H = H, T = model$T, R = R, 
        a1 = c(model$a1), P1 = model$P1, state_names = rownames(model$a1), ...)
    }
  }
  
  out
}
