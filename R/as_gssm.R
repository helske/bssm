#' Convert SSModel Object to ssm_mlg or ssm_mng Object
#'
#' Converts \code{SSModel} object of \code{KFAS} package to 
#' \code{ssm_mlg} or \code{ssm_mng} object.
#' 
#' @note These functions are included mostly in order to compare some of 
#' the implementations between \code{KFAS} and \code{bssm}, and thus they
#' might become deprecated in the future.
#' 
#' @param model Object of class \code{SSModel}.
#' @param kappa For \code{SSModel} object, a prior variance for initial state
#' used to replace exact diffuse elements of the original model.
#' @param ... Additional arguments to \code{ssm_mlg} and \code{ssm_mng} 
#' (such as prior and updating functions).
#' @return Object of class \code{ssm_mlg} or \code{ssm_mng}.
#' @rdname as_ssm_ulg
#' @export
as_ssm_mlg <- function(model, kappa = 1e5, ...) {
  
  if (!requireNamespace("KFAS", quietly = TRUE)) {
    stop("This function depends on the KFAS package. ", call. = FALSE)
  }
  
  if (any(model$distribution != "gaussian")) {
    stop("SSModel object contains non-Gaussian series. Call as_ssm_mng instead")
  }
  
  model$P1[model$P1inf > 0] <- kappa
  
  tvr <- dim(model$R)[3] > 1
  tvq <- dim(model$Q)[3] > 1
  tvrq <- max(tvr, tvq)
  
  R <- array(0, c(dim(model$R)[1:2], tvrq * (nrow(model$y) - 1) + 1))
  
  if (dim(model$R)[2] > 1) {
    for (i in 1:dim(R)[3]) {
      L <- KFAS::ldl(model$Q[, , (i - 1) * tvq + 1])
      D <- sqrt(diag(diag(L)))
      diag(L) <- 1
      R[, , i] <- model$R[, , (i - 1) * tvr + 1] %*% L %*% D
    }
  } else {
    R <- model$R * sqrt(c(model$Q))
  }
  
  if (attr(model, "p") == 1) {
    model$H <- sqrt(c(model$H))
    Z <- aperm(model$Z, c(2, 3, 1))
    dim(Z) <- dim(Z)[1:2]
    out <- ssm_ulg(y = model$y, Z = Z, H = model$H, T = model$T, R = R, 
      a1 = c(model$a1), P1 = model$P1, state_names = rownames(model$a1), ...)
  } else {
    H <- model$H
    for (i in 1:dim(H)[3]) {
      L <- KFAS::ldl(model$H[, , i])
      D <- sqrt(diag(diag(L)))
      diag(L) <- 1
      H[, , i] <- L %*% D
    }
    out <- ssm_mlg(y = model$y, Z = model$Z, H = H, T = model$T, R = R, 
      a1 = c(model$a1), P1 = model$P1, state_names = rownames(model$a1), ...)
  }
  
  out
}

#' @rdname as_ssm_mlg
#' @inheritParams as_ssm_mlg
#' @export
as_ssm_ung <- function(model, kappa = 1e5, ...) {
  
  if (!requireNamespace("KFAS", quietly = TRUE)) {
    stop("This function depends on the KFAS package. ", call. = FALSE)
  }

  if (model$distribution == "gaussian") {
    stop("SSModel object is Gaussian, call as_ssm_mlg instead.")
  }
  if (model$distribution == "gamma") {
    stop("Gamma distribution is not yet supported.")
  }
  if (model$distribution == "negative binomial" && length(unique(model$u)) > 1) {
    stop("Time-varying dispersion parameter for negative binomial is not supported in 'bssm'.")
  }
  model$P1[model$P1inf > 0] <- kappa
  
  tvr <- dim(model$R)[3] > 1
  tvq <- dim(model$Q)[3] > 1
  tvrq <- max(tvr, tvq)
  
  R <- array(0, c(dim(model$R)[1:2], tvrq * (nrow(model$y) - 1) + 1))
  
  if (dim(model$R)[2] > 1) {
    for (i in 1:dim(R)[3]) {
      L <- KFAS::ldl(model$Q[, , (i - 1) * tvq + 1])
      D <- sqrt(diag(diag(L)))
      diag(L) <- 1
      R[, , i] <- model$R[, , (i - 1) * tvr + 1] %*% L %*% D
    }
  } else {
    R <- model$R * sqrt(c(model$Q))
  }
  Z <- aperm(model$Z, c(2, 3, 1))
  dim(Z) <- dim(Z)[1:2]
  
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
      u <- rep(1, length(model$u))
    },
    "negative binomial" = {
      phi <- model$u[1]
      u <- rep(1, length(model$u))
    })
  if(!missing(phi_prior)) phi <- phi_prior
  ssm_ung(y = model$y, Z = Z, T = model$T, R = R, a1 = c(model$a1), 
    P1 = model$P1, phi = phi, u = u, 
    distribution = model$distribution, state_names = rownames(model$a1), 
    ...)
}
