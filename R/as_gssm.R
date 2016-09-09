#' Convert SSModel Object to gssm or ngssm Object
#'
#' Converts univariate \code{SSModel} object of \code{KFAS} package to 
#' \code{gssm} or \code{ngssm} object.
#'
#' @param model Object of class \code{SSModel}.
#' @param kappa For \code{SSModel} object, a prior variance for initial state
#' used to replace exact diffuse elements of the original model.
#' @return Object of class \code{gssm} or \code{ngssm}.
#' @rdname as_gssm
#' @export
as_gssm <- function(model, kappa = 1e5) {

  if (!requireNamespace("KFAS", quietly = TRUE)) {
    stop("This function depends on the KFAS package. ", call. = FALSE)
  }

  if (attr(model, "p") > 1) {
    stop("Only univariate time series are supported.")
  }

  if (model$distribution != "gaussian") {
    stop("SSModel object contains non-Gaussian series.")
  }

  model$P1[model$P1inf > 0] <- kappa
  model$H <- sqrt(c(model$H))

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
structure(list(y = model$y, Z = Z, H = model$H, T = model$T,
  R = R, a1 = c(model$a1), P1 = model$P1, xreg = matrix(0, 0, 0),
  beta = numeric(0)), class = "gssm")
}

#' @rdname as_gssm
#' @inheritParams as_gssm
#' @export
as_ngssm <- function(model, kappa = 1e5) {

  if (!requireNamespace("KFAS", quietly = TRUE)) {
    stop("This function depends on the KFAS package. ", call. = FALSE)
  }
  if (attr(model, "p") > 1) {
    stop("Only univariate time series are supported.")
  }
  if (model$distribution == "gaussian") {
    stop("SSModel object is Gaussian, call as_gssm instead.")
  }
  if (model$distribution == "gamma") {
    stop("Gamma distribution is not yet supported.")
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
  structure(list(y = model$y, Z = Z, T = model$T,
    R = R, a1 = c(model$a1), P1 = model$P1, phi = c(model$u), xreg = matrix(0, 0, 0),
    beta = numeric(0), distribution = model$distribution), class = "ngssm")
}
