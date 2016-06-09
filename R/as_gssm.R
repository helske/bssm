#' @method as_gssm SSModel
#' @rdname as_gssm
#' @export
as_gssm.SSModel <- function(model, kappa = 1e5, ...) {

  if (!requireNamespace("KFAS", quietly = TRUE)) {
    stop("This function depends on the KFAS package. ", call. = FALSE)
  }

  if (any(model$distribution != "gaussian")) {
    stop("SSModel object contains non-Gaussian series.")
  }
  if (attr(model, "p") > 1) {
    stop("Only univariate time series are supported.")
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

structure(list(y = model$y, Z = model$Z, H = model$H, T = model$T,
  R = R, a1 = model$a1, P1 = model$P1, xreg = matrix(0, 0, 0),
  beta = numeric(0)), class = "gssm")
}
