check_y <- function(x) {

  if ((!is.numeric(x) && !all(is.na(x))) || (!is.null(dim(x)[2]) && dim(x)[2] > 1)) {
    stop("Argument y must be a numeric vector or a univariate time series object.")
  }
  if (any(is.infinite(x))) {
    stop("Argument y must contain only finite or NA values.")
  }
  if (length(x) < 2) {
    stop("Length of argument y must be at least two.")
  }

}

check_sd <- function(x, type, add_prefix = TRUE) {

  if (add_prefix) {
    param <- paste0("sd_", type)
  } else {
    param <- type
  }
  if (length(x) != 1) {
    stop(paste0("Argument ", param, " must be of length one."))
  }
  if (!is.numeric(x)) {
    stop(paste0("Argument ", param, " must be numeric."))
  }
  if (length(x) < 0) {
    stop(paste0("Argument ", param, " must be non-negative."))
  }
  if (is.infinite(x)) {
    stop(paste0("Argument ", param, " must be finite."))
  }

}

check_xreg <- function(x, n) {

  if (nrow(x) != n) {
    stop("Number of rows in xreg is not equal to the length of the series y.")
  }
  if (any(!is.finite(x))) {
    stop("Argument xreg must contain only finite values. ")
  }

}

check_beta <- function(x, k) {

  if (length(x) != k) {
    stop("Number of coefficients in beta is not equal to the number of columns of xreg.")
  }
  if (any(!is.finite(x))) {
    stop("Argument beta must contain only finite values. ")
  }

}
check_ar <- function(x) {
  
  if (length(x) != 1) {
    stop(paste0("Argument ar must be of length one."))
  }
  if (abs(x) >= 1) {
    stop("Argument ar must be strictly between -1 and 1.")
  }
  
}

