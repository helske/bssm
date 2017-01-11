check_y <- function(x, distribution = "gaussian") {
  
  if ((!is.numeric(x) && !all(is.na(x))) || (!is.null(dim(x)[2]) && dim(x)[2] > 1)) {
    stop("Argument y must be a numeric vector or a univariate time series object.")
  }
  if (any(is.infinite(x))) {
    stop("Argument y must contain only finite or NA values.")
  }
  if (length(x) < 2) {
    stop("Length of argument y must be at least two.")
  }
  
  if (distribution != "gaussian") {
    if (!isTRUE(all(y == floor(y)))) stop("'y' must contain only non-negative integer values.")
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
    stop("Argument 'beta' must contain only finite values. ")
  }
  
}

check_mu <- function(x) {
  
  if (length(x) != 1) {
    stop(paste0("Argument 'mu' must be of length one."))
  }
  if (any(!is.finite(x))) {
    stop("Argument 'mu' must contain only finite values. ")
  }
  
}
check_rho <- function(x) {
  
  if (length(x) != 1) {
    stop(paste0("Argument 'rho' must be of length one."))
  }
  if (abs(x) >= 1) {
    stop("Argument 'rho' must be strictly between -1 and 1.")
  }
  
}
check_phi <- function(x, distribution) {
  if (x < 0) {
    stop("Parameter 'phi' must be non-negative.")
  }
}
check_u <- function(x) {
  if (any(x < 0)) {
    stop("All values of 'u' must be non-negative.")
  }
}
check_prior <- function(x, name) {
  if (!is_prior(x) && !is_prior_list(x)) {
    stop(paste(name, "must be of class 'bssm_prior' or 'bssm_prior_list'."))
  }
}

check_target <- function(target) {
  if(length(target) > 1 || target >= 1 || target <= 0) {
    stop("Argument 'target' must be on interval (0, 1).")
  }
}
