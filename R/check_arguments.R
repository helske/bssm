
check_y <- function(x, multivariate = FALSE, distribution = "gaussian") {
  
  if(multivariate) {
    if (!is.matrix(x)) {
      stop("Argument y must be a numeric matrix or multivariate ts object.")
    }
  } else {
    if (!(is.vector(x) && !is.list(x)) && !is.numeric(x)) {
      stop("Argument y must be a numeric vector or ts object.")
    }
    if(distribution != "gaussian" && any(x < 0)) {
      stop(paste0("Negative values not allowed for ", distribution, " distribution. "))
    }
  }
  if (any(is.infinite(x))) {
    stop("Argument y must contain only finite or NA values.")
  }
  if (length(x) < 2) {
    stop("Length of argument y must be at least two.")
  }
  
  
}

check_distribution <- function(x, distribution) {
  for(i in 1:ncol(x)) {
    if(distribution[i] != "gaussian" && any(x[,i] < 0)) {
      stop(paste0("Negative values not allowed for ", distribution[i], " distribution. "))
    }
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
  if (x < 0) {
    stop(paste0("Argument ", param, " must be non-negative."))
  }
  if (is.infinite(x)) {
    stop(paste0("Argument ", param, " must be finite."))
  }
  
}

check_xreg <- function(x, n) {
  
  if (!(nrow(x) %in% c(0, n))) {
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
check_u <- function(x, multivariate = FALSE) {
  if (any(x < 0)) {
    stop("All values of 'u' must be non-negative.")
  }
  if(multivariate) {
    if (!is.matrix(x) && !is.numeric(x)) {
      stop("Argument 'u' must be a numeric matrix or multivariate ts object.")
    }
  } else {
    if (!(is.vector(x) && !is.list(x)) && !is.numeric(x)) {
      stop("Argument 'u' must be a numeric vector or ts object.")
    }
  }
  if (any(is.infinite(x))) {
    stop("Argument 'u' must contain only finite values.")
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

check_D <- function(x, p, n) {
  if (is.null(dim(x)) || nrow(x) != p || !(ncol(x) %in% c(1,n))) {
    stop("'D' must be p x 1 or p x n matrix, where p is the number of series.")
  } 
}

check_C <- function(x, m, n) {
  if (is.null(dim(x)) || nrow(x) != m || !(ncol(x) %in% c(1,n))) {
    stop("'C' must be m x 1 or m x n matrix, where m is the number of states.")
  } 
}
