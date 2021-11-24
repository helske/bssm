#' Check Arguments
#' 
#' @importFrom checkmate test_atomic_vector test_count test_double test_flag 
#' test_integerish test_int
#' 
#' @param x Variable to be checked.
#' @param name Name of the argument used in printing error messages.
#' @param positive Logical, check for positiveness of \code{x}.
#' @param max Maximum value of \code{x}.
#' @param p An integer, number of time series.
#' @param n An integer, number of time points.
#' @param m An integer, dimensionality of the state vector.
#' @param k An integer, number of predictors.
#' @param multivariate Logical, should \code{p} be larger than 1?
#' @param beta A vector of regression coefficients.
#' @param xreg A matrix or vector of predictors.
#' @param distribution Distribution(s) of the responses.
#' @param y The response time series.
#' @param type Name to be added to the sd parameter name.
#' @param add_prefix Logical, add \code{type} to parameter name.
#' @noRd
check_y <- function(x, multivariate = FALSE, distribution = "gaussian") {
  if (any(!is.na(x))) {
    if (multivariate) {
      if (!is.matrix(x)) {
        stop("Argument 'y' must be a matrix or multivariate ts object.")
      }
      if (nrow(x) < 2) {
        stop("Number of rows in 'y', i.e. number of time points, must be > 1. ")
      }
    } else {
      if (!is.vector(x) || is.list(x)) {
        if (is.ts(x) || is.matrix(x)) {
          if (!is.null(dim(x)) && ncol(x) == 1 && length(dim(x)) < 3) {
            dim(x) <- NULL
          } else {
            if(!is.null(dim(x)) && ncol(x) > 1) {
              stop("Argument 'y' must be a vector or univariate ts object.")
            }
          }
        } else {
          stop("Argument 'y' must be a vector or univariate ts object.")
        }
      }
      if (length(x) < 2) {
        stop("Length of argument y, i.e. number of time points, must be > 1.")
      }
      if (distribution != "gaussian" && any(na.omit(x) < 0)) {
        stop(paste0("Negative values not allowed for ", distribution, 
          " distribution. "))
      } else {
        if (distribution %in% 
            c("negative binomial", "binomial", "poisson") && 
            any(na.omit(x[is.finite(x)] != as.integer(x[is.finite(x)])))) {
          stop(paste0("Non-integer values not allowed for ", distribution, 
            " distribution. "))
        }
      }
    }
    if (any(is.infinite(x))) {
      stop("Argument 'y' must contain only finite or NA values.")
    }
  }
  x
}

check_period <- function(x, n) {
  if (!test_int(x)) {
    stop("Argument 'period' should be a single integer. ")
  } else {
    if (x < 3) {
      stop("Argument 'period' should be a integer larger than 2. ")
    }
    if (x >= n) {
      stop("Period should be less than the number of time points.")
    }
  }
  as.integer(x)
}
#' @srrstats {BS2.5} Checks that observations are compatible with their 
#' distributions are made.
check_distribution <- function(x, distribution) {
  for (i in seq_len(ncol(x))) {
    if (distribution[i] != "gaussian" && any(na.omit(x[, i]) < 0)) {
      stop(paste0("Negative values not allowed for ", distribution[i], 
        " distribution. "))
    } else {
      if (distribution[i] %in% 
          c("negative binomial", "binomial", "poisson") && 
          any(na.omit(x[, i] != as.integer(x[, i])))) {
        stop(paste0("Non-integer values not allowed for ", distribution[i], 
          " distribution. "))
      }
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
    stop(paste0("Argument ", param, 
      " must be of length one (scalar or bssm_prior)."))
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
  if(!is.numeric(x)) stop("'beta' must be numeric. ")
  if (length(x) != k) {
    stop(paste("Number of coefficients in beta is not equal to the number",
      "of columns of xreg.", sep = " "))
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


check_phi <- function(x) {
  if (x < 0) {
    stop("Parameter 'phi' must be non-negative.")
  }
}

check_u <- function(x, y, multivariate = FALSE) {
  if (any(x < 0)) {
    stop("All values of 'u' must be non-negative.")
  }
  if (multivariate) {
    if (length(x) == 1) x <- matrix(x, nrow(y), ncol(y))
    
    if (!is.matrix(x) && !is.numeric(x)) {
      stop("Argument 'u' must be a numeric matrix or multivariate ts object.")
    }
    if(!identical(dim(y), dim(x))) 
      stop("Dimensions of 'y' and 'u' do not match. ")
  } else {
    if (length(x) == 1) x <- rep(x, length(y))
    if (!(is.vector(x) && !is.list(x)) && !is.numeric(x)) {
      stop("Argument 'u' must be a numeric vector or ts object.")
    }
    if (length(x) != length(y))
      stop("Lengths of 'u' and 'y' do not match.")
    dim(x) <- NULL
  }
  if (any(is.infinite(x))) {
    stop("Argument 'u' must contain only finite values.")
  }
  x
}

check_prior <- function(x, name) {
  if (!is_prior(x) && !is_prior_list(x)) {
    stop(paste(name, "must be of class 'bssm_prior' or 'bssm_prior_list'."))
  }
}

check_prop <- function(x, name = "target") {
  if (length(x) > 1 || x >= 1 || x <= 0) {
    stop(paste0("Argument '", name, "' must be on interval (0, 1)."))
  }
}

check_D <- function(x, p, n) {
  if (missing(x) || is.null(x)) {
    x <- if (p == 1) 0 else matrix(0, p, 1)
  } else {
    if(!is.numeric(x)) stop("'D' must be numeric. ")
    if (p == 1) {
      if (!(length(x) %in% c(1, n))) {
        stop(paste("'D' must be a scalar or length n, where n is the number of",
          "observations.", sep = " "))
        x <- as.numeric(x)
      } 
    } else {
      if (is.null(dim(x)) || nrow(x) != p || !(ncol(x) %in% c(1, n))) {
        stop(paste("'D' must be p x 1 or p x n matrix, where p is the number",
          "of series.", sep = " "))
      } 
    }
  }
  x
}

check_C <- function(x, m, n) {
  if (missing(x) || is.null(x)) {
    x <- matrix(0, m, 1)
  } else {
    if(!is.numeric(x)) stop("'C' must be numeric. ")
    if (is.null(dim(x)) || nrow(x) != m || !(ncol(x) %in% c(1, n))) {
      stop(paste("'C' must be m x 1 or m x n matrix, where m is", 
        "the number of states.", sep = " "))
    } 
  }
  x
}

  
  
  

create_regression <- function(beta, xreg, n) {
  if (missing(xreg) || is.null(xreg)) {
    list(xreg = matrix(0, 0, 0), coefs = numeric(0), beta = NULL)
  } else {
    if (missing(beta) || is.null(beta)) {
      stop("No prior defined for beta. ")
    } else {
      if (!is_prior(beta) && !is_prior_list(beta)) {
        stop(paste("Prior for beta must be of class 'bssm_prior' or", 
          "'bssm_prior_list.", sep = " " ))
      } else {
        if (is.null(dim(xreg))) {
          if (length(xreg) == n) {
            dim(xreg) <- c(n, 1)
          } else {
            stop("Length of xreg is not equal to the length of the series y.")
          }
        }
        check_xreg(xreg, n)
        nx <- ncol(xreg)
        if (nx == 1 && is_prior_list(beta)) beta <- beta[[1]]
        if (nx > 1) {
          coefs <- vapply(beta, "[[", "init", FUN.VALUE = 1)
        } else {
          coefs <- beta$init
        }
        check_beta(coefs, nx)
        if (nx > 0 && is.null(colnames(xreg))) {
          colnames(xreg) <- paste0("coef_", seq_len(ncol(xreg)))
        }
        names(coefs) <- colnames(xreg)
      }
    }
    list(xreg = xreg, coefs = coefs, beta = beta)
  }
}

check_Z <- function(x, p, n, multivariate = FALSE) {
  if(!is.numeric(x)) stop("'Z' must be numeric. ")
  if (!multivariate) {
    if (length(x) == 1) {
      dim(x) <- c(1, 1)
    } else {
      if (!(dim(x)[2] %in% c(1, NA, n))) {
        stop(paste("'Z' must be a (m x 1) or (m x n) matrix, where",
          "m is the number of states and n is the length of the series. ",
          sep = " "))
      } else {
        dim(x) <- 
          c(dim(x)[1], (n - 1) * (max(dim(x)[2], 0, na.rm = TRUE) > 1) + 1)
      }
    } 
  } else {
    if(p == 1 && length(x) == 1) {
      dim(x) <- c(1, 1, 1)
    } else {
      if (is.null(dim(x)) || dim(x)[1] != p || !(dim(x)[3] %in% c(1, NA, n))) {
        stop(paste("'Z' must be a (p x m) matrix or (p x m x n) array",
          "where p is the number of series, m is the number of states,", 
          "and n is the length of the series. ", sep = " "))
      }
    }
    dim(x) <- 
      c(p, dim(x)[2], (n - 1) * (max(dim(x)[3], 0, na.rm = TRUE) > 1) + 1)
  }
  x
}

check_T <- function(x, m, n) {
  if(!is.numeric(x)) stop("'T' must be numeric. ")
  if (length(x) == 1 && m == 1) {
    dim(x) <- c(1, 1, 1)
  } else {
    if ((length(x) == 1) || any(dim(x)[1:2] != m) || 
        !(dim(x)[3] %in% c(1, NA, n))) {
      stop(paste("'T' must be a (m x m) matrix, (m x m x 1) or",
        "(m x m x n) array, where m is the number of states. ", sep = " "))
    }
    dim(x) <- c(m, m, (n - 1) * (max(dim(x)[3], 0, na.rm = TRUE) > 1) + 1)
  }
  x
}

check_R <- function(x, m, n) {
  if (length(x) == m) {
    dim(x) <- c(m, 1, 1)
  } else {
    if(!is.numeric(x)) stop("'R' must be numeric. ")
    if (!(dim(x)[1] == m) || dim(x)[2] > m || !dim(x)[3] %in% c(1, NA, n)) {
      stop(paste("'R' must be a (m x k) matrix, (m x k x 1) or", 
        "(m x k x n) array, where k<=m is the number of disturbances eta,", 
        "and m is the number of states. ", sep = " "))
    } else {
      dim(x) <- 
        c(m, dim(x)[2], (n - 1) * (max(dim(x)[3], 0, na.rm = TRUE) > 1) + 1)
    }
  }
  x
}

check_a1 <- function(x, m) {
  if (missing(x) || is.null(x)) {
    x <- numeric(m)
  } else {
    if(!is.numeric(x)) stop("'a1' must be numeric. ")
    if (length(x) == 1 || length(x) == m) {
      x <- rep(x, length.out = m)
    } else {
      stop(paste("Misspecified a1, argument a1 must be a vector of length m,",
        "where m is the number of state_names and 1<=t<=m.", sep = " "))
    }
  }
  x
}


check_P1 <- function(x, m) {
  if (missing(x) || is.null(x)) {
    x <- matrix(0, m, m)
  } else {
    if(!is.numeric(x)) stop("'P1' must be numeric. ")
    if (length(x) == 1 && m == 1) {
      dim(x) <- c(1, 1)
    } else {
      if (!identical(dim(x), c(m, m)))
        stop(paste("Argument P1 must be (m x m) matrix, where m is the number",
          "of states. ", sep = " "))
    }
  }
  x
}


check_H <- function(x, p, n, multivariate = FALSE) {
  
  if(!is.numeric(x)) stop("'H' must be numeric. ")
  
  if (!multivariate) {
    if (!(length(x) %in% c(1, n))) {
      stop(paste("'H' must be a scalar or length n, where n is the length of",
        "the time series y", sep = " "))
    } else x <- as.numeric(x)
  } else {
    if (any(dim(x)[1:2] != p) || !(dim(x)[3] %in% c(1, n, NA))) {
      stop(paste("'H' must be p x p matrix or p x p x n array, where p is the",
        "number of series and n is the length of the series.", sep = " "))
    } else {
      dim(x) <- c(p, p, (n - 1) * (max(dim(x)[3], 0, na.rm = TRUE) > 1) + 1)
    }
  }
  x
}


check_intmax <- function(x, name = "particles", positive = TRUE, max = 1e5) {
  # autotest complains without additional positivity test
  if (!test_count(x, positive) | (positive & x <= 0)) {
    stop(paste0("Argument '", name, "' should be a ",
      ifelse(positive, "positive", "non-negative"), " integer. "))
  }
  if (x > max) {
    stop(paste0("You probably do not want '", name, "' > ", max,
      ". If you really do, please file an issue at Github. "))
  }
  as.integer(x)
}

check_positive_real <- function(x, name) {
  if (!test_double(x, lower=0, finite = TRUE, any.missing = FALSE, len = 1)) {
    stop(paste0("Argument '", name, "' should be positive real value."))
  }
  x
}

check_theta <- function(x) {
  
  if (!is.numeric(x) || !test_atomic_vector(x)) {
    stop("Argument 'theta' should be a numeric vector.")
  }
  if (is.null(names(x))) {
    names(x) <- paste("theta_", seq_len(length(x)))
  }
  x
}

check_missingness <- function(x) {
  if (!inherits(x, c("ssm_nlg", "ssm_sde"))) {
    if (is.null(x$prior_parameters)) {
      contains_na <- 
        anyNA(x[-which(names(x) %in% c("y", "update_fn", "prior_fn"))], 
          recursive = TRUE)
      if (contains_na) stop(paste(
        "Missing values not allowed in the model object", 
        "(except in component 'y')."))
    } else {
      contains_na <- anyNA(x[-which(names(x) %in% c("y", "prior_parameters"))], 
        recursive = TRUE)
      if (contains_na) stop(paste(
        "Missing values not allowed in the model object", 
        "(except in components 'y' and 'prior_parameters')."))
    }
  }
}
