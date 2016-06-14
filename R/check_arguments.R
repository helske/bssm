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

check_sd <- function(x, type) {
  if (length(x) != 1) {
    stop(paste0("Argument sd_",type," must be of length one."))
  }
  if (!is.numeric(x)) {
    stop(paste0("Argument sd_",type," must be numeric."))
  }
  if (length(x) < 0) {
    stop(paste0("Argument sd_",type," must be non-negative."))
  }
  if (is.infinite(x)) {
    stop(paste0("Argument sd_",type," must be finite."))
  }
}
