
#' @importFrom stats qlogis
init_mode <- function(y, u, distribution, xbeta = NULL) {

  switch(distribution,
    poisson = {
      y <- y / u
      y[y < 0.1 | is.na(y)] <- 0.1
      y <- log(y)
    },
    binomial = {
      y <- qlogis((ifelse(is.na(y), 0.5, y) + 0.5)/(u + 1))
    },
    gamma = {
      y[is.na(y) | y < 1] <- 1
      y <- log(y)
    },
    "negative binomial" = {
      y[is.na(y) | y < 1/6] <- 1/6
      y <- log(y)
    },
    stop("Argument distribution must be 'poisson', 'binomial', 'gamma' or 'negative binomial'.")
  )
  if (!is.null(xbeta)) y <- y - xbeta
  y
}
