
#' @importFrom stats qlogis
init_mode <- function(y, u, distribution) {

  switch(distribution,
    poisson = {
      y <- y / u
      y[y < 0.1 | is.na(y)] <- 0.1
      y <- log(y)
    },
    binomial = {
      y <- qlogis((ifelse(is.na(y), 0.5, y) + 0.5) / (u + 1))
    },
    gamma = {
      y <- y / u
      y[is.na(y) | y < 1] <- 1
      y <- log(y)
    },
    "negative binomial" = {
      y <- y / u
      y[is.na(y) | y < 1 / 6] <- 1 / 6
      y <- log(y)
    },
    gaussian = {
      
    },
    stop(paste("Argument distribution must be 'poisson', 'binomial', 'gamma',",
    "'gaussian', or 'negative binomial'.", sep = " "))
  )
  y
}
