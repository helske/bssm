

#' @importFrom stats qlogis
initial_signal <- function(y, u, distribution) {
  
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
    })
  y
}
