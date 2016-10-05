

#' @importFrom stats qlogis
initial_signal <- function(y, u, distribution) {
  
  if (distribution == "poisson") {
    y <- y / u
    y[y < 0.1 | is.na(y)] <- 0.1
    y <- log(y)
  }
  if (distribution == "binomial") {
    y <- qlogis((ifelse(is.na(y), 0.5, y) + 0.5)/(u + 1))
  }
  if (distribution == "gamma") {
    y[is.na(y) | y < 1] <- 1
    y <- log(y)
  }
  if (distribution == "negative binomial") {
    y[is.na(y) | y < 1/6] <- 1/6
    y <- log(y)
  }
  y
}