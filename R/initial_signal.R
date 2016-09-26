

#' @importFrom stats qlogis
initial_signal <- function(y, phi, distribution) {
  
  if (distribution == "poisson") {
    y <- y/phi
    y[y < 0.1 | is.na(y)] <- 0.1
    y <- log(y)
  }
  if (distribution == "binomial") {
    y <- qlogis((ifelse(is.na(y), 0.5, y) + 0.5)/(phi + 1))
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