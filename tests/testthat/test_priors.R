context("Test rest of warnings and errors")

#' @srrstats {G5.2, G5.2a, G5.2b} Test the rest of the warnings that are not 
#' already triggered otherwise.
#' 
test_that("priors give errors with wrong arguments", {
  expect_error(normal("a", 0, 1))
  expect_error(uniform(1, 2, 0))
  expect_error(uniform(2, 0, 1))
  expect_error(normal(0, 0, -1))
  expect_error(halfnormal(0, -1))
  expect_error(halfnormal(-1, 0, 1))
  expect_error(tnormal(0, 0, -1))
  expect_error(tnormal(10, 0, 4, 0, 5))
  expect_error(gamma("a", 2, 1))
  expect_error(gamma(1, 0, 1))
  expect_error(gamma(1, -1, 1))
  expect_error(gamma(1, 2, 0))
})

