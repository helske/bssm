context("Test as.data.frame")

set.seed(123)
model_bssm <- bsm_lg(rnorm(10, 3), P1 = diag(2, 2), sd_slope = 0,
  sd_y = uniform(1, 0, 10), 
  sd_level = uniform(1, 0, 10))

test_that("expanded and not expanded data frame work equally for theta", {

  expect_error(mcmc_bsm <- run_mcmc(model_bssm, iter = 50, seed = 1), NA)
  d <- expect_error(as.data.frame(mcmc_bsm), NA)
  expect_equal(colnames(d), c("iter", "value", "variable", "weight"))
  expect_equal(unique(d$variable), c("sd_y", "sd_level"))
  expect_equal(mean(d$value[d$variable == "sd_level"]), 
    weighted.mean(mcmc_bsm$theta[, 2], mcmc_bsm$counts))
  
  d <- expect_error(as.data.frame(mcmc_bsm, expand = FALSE), NA)
  expect_equal(colnames(d), c("iter", "value", "variable", "weight"))
  expect_equal(unique(d$variable), c("sd_y", "sd_level"))
  expect_equal(weighted.mean(d$value[d$variable == "sd_level"], 
    d$weight[d$variable == "sd_level"]), 
    weighted.mean(mcmc_bsm$theta[, 2], mcmc_bsm$counts))
})

test_that("expanded and not expanded data frame work equally for states", {
  
  expect_error(mcmc_bsm <- run_mcmc(model_bssm, iter = 50, seed = 1), NA)
  d <- expect_error(as.data.frame(mcmc_bsm, variable = "state"), NA)
  expect_equal(colnames(d), c("iter", "value", "variable", "time", "weight"))
  expect_equal(unique(d$variable), c("level", "slope"))
  expect_equal(mean(d$value[d$variable == "slope" & d$time == 3]), 
    weighted.mean(mcmc_bsm$alpha[3, 2, ], mcmc_bsm$counts))
  
  expect_error(d <- as.data.frame(mcmc_bsm, variable = "state", 
    expand = FALSE), NA)
  expect_equal(colnames(d), c("iter", "value", "variable", "time", "weight"))
  expect_equal(unique(d$variable), c("level", "slope"))
  expect_equal(weighted.mean(d$value[d$variable == "slope" & d$time == 3],
    d$weight[d$variable == "slope" & d$time == 3]), 
    weighted.mean(mcmc_bsm$alpha[3, 2, ], mcmc_bsm$counts))
  
  expect_error(d <- as.data.frame(mcmc_bsm, variable = "theta"), NA)
  expect_error(sumr <- summary(mcmc_bsm, variable = "both", return_se = TRUE), 
    NA)
  expect_equal(mean(d$value[d$variable == "sd_y"]), 
    sumr$theta[2, 2])
  
})
