context("Test Gaussian approximation")

test_that("results for poisson GLM are equal to glm function",{
  d <- data.frame(treatment = gl(3,3), outcome = gl(3,1,9), counts = c(18,17,15,20,10,20,25,13,12))
  glm_poisson <- glm(counts ~ outcome + treatment, data = d, family = poisson())
  xreg <- model.matrix(~ outcome + treatment, data = d)
  expect_error(model_poisson <- ngssm(d$counts, Z = t(xreg), T = diag(5), R = diag(0, 5), 
    P1 = diag(1e7, 5), distribution = 'poisson', state_names = colnames(xreg)), NA)
  expect_error(sm <- smoother(model_poisson), NA)
  expect_equal(sm$alphahat[1,], coef(glm_poisson))
  expect_equal(sm$V[,,1], vcov(glm_poisson))
  
  expect_equivalent(c(gaussian_approx(model_poisson)$signal), glm_poisson$linear.predictors)
  xreg <- model.matrix(~ outcome + treatment, data = d)[, -1]
  expect_error(model_poisson <- ng_bsm(d$counts, sd_level = 0, xreg = xreg, 
    beta = normal(coef(glm_poisson)[-1], 0, 10), distribution = 'poisson'), NA)
  expect_equivalent(smoother(model_poisson)$alphahat[1,], coef(glm_poisson)[1])
})

test_that("results for binomial GLM are equal to glm function",{
  
  ldose <- rep(0:5, 2)
  numdead <- c(1, 4, 9, 13, 18, 20, 0, 2, 6, 10, 12, 16)
  sex <- factor(rep(c("M", "F"), c(6, 6)))
  SF <- cbind(numdead, numalive = 20-numdead)
  glm_binomial <- glm(SF ~ sex * ldose, family = binomial)
  xreg <- model.matrix(~  sex * ldose)
  expect_error(model_binomial <- ngssm(numdead, Z = t(xreg), T = diag(4), R = diag(0, 4), P1 = diag(1e5, 4), 
    distribution = 'binomial', phi = 20, state_names = colnames(xreg)), NA)
  expect_error(sm <- smoother(model_binomial), NA)
  # non-exact diffuse initialization is numerically difficult...
  expect_equal(sm$alphahat[1, ], coef(glm_binomial), tolerance = 1e-5)
  expect_equal(sm$V[, , 1], vcov(glm_binomial), tolerance = 1e-4)
  
})
