context("Test Gaussian approximation")


test_that("Gaussian approximation results of bssm and KFAS coincide",{
  library(KFAS)
  set.seed(123)
  model_KFAS <- SSModel(rpois(10, exp(2)) ~ SSMtrend(2, Q = list(1, 1), 
    P1 = diag(100, 2)), distribution = "poisson")
  expect_error(model_bssm <- bsm_ng(model_KFAS$y, sd_level = 1, 
    sd_slope = 1, distribution = "poisson"), NA)
  approx_KFAS <- approxSSM(model_KFAS)
  expect_error(approx_bssm <- gaussian_approx(model_bssm), NA)
  all.equal(c(approx_bssm$H^2),c(approx_KFAS$H))
  expect_error(alphahat <- fast_smoother(approx_bssm), NA)
  expect_equivalent(KFS(approx_KFAS)$alphahat, alphahat)
  expect_equivalent(logLik(approx_KFAS), logLik(approx_bssm))
})


test_that("Gaussian approximation works for SV model",{
  set.seed(123)
  expect_error(model_bssm <- svm(rnorm(5), sigma = uniform(1,0,10), rho = uniform(0.950, 0, 1), 
    sd_ar = uniform(0.1,0,1)), NA)
  expect_error(approx_bssm <- gaussian_approx(model_bssm, max_iter = 2, conv_tol = 1e-8), NA)
  
  expect_equivalent(c(-1.47548927809174, -11.2190916117862, 
    0.263154138901814, -121.519769682058, -36.0386937004332), approx_bssm$y[1:5])
  expect_equivalent(c(2.01061310553144, 4.84658294043645, 0.712674409714633, 
    15.6217737012134, 8.54936618861792), approx_bssm$H[1:5])
  expect_equivalent(c(-0.0999179077423753, -0.101594935319188, 
    -0.0985572218431492, -0.103275329248674, -0.103028083292436), fast_smoother(approx_bssm)[1:5])
})

test_that("results for poisson GLM are equal to glm function",{
  d <- data.frame(treatment = gl(3,3), outcome = gl(3,1,9), counts = c(18,17,15,20,10,20,25,13,12))
  glm_poisson <- glm(counts ~ outcome + treatment, data = d, family = poisson())
  xreg <- model.matrix(~ outcome + treatment, data = d)
  expect_error(model_poisson1 <- ssm_ung(d$counts, Z = t(xreg), T = diag(5), R = diag(0, 5), 
    P1 = diag(1e7, 5), distribution = 'poisson', state_names = colnames(xreg)), NA)
  expect_error(sm <- smoother(model_poisson1), NA)
  expect_equal(sm$alphahat[1,], coef(glm_poisson))
  expect_equal(sm$V[,,1], vcov(glm_poisson))
  
  xreg <- model.matrix(~ outcome + treatment, data = d)[, -1]
  expect_error(model_poisson2 <- bsm_ng(d$counts, sd_level = 0, xreg = xreg, P1=matrix(1e7),
    beta = normal(coef(glm_poisson)[-1], 0, 10), distribution = 'poisson'), NA)
  expect_equivalent(smoother(model_poisson2)$alphahat[1,], coef(glm_poisson)[1])
  
  model_poisson3 <- model_poisson1
  model_poisson3$P1[] <- 0
  model_poisson3$P1[1] <- 1e7
  model_poisson3$a1[2:5] <- coef(glm_poisson)[-1]
  
  model_poisson4 <- ssm_ung(d$counts, Z = 1, T = 1, R = 0, 
    D = t(model_poisson2$xreg %*% model_poisson2$beta),
    P1 = 1e7, distribution = 'poisson')
  
  expect_equivalent(gaussian_approx(model_poisson1)$y, 
    gaussian_approx(model_poisson2)$y)
  
  expect_equivalent(gaussian_approx(model_poisson1)$y, 
    gaussian_approx(model_poisson3)$y)
  
  expect_equivalent(gaussian_approx(model_poisson1)$y, 
    gaussian_approx(model_poisson4)$y)
  
  expect_equivalent(gaussian_approx(model_poisson1)$H, 
    gaussian_approx(model_poisson2)$H)
  
  expect_equivalent(gaussian_approx(model_poisson1)$H, 
    gaussian_approx(model_poisson3)$H)
  
  expect_equivalent(gaussian_approx(model_poisson1)$H, 
    gaussian_approx(model_poisson4)$H)
  
})

test_that("results for binomial GLM are equal to glm function",{
  
  ldose <- rep(0:5, 2)
  numdead <- c(1, 4, 9, 13, 18, 20, 0, 2, 6, 10, 12, 16)
  sex <- factor(rep(c("M", "F"), c(6, 6)))
  SF <- cbind(numdead, numalive = 20-numdead)
  glm_binomial <- glm(SF ~ sex * ldose, family = binomial)
  xreg <- model.matrix(~  sex * ldose)
  expect_error(model_binomial <- ssm_ung(numdead, Z = t(xreg), 
    T = diag(4), R = diag(0, 4), P1 = diag(1e5, 4), 
    distribution = 'binomial', u = 20, state_names = colnames(xreg)), NA)
  expect_error(sm <- smoother(model_binomial), NA)
  # non-exact diffuse initialization is numerically difficult...
  expect_equal(sm$alphahat[1, ], coef(glm_binomial), tolerance = 1e-5)
  expect_equal(sm$V[, , 1], vcov(glm_binomial), tolerance = 1e-4)
  
})


test_that("state estimates for negative binomial GLM are equal to glm function",{
  library(MASS)
  set.seed(123)
  offs <- quine$Days + sample(10:20, size = nrow(quine), replace = TRUE)
  glm_nb <- glm.nb(Days ~ 1 + offset(log(offs)), data = quine)
  expect_error(model_nb <- bsm_ng(quine$Days, u = offs, sd_level = 0,
    P1 = matrix(1e7), phi = glm_nb$theta,
    distribution = 'negative binomial'), NA)
  
  approx_model <- gaussian_approx(model_nb, conv_tol = 1e-12)
  expect_error(sm <- smoother(approx_model), NA)
  expect_equivalent(sm$alphahat[1], unname(coef(glm_nb)[1]))
})

test_that("multivariate iid model gives same results as two univariate models", {
  set.seed(1)
  y <- matrix(rbinom(20, size = 10, prob = plogis(rnorm(20, sd = 0.5))), 10, 2)
  expect_error(model <- ssm_mng(y, Z = diag(2), phi = 2, 
    T = diag(2), R = array(diag(0.5, 2), c(2, 2, 1)), 
    a1 = matrix(0, 2, 1), P1 = diag(2), distribution = "negative binomial", 
    init_theta = c(0,0)), NA)
  expect_error(model1 <- ssm_ung(y[,1], Z = 1, phi = 2, 
    T = 1, R = 0.5, P1 = 1, distribution = "negative binomial", 
    init_theta = 0), NA)
  expect_error(model2 <- ssm_ung(y[,2], Z = 1, phi = 2, 
    T = 1, R = 0.5, P1 = 1, distribution = "negative binomial", 
    init_theta = 0), NA)
  expect_equivalent(gaussian_approx(model)$y, 
    cbind(gaussian_approx(model1)$y, gaussian_approx(model2)$y))
})
