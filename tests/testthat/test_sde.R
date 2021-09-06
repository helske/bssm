context("Test SDE")
test_that("MCMC for SDE works", {
  skip_on_cran()
  code <- ' 
    #include <RcppArmadillo.h>
    // [[Rcpp::depends(RcppArmadillo)]]
    // [[Rcpp::interfaces(r, cpp)]]
    
    // Drift function
    // [[Rcpp::export]]
    double drift(const double x, const arma::vec& theta) {
      return theta(0) * x;
    }
    // diffusion function
    // [[Rcpp::export]]
    double diffusion(const double x, const arma::vec& theta) {
      return std::max(0.0, theta(1) * x);
    }
    // Derivative of the diffusion function
    // [[Rcpp::export]]
    double ddiffusion(const double x, const arma::vec& theta) {
      return theta(1) * (x > 0.0);
    }
    
    // log-density of the prior
    // [[Rcpp::export]]
    double log_prior_pdf(const arma::vec& theta) {
      
      double log_pdf = 0.0;
      if(theta(0) <= -0.05 || theta(1) <= 0.0 || theta(2) <= 0.0) {
        log_pdf = -std::numeric_limits<double>::infinity();
      }
      else {
        log_pdf = R::dnorm(theta(0), 0, 0.05, 1) +
          R::dnorm(theta(1), 0, 0.5, 1) +
          R::dnorm(theta(2), 1, 0.1, 1);
      }
      return log_pdf;
    }
    
    // log-density of observations
    // given vector of sampled states alpha
    // [[Rcpp::export]]
    arma::vec log_obs_density(const double y, 
      const arma::vec& alpha, const arma::vec& theta) {
      
      arma::vec log_pdf(alpha.n_elem);
      for (unsigned int i = 0; i < alpha.n_elem; i++) {
        log_pdf(i) = R::dnorm(y, std::log(alpha(i)), theta(2), 1);
      }
      return log_pdf;
    }
    
    // [[Rcpp::export]]
    Rcpp::List create_xptrs() {
      // typedef for a pointer of drift/volatility function
      typedef double (*fnPtr)(const double x, const arma::vec& theta);
      // typedef for log_prior_pdf
      typedef double (*prior_fnPtr)(const arma::vec& theta);
      // typedef for log_obs_density
      typedef arma::vec (*obs_fnPtr)(const double y, 
        const arma::vec& alpha, const arma::vec& theta);
      
      return Rcpp::List::create(
        Rcpp::Named("drift") = Rcpp::XPtr<fnPtr>(new fnPtr(&drift)),
        Rcpp::Named("diffusion") = Rcpp::XPtr<fnPtr>(new fnPtr(&diffusion)),
        Rcpp::Named("ddiffusion") = Rcpp::XPtr<fnPtr>(new fnPtr(&ddiffusion)),
        Rcpp::Named("prior") = 
          Rcpp::XPtr<prior_fnPtr>(new prior_fnPtr(&log_prior_pdf)),
        Rcpp::Named("obs_density") = 
          Rcpp::XPtr<obs_fnPtr>(new obs_fnPtr(&log_obs_density)));
    }
  '
  Rcpp::sourceCpp(code = code)
  pntrs <- create_xptrs()
  set.seed(1)
  x <- 1
  y <- rep(NA, 10)
  dt <- 1
  mu <- 0.05
  sigma_x <- 0.3
  sigma_y <- 1
  for (k in 1:10) {
    x <- x*exp((mu-0.5*sigma_x^2)*dt + sqrt(dt) * rnorm(1, sd=sigma_x))
    y[k] <- rnorm(1, mean=log(x), sd=sigma_y)
  }
  
  set.seed(123)
  model <- ssm_sde(y, pntrs$drift, pntrs$diffusion, 
    pntrs$ddiffusion, pntrs$obs_density,
    pntrs$prior, c(0.05, 0.3, 1), x0 = 1, positive = TRUE)
  
  expect_error(ll <- logLik(model, 10000, L = 3), NA)
  expect_equal(ll, -17, tol = 1)
  expect_error(out_bsf <- bootstrap_filter(model, 1000, L = 3), NA)
  expect_equal(ll, out_bsf$logLik, tol = 1)
  
  expect_error(out <- run_mcmc(model, iter = 500, 
    particles = 10, mcmc_type = "pm", L_f = 2), NA)
  expect_gt(out$acceptance_rate, 0)
  
  expect_error(out <- run_mcmc(model, iter = 500, 
    particles = 10, mcmc_type = "da", L_c = 2, L_f = 3), NA)
  expect_gt(out$acceptance_rate, 0)
  
  expect_error(out2 <- run_mcmc(model, iter = 500, 
    particles = 10, mcmc_type = "is2", L_c = 1, L_f = 2), NA)
  
  expect_gt(out2$acceptance_rate, 0)
  expect_equal(mean(colMeans(out$theta)-colMeans(out2$theta)), 0, tol = 1)
  
  expect_error(out2 <- run_mcmc(model, iter = 500, 
    particles = 10, mcmc_type = "is1", L_c = 1, L_f = 2, threads = 2), NA)
  
  expect_gt(out2$acceptance_rate, 0)
  expect_equal(mean(colMeans(out$theta)-colMeans(out2$theta)), 0, tol = 1)
})
