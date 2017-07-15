// Functions for univariate SDE models

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::interfaces(r, cpp)]]

// x: state
// theta: vector of parameters

// Geometric Brownian motion case:
// theta(0) = mu
// theta(1) = sigma

// Drift function
// [[Rcpp::export]]
double drift(const double x, const arma::vec& theta) {
  return theta(0) * x;
}
// Volatility function
// [[Rcpp::export]]
double diffusion(const double x, const arma::vec& theta) {
  return std::max(0.0, theta(1) * x);
}
// Derivative of volatility
// [[Rcpp::export]]
double ddiffusion(const double x, const arma::vec& theta) {
  return theta(1) * (x > 0.0);
}

// log-density of the prior
// [[Rcpp::export]]
double log_prior_pdf(const arma::vec& theta) {
  
  double log_pdf;
  if(theta(1) <= 0.0 || theta(2) <= 0.0) {
    log_pdf = -arma::datum::inf;
  } else {
    // weakly informative priors. 
    // Note that negative values are handled above
    log_pdf = R::dnorm(theta(0), 0, 1, 1) + R::dnorm(theta(1), 0, 1, 1) + 
      R::dnorm(theta(2), 0, 1, 1);
  }
  return log_pdf;
}

// log-density of observations
// [[Rcpp::export]]
arma::vec log_obs_density(const double y, 
  const arma::vec& alpha, const arma::vec& theta) {
  
  arma::vec log_pdf(alpha.n_elem);
  for (unsigned int i = 0; i < alpha.n_elem; i++) {
    log_pdf(i) = R::dnorm(y, std::log(alpha(i)), theta(2), 1);
  }
  return log_pdf;
}


// Function which returns the pointers to above functions (no need to modify)

// [[Rcpp::export]]
Rcpp::List create_pntrs() {
  // typedef for a pointer of drift/volatility function
  typedef double (*funcPtr)(const double x, const arma::vec& theta);
  typedef double (*prior_funcPtr)(const arma::vec& theta);
  typedef arma::vec (*obs_funcPtr)(const double y, 
    const arma::vec& alpha, const arma::vec& theta);
  
  return Rcpp::List::create(
    Rcpp::Named("drift") = Rcpp::XPtr<funcPtr>(new funcPtr(&drift)),
    Rcpp::Named("diffusion") = Rcpp::XPtr<funcPtr>(new funcPtr(&diffusion)),
    Rcpp::Named("ddiffusion") = Rcpp::XPtr<funcPtr>(new funcPtr(&ddiffusion)),
    Rcpp::Named("prior") = Rcpp::XPtr<prior_funcPtr>(new prior_funcPtr(&log_prior_pdf)),
    Rcpp::Named("obs_density") = Rcpp::XPtr<obs_funcPtr>(new obs_funcPtr(&log_obs_density)));
}
