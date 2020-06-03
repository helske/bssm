// A template for building a univariate discretely observed diffusion model
// Here we define a latent Ornstein–Uhlenbeck process with Poisson observations
// d\alpha_t = \rho (\nu - \alpha_t) dt + \sigma dB_t, t>=0
// y_k ~ Poisson(exp(\alpha_k)), k = 1,...,n

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(BH)]]
// [[Rcpp::interfaces(r, cpp)]]

// x: state
// theta: vector of parameters

// theta(0) = rho
// theta(1) = nu
// theta(2) = sigma

// Drift function
// [[Rcpp::export]]
double drift(const double x, const arma::vec& theta) {
  return theta(0) * (theta(1) - x);
}
// diffusion function
// [[Rcpp::export]]
double diffusion(const double x, const arma::vec& theta) {
  return theta(2);
}
// Derivative of the diffusion function
// [[Rcpp::export]]
double ddiffusion(const double x, const arma::vec& theta) {
  return 0.0;
}

// log-density of the prior
// [[Rcpp::export]]
double log_prior_pdf(const arma::vec& theta) {
  
  double log_pdf;
  if(theta(0) <= 0.0 || theta(2) <= 0.0) {
    log_pdf = -std::numeric_limits<double>::infinity();
  } else {
    // weakly informative priors. 
    // Note that negative values are handled above
    log_pdf = R::dnorm(theta(0), 0, 10, 1) + R::dnorm(theta(1), 0, 10, 1) + 
      R::dnorm(theta(2), 0, 10, 1);
  }
  return log_pdf;
}

// log-density of observations
// [[Rcpp::export]]
arma::vec log_obs_density(const double y, 
  const arma::vec& alpha, const arma::vec& theta) {
  
  arma::vec log_pdf(alpha.n_elem);
  for (unsigned int i = 0; i < alpha.n_elem; i++) {
    log_pdf(i) = R::dpois(y, exp(alpha(i)), 1);
  }
  return log_pdf;
}


// Function which returns the pointers to above functions (no need to modify)

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
    Rcpp::Named("prior") = Rcpp::XPtr<prior_fnPtr>(new prior_fnPtr(&log_prior_pdf)),
    Rcpp::Named("obs_density") = Rcpp::XPtr<obs_fnPtr>(new obs_fnPtr(&log_obs_density)));
}
