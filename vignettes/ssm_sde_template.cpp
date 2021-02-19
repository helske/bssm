// A template for building a univariate discretely observed diffusion model
// Here we define a latent Ornstein–Uhlenbeck process with Poisson observations
// d\alpha_t = \rho (\nu - \alpha_t) dt + \sigma dB_t, t>=0
// y_k ~ Poisson(exp(\alpha_k)), k = 1,...,n

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::interfaces(r, cpp)]]

// x: state
// theta: vector of parameters

// theta(0) = log_rho
// theta(1) = nu
// theta(2) = log_sigma

// Drift function
// [[Rcpp::export]]
double drift(const double x, const arma::vec& theta) {
  return exp(theta(0)) * (theta(1) - x);
}
// diffusion function
// [[Rcpp::export]]
double diffusion(const double x, const arma::vec& theta) {
  return exp(theta(2));
}
// Derivative of the diffusion function
// [[Rcpp::export]]
double ddiffusion(const double x, const arma::vec& theta) {
  return 0.0;
}

// log-density of the prior
// [[Rcpp::export]]
double log_prior_pdf(const arma::vec& theta) {
  
  // rho ~ gamma(2, 0.5) // shape-scale parameterization
  // nu ~ N(0, 4)
  // sigma ~ half-N(0,1) (theta(2) is log(sigma))
  double log_pdf = 
    R::dgamma(exp(theta(0)), 2, 0.5, 1) + 
    R::dnorm(theta(1), 0, 4, 1) + 
    R::dnorm(exp(theta(2)), 0, 1, 1) + 
    theta(0) + theta(2); // jacobians of transformations
  return log_pdf;
}

// log-density of observations
// given vector of sampled states alpha
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
