// A template for building a general non-linear Gaussian state space model
// Here we define an univariate growth model (see vignette growth_model)

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::interfaces(r, cpp)]]

// Unknown parameters theta:
// theta(0) = log(H)
// theta(1) = log(R_1)
// theta(2) = log(R_2)

// Function for the prior mean of alpha_1
// [[Rcpp::export]]
arma::vec a1_fn(const arma::vec& theta, const arma::vec& known_params) {
  
  arma::vec a1(2);
  a1(0) = known_params(2);
  a1(1) = known_params(3);
  return a1;
}
// Function for the prior covariance matrix of alpha_1
// [[Rcpp::export]]
arma::mat P1_fn(const arma::vec& theta, const arma::vec& known_params) {
  
  arma::mat P1(2, 2, arma::fill::zeros);
  P1(0,0) = known_params(4);
  P1(1,1) = known_params(5);
  return P1;
}

// Function for the observational level standard deviation
// [[Rcpp::export]]
arma::mat H_fn(const unsigned int t, const arma::vec& alpha, const arma::vec& theta, 
  const arma::vec& known_params, const arma::mat& known_tv_params) {
  arma::mat H(1,1);
  H(0, 0) = exp(theta(0));
  return H;
}

// Function for the Cholesky of state level covariance
// [[Rcpp::export]]
arma::mat R_fn(const unsigned int t, const arma::vec& alpha, const arma::vec& theta, 
  const arma::vec& known_params, const arma::mat& known_tv_params) {
  arma::mat R(2, 2, arma::fill::zeros);
  R(0, 0) = exp(theta(1));
  R(1, 1) = exp(theta(2));
  return R;
}


// Z function
// [[Rcpp::export]]
arma::vec Z_fn(const unsigned int t, const arma::vec& alpha, const arma::vec& theta, 
  const arma::vec& known_params, const arma::mat& known_tv_params) {
  arma::vec tmp(1);
  tmp(0) = alpha(1);
  return tmp;
}
// Jacobian of Z function
// [[Rcpp::export]]
arma::mat Z_gn(const unsigned int t, const arma::vec& alpha, const arma::vec& theta, 
  const arma::vec& known_params, const arma::mat& known_tv_params) {
  arma::mat Z_gn(1, 2);
  Z_gn(0, 0) = 0.0;
  Z_gn(0, 1) = 1.0;
  return Z_gn;
}

// T function
// [[Rcpp::export]]
arma::vec T_fn(const unsigned int t, const arma::vec& alpha, const arma::vec& theta, 
  const arma::vec& known_params, const arma::mat& known_tv_params) {
  
  double dT = known_params(0);
  double K = known_params(1);
  
  arma::vec alpha_new(2);
  alpha_new(0) = alpha(0);
  double r = exp(alpha(0)) / (1.0 + exp(alpha(0)));
  alpha_new(1) = K * alpha(1) * exp(r * dT) / 
    (K + alpha(1) * (exp(r * dT) - 1));
  return alpha_new;
}

// Jacobian of T function
// [[Rcpp::export]]
arma::mat T_gn(const unsigned int t, const arma::vec& alpha, const arma::vec& theta, 
  const arma::vec& known_params, const arma::mat& known_tv_params) {
  
  double dT = known_params(0);
  double K = known_params(1);
  
  double r = exp(alpha(0)) / (1 + exp(alpha(0)));
  
  double tmp = exp(r * dT) / std::pow(K + alpha(1) * (exp(r * dT) - 1), 2);
  
  arma::mat Tg(2, 2);
  Tg(0, 0) = 1.0;
  Tg(0, 1) = 0;
  Tg(1, 0) = dT * K * alpha(1) * (K - alpha(1)) * tmp * r / (1 + exp(alpha(0)));
  Tg(1, 1) = K * K * tmp;
  
  return Tg;
}

// # log-prior pdf for theta
// [[Rcpp::export]]
double log_prior_pdf(const arma::vec& theta) {
  
  // weakly informative half-N(0, 4) priors. 
  
  // Note that the sampling is on log-scale, 
  // so we need to add jacobians of the corresponding transformations
  // we could also sample on natural scale with check such as
  // if(arma::any(theta < 0)) return -std::numeric_limits<double>::infinity();
  // but this would be less efficient.
  
  // You can use R::dnorm and similar functions, see, e.g.
  // https://teuder.github.io/rcpp4everyone_en/220_dpqr_functions.html
  double log_pdf =  
    R::dnorm(exp(theta(0)), 0, 2, 1) +
    R::dnorm(exp(theta(1)), 0, 2, 1) +
    R::dnorm(exp(theta(2)), 0, 2, 1) + 
    arma::accu(theta); //jacobian term
  
  return log_pdf;
}

// Create pointers, no need to touch this if
// you don't alter the function names above
// [[Rcpp::export]]
Rcpp::List create_xptrs() {
  
  // typedef for a pointer of nonlinear function of model equation returning vec (T, Z)
  typedef arma::vec (*nvec_fnPtr)(const unsigned int t, const arma::vec& alpha, 
    const arma::vec& theta, const arma::vec& known_params, const arma::mat& known_tv_params);
  // typedef for a pointer of nonlinear function returning mat (Tg, Zg, H, R)
  typedef arma::mat (*nmat_fnPtr)(const unsigned int t, const arma::vec& alpha, 
    const arma::vec& theta, const arma::vec& known_params, const arma::mat& known_tv_params);
  
  // typedef for a pointer returning a1
  typedef arma::vec (*a1_fnPtr)(const arma::vec& theta, const arma::vec& known_params);
  // typedef for a pointer returning P1
  typedef arma::mat (*P1_fnPtr)(const arma::vec& theta, const arma::vec& known_params);
  // typedef for a pointer of log-prior function
  typedef double (*prior_fnPtr)(const arma::vec& theta);
  
  return Rcpp::List::create(
    Rcpp::Named("a1_fn") = Rcpp::XPtr<a1_fnPtr>(new a1_fnPtr(&a1_fn)),
    Rcpp::Named("P1_fn") = Rcpp::XPtr<P1_fnPtr>(new P1_fnPtr(&P1_fn)),
    Rcpp::Named("Z_fn") = Rcpp::XPtr<nvec_fnPtr>(new nvec_fnPtr(&Z_fn)),
    Rcpp::Named("H_fn") = Rcpp::XPtr<nmat_fnPtr>(new nmat_fnPtr(&H_fn)),
    Rcpp::Named("T_fn") = Rcpp::XPtr<nvec_fnPtr>(new nvec_fnPtr(&T_fn)),
    Rcpp::Named("R_fn") = Rcpp::XPtr<nmat_fnPtr>(new nmat_fnPtr(&R_fn)),
    Rcpp::Named("Z_gn") = Rcpp::XPtr<nmat_fnPtr>(new nmat_fnPtr(&Z_gn)),
    Rcpp::Named("T_gn") = Rcpp::XPtr<nmat_fnPtr>(new nmat_fnPtr(&T_gn)),
    Rcpp::Named("log_prior_pdf") = 
      Rcpp::XPtr<prior_fnPtr>(new prior_fnPtr(&log_prior_pdf)));
  
}
