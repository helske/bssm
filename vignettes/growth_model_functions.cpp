#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::interfaces(r, cpp)]]

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
  H(0, 0) = theta(0);
  return H;
}

// Function for the Cholesky of state level covariance
// [[Rcpp::export]]
arma::mat R_fn(const unsigned int t, const arma::vec& alpha, const arma::vec& theta, 
  const arma::vec& known_params, const arma::mat& known_tv_params) {
  arma::mat R(2, 2, arma::fill::zeros);
  R(0, 0) = theta(1);
  R(1, 1) = theta(2);
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
  double k = known_params(1);

  arma::vec alpha_new(2);
  alpha_new(0) = alpha(0);
  alpha_new(1) = k * alpha(1) * exp(alpha(0) * dT) / 
    (k + alpha(1) * (exp(alpha(0) * dT) -1));
  
  return alpha_new;
}

// Jacobian of T function
// [[Rcpp::export]]
arma::mat T_gn(const unsigned int t, const arma::vec& alpha, const arma::vec& theta, 
  const arma::vec& known_params, const arma::mat& known_tv_params) {
  
  double dT = known_params(0);
  double k = known_params(1);
  
  double tmp = exp(alpha(0) * dT) / 
    std::pow(k + alpha(1) * (exp(alpha(0) * dT) - 1), 2);
  
  arma::mat Tg(2, 2);
  Tg(0, 0) = 1.0;
  Tg(0, 1) = 0;
  Tg(1, 0) = k * alpha(1) * dT * (k - alpha(1)) * tmp;
  Tg(1, 1) = k * k * tmp;
  
  return Tg;
}



// # log-prior pdf for theta
// [[Rcpp::export]]
double log_prior_pdf(const arma::vec& theta) {
  
  double log_pdf;
  if(arma::any(theta < 0)) {
     log_pdf = -std::numeric_limits<double>::infinity();
   } else {
    // weakly informative priors. 
    // Note that negative values are handled above
    log_pdf = R::dnorm(theta(0), 0, 10, 1) + R::dnorm(theta(1), 0, 10, 1) + 
      R::dnorm(theta(2), 0, 10, 1);
  }
  return log_pdf;
}

// Create pointers, no need to touch this if
// you don't alter the function names above
// [[Rcpp::export]]
Rcpp::List create_xptrs() {
  // typedef for a pointer of nonlinear function of model equation returning vec (T, Z)
  typedef arma::vec (*vec_fnPtr)(const unsigned int t, const arma::vec& alpha, const arma::vec& theta, 
    const arma::vec& known_params, const arma::mat& known_tv_params);
  // typedef for a pointer of nonlinear function of model equation returning mat (Tg, Zg, H, R)
  typedef arma::mat (*mat_fnPtr)(const unsigned int t, const arma::vec& alpha, const arma::vec& theta, 
    const arma::vec& known_params, const arma::mat& known_tv_params);
  // typedef for a pointer of nonlinear function of model equation returning vec (a1)
  typedef arma::vec (*vec_initfnPtr)(const arma::vec& theta, const arma::vec& known_params);
  // typedef for a pointer of nonlinear function of model equation returning mat (P1)
  typedef arma::mat (*mat_initfnPtr)(const arma::vec& theta, const arma::vec& known_params);
  // typedef for a pointer of log-prior function
  typedef double (*double_fnPtr)(const arma::vec&);
  
  return Rcpp::List::create(
    Rcpp::Named("a1_fn") = Rcpp::XPtr<vec_initfnPtr>(new vec_initfnPtr(&a1_fn)),
    Rcpp::Named("P1_fn") = Rcpp::XPtr<mat_initfnPtr>(new mat_initfnPtr(&P1_fn)),
    Rcpp::Named("Z_fn") = Rcpp::XPtr<vec_fnPtr>(new vec_fnPtr(&Z_fn)),
    Rcpp::Named("H_fn") = Rcpp::XPtr<mat_fnPtr>(new mat_fnPtr(&H_fn)),
    Rcpp::Named("T_fn") = Rcpp::XPtr<vec_fnPtr>(new vec_fnPtr(&T_fn)),
    Rcpp::Named("R_fn") = Rcpp::XPtr<mat_fnPtr>(new mat_fnPtr(&R_fn)),
    Rcpp::Named("Z_gn") = Rcpp::XPtr<mat_fnPtr>(new mat_fnPtr(&Z_gn)),
    Rcpp::Named("T_gn") = Rcpp::XPtr<mat_fnPtr>(new mat_fnPtr(&T_gn)),
    Rcpp::Named("log_prior_pdf") = 
      Rcpp::XPtr<double_fnPtr>(new double_fnPtr(&log_prior_pdf)));
}
