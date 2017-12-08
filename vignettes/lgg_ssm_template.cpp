
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::interfaces(r, cpp)]]

// theta:
// theta(0) = standard deviation sigma_y
// theta(1) = standard deviation sigma_level
// theta(2) = standard deviation sigma_slope
//
// known_params contains the prior for the initial state
// known_tv_params is not used in this model
// known_params(0): prior mean a_1
// known_params(1): prior _variance_ P_1

// Function for the prior mean of alpha_1
// [[Rcpp::export]]
arma::vec a1_fn(const arma::vec& theta, const arma::vec& known_params) {
  
  arma::vec a1(1);
  a1(0) = known_params(0);
  
  return a1;
}
// Function for the prior variance of alpha_1
// [[Rcpp::export]]
arma::mat P1_fn(const arma::vec& theta, const arma::vec& known_params) {
  
  arma::mat P1(1, 1);
  P1(0,0) = known_params(1);
  
  return P1;
}

// Function for the observational level standard deviation
// [[Rcpp::export]]
arma::mat H_fn(const unsigned int t, const arma::vec& theta, 
  const arma::vec& known_params, const arma::mat& known_tv_params) {
  
  arma::mat H(1,1);
  H(0, 0) = exp(theta(0)); //force standard deviation to positive via transformation
  return H;
}

// Function for the Cholesky of state level covariance
// [[Rcpp::export]]
arma::mat R_fn(const unsigned int t, const arma::vec& theta, 
  const arma::vec& known_params, const arma::mat& known_tv_params) {
  arma::mat R(1, 1);
  R(0, 0) = exp(theta(1));
  return R;
}


// Z function
// [[Rcpp::export]]
arma::mat Z_fn(const unsigned int t, const arma::vec& theta, 
  const arma::vec& known_params, const arma::mat& known_tv_params) {
  arma::mat Z(1, 2, arma::fill::zeros);
  Z(0, 0) = 1.0;
  return Z;
}

// T function
// [[Rcpp::export]]
arma::mat T_fn(const unsigned int t, const arma::vec& theta, 
  const arma::vec& known_params, const arma::mat& known_tv_params) {
  arma::mat T(2, 2, arma::fill::ones);
  T(1, 0) = 0.0;
  return T;
}

// input to state equation
// [[Rcpp::export]]
arma::vec C_fn(const unsigned int t, const arma::vec& theta, 
  const arma::vec& known_params, const arma::mat& known_tv_params) {
  return arma::vec(1, arma::fill::zeros);
}
// input to observation equation
// [[Rcpp::export]]
arma::vec D_fn(const unsigned int t, const arma::vec& theta, 
  const arma::vec& known_params, const arma::mat& known_tv_params) {
  return arma::vec(1, arma::fill::zeros);
}

// # log-prior pdf for theta
// [[Rcpp::export]]
double log_prior_pdf(const arma::vec& theta) {
  
  double log_pdf = R::dnorm(theta(0), 0, 10, 1) + 
    R::dnorm(theta(1), 0, 10, 1) + 
    R::dnorm(theta(2), 0, 10, 1);
  
  return log_pdf;
}


// Create pointers, no need to touch this if
// you don't alter the function names above
// [[Rcpp::export]]
Rcpp::List create_xptrs() {
  
  // typedef for a pointer of linear function of lgg-model equation returning vec (T, Z)
  typedef arma::mat (*mat_fnPtr2)(const unsigned int t, const arma::vec& theta, 
    const arma::vec& known_params, const arma::mat& known_tv_params);
  // typedef for intercept terms (C, D)
  typedef arma::vec (*vec_fnPtr2)(const unsigned int t, const arma::vec& theta, 
    const arma::vec& known_params, const arma::mat& known_tv_params);
  
  // typedef for a pointer of function of model equation returning mat (R, H)
  typedef arma::mat (*mat_varfnPtr)(const unsigned int t, const arma::vec& theta, 
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
    Rcpp::Named("Z_fn") = Rcpp::XPtr<mat_fnPtr2>(new mat_fnPtr2(&Z_fn)),
    Rcpp::Named("H_fn") = Rcpp::XPtr<mat_varfnPtr>(new mat_varfnPtr(&H_fn)),
    Rcpp::Named("T_fn") = Rcpp::XPtr<mat_fnPtr2>(new mat_fnPtr2(&T_fn)),
    Rcpp::Named("R_fn") = Rcpp::XPtr<mat_varfnPtr>(new mat_varfnPtr(&R_fn)),
    Rcpp::Named("D_fn") = Rcpp::XPtr<vec_fnPtr2>(new vec_fnPtr2(&D_fn)),
    Rcpp::Named("C_fn") = Rcpp::XPtr<vec_fnPtr2>(new vec_fnPtr2(&C_fn)),
    Rcpp::Named("log_prior_pdf") = 
      Rcpp::XPtr<double_fnPtr>(new double_fnPtr(&log_prior_pdf)));
}
