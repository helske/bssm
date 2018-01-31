// A template for building a general linear-Gaussian state space model
// Here we define an univariate local linear trend model which could be 
// constructed also with bsm function.

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::interfaces(r, cpp)]]

// theta:
// theta(0) = standard deviation sigma_y
// theta(1) = standard deviation sigma_level
// theta(2) = standard deviation sigma_slope
//
// Function for the prior mean of alpha_1
// [[Rcpp::export]]
arma::vec a1_fn(const arma::vec& theta, const arma::vec& known_params) {
  return arma::vec(2, arma::fill::zeros);
}
// Function for the prior variance of alpha_1
// [[Rcpp::export]]
arma::mat P1_fn(const arma::vec& theta, const arma::vec& known_params) {
  
  arma::mat P1(2, 2, arma::fill::zeros);
  P1(0, 0) = 1000;
  P1(1, 1) = 1000;
  return P1;
}

// Function for the Cholesky of the observational level covariance matrix
// [[Rcpp::export]]
arma::mat H_fn(const unsigned int t, const arma::vec& theta, 
  const arma::vec& known_params, const arma::mat& known_tv_params) {
  // note no transformations, needs to check for positivity in prior
  // we could also use exp(theta) here and work with the corresponding prior
  arma::mat H(1,1);
  H(0, 0) = theta(0);
  return H;
}

// Function for the Cholesky of state level covariance matrix
// [[Rcpp::export]]
arma::mat R_fn(const unsigned int t, const arma::vec& theta, 
  const arma::vec& known_params, const arma::mat& known_tv_params) {
  arma::mat R(2, 2, arma::fill::zeros);
  R(0, 0) = theta(1);
  R(1, 1) = theta(2);
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
  return arma::vec(2, arma::fill::zeros);
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
  
  double log_pdf = -std::numeric_limits<double>::infinity();
  if (arma::all(theta >= 0)) {
   log_pdf = R::dnorm(theta(0), 0, 10, 1) + 
     R::dnorm(theta(1), 0, 10, 1) + 
     R::dnorm(theta(2), 0, 10, 1); 
  }
  
  return log_pdf;
}


// Create pointers, no need to touch this if
// you don't alter the function names above
// [[Rcpp::export]]
Rcpp::List create_xptrs() {
  
  // typedef for a pointer returning matrices Z, H, T, and R
  typedef arma::mat (*lmat_fnPtr)(const unsigned int t, const arma::vec& theta, 
    const arma::vec& known_params, const arma::mat& known_tv_params);
  // typedef for a pointer of linear function of lgg-model equation returning vectors D and C
  typedef arma::vec (*lvec_fnPtr)(const unsigned int t, const arma::vec& theta, 
    const arma::vec& known_params, const arma::mat& known_tv_params);
  
  // typedef for a pointer returning vector a1
  typedef arma::vec (*a1_fnPtr)(const arma::vec& theta, const arma::vec& known_params);
  // typedef for a pointer returning matrix P1
  typedef arma::mat (*P1_fnPtr)(const arma::vec& theta, const arma::vec& known_params);
  // typedef for a pointer of log-prior function
  typedef double (*prior_fnPtr)(const arma::vec&);
  
  return Rcpp::List::create(
    Rcpp::Named("a1_fn") = Rcpp::XPtr<a1_fnPtr>(new a1_fnPtr(&a1_fn)),
    Rcpp::Named("P1_fn") = Rcpp::XPtr<P1_fnPtr>(new P1_fnPtr(&P1_fn)),
    Rcpp::Named("Z_fn") = Rcpp::XPtr<lmat_fnPtr>(new lmat_fnPtr(&Z_fn)),
    Rcpp::Named("H_fn") = Rcpp::XPtr<lmat_fnPtr>(new lmat_fnPtr(&H_fn)),
    Rcpp::Named("T_fn") = Rcpp::XPtr<lmat_fnPtr>(new lmat_fnPtr(&T_fn)),
    Rcpp::Named("R_fn") = Rcpp::XPtr<lmat_fnPtr>(new lmat_fnPtr(&R_fn)),
    Rcpp::Named("D_fn") = Rcpp::XPtr<lvec_fnPtr>(new lvec_fnPtr(&D_fn)),
    Rcpp::Named("C_fn") = Rcpp::XPtr<lvec_fnPtr>(new lvec_fnPtr(&C_fn)),
    Rcpp::Named("log_prior_pdf") = 
      Rcpp::XPtr<prior_fnPtr>(new prior_fnPtr(&log_prior_pdf)));
}
