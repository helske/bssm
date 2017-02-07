#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::interfaces(r, cpp)]]

arma::vec a1_fn(const arma::vec& theta, const arma::vec& psi) {
  arma::vec a1(1);
  a1(0) = theta(0);
  return a1;
}
arma::mat P1_fn(const arma::vec& theta, const arma::vec& psi) {
  arma::mat P1(1, 1);
  P1(0, 0) = theta(2) * theta(2) / (1.0 - theta(1) * theta(1));
  return P1;
}

arma::vec Z_fn(const unsigned int t, const arma::vec& alpha, const arma::vec& theta, 
  const arma::vec& psi) {
  return arma::vec(1, arma::fill::zeros);
}
arma::mat H_fn(const unsigned int t, const arma::vec& alpha, const arma::vec& theta, 
  const arma::vec& psi) {
  return exp(alpha / 2.0);
}
arma::vec T_fn(const unsigned int t, const arma::vec& alpha, const arma::vec& theta, 
  const arma::vec& psi) {
  return theta(0) + theta(1) * (alpha - theta(0));
}
arma::mat R_fn(const unsigned int t, const arma::vec& alpha, const arma::vec& theta, 
  const arma::vec& psi) {
  arma::mat chol_cov(1, 1);
  chol_cov(0, 0) = theta(2);
  return chol_cov;
}
arma::mat Z_gn(const unsigned int t, const arma::vec& alpha, const arma::vec& theta, 
  const arma::vec& psi) {
  return arma::mat(1, 1, arma::fill::zeros);
}
arma::mat T_gn(const unsigned int t, const arma::vec& alpha, const arma::vec& theta, 
  const arma::vec& psi) {
  arma::mat phi(1,1);
  phi(0, 0) = theta(1);
  return phi;
}

double log_prior_pdf(const arma::vec& theta) {
  
  double log_pdf;
  if(theta(2) < 0 || theta(1) <= -1.0 || theta(1) >= 1.0) {
    log_pdf = -arma::datum::inf;
  } else {
    // prior for \mu: N(0, 5^2)
    log_pdf = R::dnorm(theta(0), 0, 5, 1);
    // prior for \phi: U(-0.9999, 0.9999)
    log_pdf += R::dunif(theta(1), -1, 1, 1);
    // prior for \sigma: Half-Normal with sd 5
    log_pdf += log(2.0) + R::dnorm(theta(2), 0, 5, 1);
  }
  return log_pdf;
}

// [[Rcpp::export]]
Rcpp::List create_xptrs() {
  // typedef for a pointer of nonlinear function of model equation returning vec
  typedef arma::vec (*vec_fnPtr)(unsigned int t, const arma::vec&, const arma::vec&, const arma::vec&);
  // typedef for a pointer of nonlinear function of model equation returning mat
  typedef arma::mat (*mat_fnPtr)(unsigned int t, const arma::vec&, const arma::vec&, const arma::vec&);
  // typedef for a pointer of nonlinear function of model equation returning vec
  typedef arma::vec (*vec_initfnPtr)(const arma::vec&, const arma::vec&);
  // typedef for a pointer of nonlinear function of model equation returning mat
  typedef arma::mat (*mat_initfnPtr)(const arma::vec&, const arma::vec&);
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