#include "ung_ar1.h"

// from Rcpp::List
ung_ar1::ung_ar1(const Rcpp::List& model, const unsigned int seed) :
  ung_ssm(model, seed), mu_est(Rcpp::as<bool>(model["mu_est"])) {
}

void ung_ar1::update_model(const arma::vec& theta) {
  
  
  T(0, 0, 0) = theta(0);
  R(0, 0, 0) = theta(1);
  if (mu_est) {
    a1(0) = theta(2);
    C.fill(theta(2) * (1.0 - theta(0)));
  }
  P1(0, 0) = std::pow(theta(1), 2) / (1.0 - std::pow(theta(0), 2));
  
  compute_RR();
  
  if(phi_est) {
    phi = theta(2 + mu_est);
  }
  
  if(xreg.n_cols > 0) {
    beta = theta.subvec(theta.n_elem - xreg.n_cols, theta.n_elem - 1);
    compute_xbeta();
  }
}

arma::vec ung_ar1::get_theta(void) const {
  
  unsigned int npar = 2 + mu_est + xreg.n_cols + phi_est;
  
  arma::vec theta(npar);
  theta(0) = T(0, 0, 0);
  theta(1) = R(0, 0, 0);
  if (mu_est) {
    theta(2) = a1(0);
  }
  
  if (phi_est) {
    theta(2 + mu_est) = phi;
  }
  
  if (xreg.n_cols > 0) {
    theta.subvec(theta.n_elem - xreg.n_cols, theta.n_elem - 1) = beta;
  }
  return theta;
}
