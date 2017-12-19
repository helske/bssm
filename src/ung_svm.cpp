#include "ung_svm.h"

// construct SV model from Rcpp::List
ung_svm::ung_svm(const Rcpp::List& model, const unsigned int seed) :
  ung_ssm(model, seed), svm_type(model["svm_type"]) {
}

// update model given the parameters theta
void ung_svm::update_model(const arma::vec& theta) {

  if(svm_type == 0) {
    phi = theta(2);
  } else {
    a1(0) = theta(2);
    C.fill(theta(2) * (1.0 - theta(0)));
  }

  T(0, 0, 0) = theta(0);
  R(0, 0, 0) = theta(1);
  compute_RR();
  P1(0, 0) = theta(1) * theta(1) / (1 - theta(0) * theta(0));

  if(xreg.n_cols > 0) {
    beta = theta.subvec(theta.n_elem - xreg.n_cols, theta.n_elem - 1);
    compute_xbeta();
  }
}

// extract theta from the model
arma::vec ung_svm::get_theta() const {
  
  arma::vec theta(3 + xreg.n_cols);

  theta(0) = T(0, 0, 0);
  theta(1) = R(0, 0, 0);
  if(svm_type == 0) {
    theta(2) = phi;
  } else {
    theta(2) = a1(0);
  }
  if(xreg.n_cols > 0) {
    theta.subvec(theta.n_elem - xreg.n_cols, theta.n_elem - 1) = beta;
  }
  return theta;
}
