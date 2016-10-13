#include "svm.h"


// from List
svm::svm(const List& model, unsigned int seed) :
  ngssm(model, seed), nz_y(as<arma::vec>(model["y"])), svm_type(model["svm_type"]) {
  nz_y(arma::find(abs(y) < 1e-4)).fill(1e-4);
}

//general constructor
svm::svm(arma::vec y, arma::mat Z, arma::cube T,
  arma::cube R, arma::vec a1, arma::mat P1, double phi,
  arma::mat xreg, arma::vec beta, arma::mat C, unsigned int svm_type, unsigned int seed) :
  ngssm(y, Z, T, R, a1, P1, phi, arma::vec(1), xreg, beta, C, 0, seed, true),
  nz_y(y), svm_type(svm_type) {
  
  nz_y(arma::find(abs(y) < 1e-4)).fill(1e-4);
}

void svm::update_model(arma::vec theta) {
  
  T(0, 0, 0) = theta(0);
  R(0, 0, 0) = theta(1);
  compute_RR();
  P1(0, 0) = theta(1) * theta(1) / (1 - theta(0) * theta(0));
  if(svm_type == 0) {
    phi = theta(2);
  } else {
    C.fill(theta(2) * (1.0 - theta(1)));
  }
  
  
  if(xreg.n_cols > 0) {
    beta = theta.subvec(theta.n_elem - xreg.n_cols, theta.n_elem - 1);
    compute_xbeta();
  }
  
}

arma::vec svm::get_theta(void) {
  arma::vec theta(3 + xreg.n_cols);
  
  theta(0) = T(0, 0, 0);
  theta(1) = R(0, 0, 0);
  if(svm_type == 0) {
    theta(2) = phi;
  } else {
    theta(2) = C(0) / (1.0 - theta(1));
  }
  if(xreg.n_cols > 0) {
    theta.subvec(theta.n_elem - xreg.n_cols, theta.n_elem - 1) = beta;
  }
  return theta;
}

// compute new values of pseudo y and H given the signal
// and the new signal using Kalman smoothing
arma::vec svm::approx_iter(arma::vec& signal) {
  // new pseudo y and H
  HH = 2.0 * exp(signal) / pow((nz_y - xbeta)/phi, 2);
  y = signal + 1.0 - 0.5 * HH;
  H = sqrt(HH);
  
  return arma::vectorise(fast_smoother(false));
}
