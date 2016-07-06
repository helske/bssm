#include "svm.h"

//general constructor
svm::svm(arma::vec y, arma::mat Z, arma::cube T,
  arma::cube R, arma::vec a1, arma::mat P1, double mean, double sigma,
  unsigned int seed) :
  ngssm(y, Z, T, R, a1, P1, arma::vec(1), arma::mat(n, 1, arma::fill::ones), arma::vec(mean), 0, seed),
  sigma(sigma), nz_y(y) {

  nz_y(arma::find(abs(y) < 1e-4)) += 1e-4;
}

double svm::proposal(const arma::vec& theta, const arma::vec& theta_prop) {
  return 0.0;
}

void svm::update_model(arma::vec theta) {

  beta(0) = theta(0);
  compute_xbeta();
  T(0, 0, 0) = theta(1);
  sigma = theta(2);
  R(0, 0, 0) = theta(3);
}

arma::vec svm::get_theta(void) {

  arma::vec theta(3);
  theta(0) = beta(0);
  theta(1) = T(0, 0, 0);
  theta(2) = sigma;
  theta(3) = R(0, 0, 0);

  return theta;
}

// compute new values of pseudo y and H given the signal
// and the new signal using Kalman smoothing
arma::vec svm::approx_iter(arma::vec& signal) {

  // new pseudo y and H
  HH = 2 * exp(signal) / pow((nz_y - mean)/sigma, 2);
  y = signal + 1.0 - HH / 2.0;

  // new signal

  arma::mat alpha = fast_smoother();
  arma::vec signal_new(n);

  for (unsigned int t = 0; t < n; t++) {
    signal_new(t) = arma::as_scalar(Z.col(Ztv * t).t() * alpha.col(t) + xbeta(t));
  }
  H = sqrt(HH);

  return signal_new;
}


// log[p(y | signal)]
double svm::logp_y(arma::vec& signal) {

  double logp = 0.0;

  for (unsigned int t = 0; t < n; t++) {
    if (arma::is_finite(y(t))) {
      logp -= 0.5 * (LOG2PI + 2.0 * log(sigma) + signal(t) + pow((ng_y(t) - mean)/sigma, 2) * exp(-signal(t)));
    }
  }

  return logp;
}

