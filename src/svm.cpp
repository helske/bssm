#include "svm.h"

//general constructor
svm::svm(arma::vec y, arma::mat Z, arma::cube T,
  arma::cube R, arma::vec a1, arma::mat P1, arma::vec phi,
  arma::mat xreg, arma::vec beta, unsigned int seed, bool log_space) :
  ngssm(y, Z, T, R, a1, P1, phi, xreg, beta, 0, seed),
  nz_y(y), log_space(log_space) {

  nz_y(arma::find(abs(y) < 1e-4)).fill(1e-4);
}

svm::svm(arma::vec y, arma::mat Z, arma::cube T,
  arma::cube R, arma::vec a1, arma::mat P1, arma::vec phi,
  arma::mat xreg, arma::vec beta, unsigned int seed) :
  ngssm(y, Z, T, R, a1, P1, phi, xreg, beta, 0, seed),
  nz_y(y), log_space(false) {

  nz_y(arma::find(abs(y) < 1e-4)).fill(1e-4);
}

double svm::proposal(const arma::vec& theta, const arma::vec& theta_prop) {
  double q = 0.0;
  if (log_space) {
    q = theta_prop(1) - theta(1) + theta_prop(2) - theta(2);
  }
  return q;
}

void svm::update_model(arma::vec theta) {

  if (log_space) {
    T(0, 0, 0) = theta(0);
    R(0, 0, 0) = exp(theta(1));
    compute_RR();
    P1(0, 0) = R(0, 0, 0) * R(0, 0, 0) / (1 - theta(0) * theta(0));
    phi.fill(exp(theta(2)));
  } else {
    T(0, 0, 0) = theta(0);
    R(0, 0, 0) = theta(1);
    compute_RR();
    P1(0, 0) = theta(1) * theta(1) / (1 - theta(0) * theta(0));
    phi.fill(theta(2));
  }
  if(xreg.n_cols > 0) {
    beta = theta.subvec(theta.n_elem - xreg.n_cols, theta.n_elem - 1);
    compute_xbeta();
  }

}

arma::vec svm::get_theta(void) {
  arma::vec theta(3 + xreg.n_cols);

  if (log_space) {
    theta(0) = T(0, 0, 0);
    theta(1) = log(R(0, 0, 0));
    theta(2) = log(phi(0));
  } else {
    theta(0) = T(0, 0, 0);
    theta(1) = R(0, 0, 0);
    theta(2) = phi(0);
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
  HH = 2 * exp(signal) / pow((nz_y - xbeta)/phi(0), 2);
  y = signal + 1.0 - 0.5 * HH;
  // new signal

  // arma::mat alpha = arma::vectorise(fast_smoother(false));
  // arma::vec signal_new = alpha.row(0).t();

  // for (unsigned int t = 0; t < n; t++) {
  //   signal_new(t) = arma::as_scalar(Z.col(Ztv * t).t() * alpha.col(t));
  // }
  H = sqrt(HH);

  return arma::vectorise(fast_smoother(false));
}


// log[p(y | signal)]
double svm::logp_y(arma::vec& signal) {

  double logp = 0.0;

  for (unsigned int t = 0; t < n; t++) {
    if (arma::is_finite(y(t))) {
      logp -= 0.5 * (LOG2PI + 2.0 * log(phi(0)) + signal(t) + pow((ng_y(t) - xbeta(t))/phi(0), 2) * exp(-signal(t)));
    }
  }

  return logp;
}

