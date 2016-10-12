#include "svm.h"

// from List
svm::svm(const List& model, unsigned int seed) :
  ngssm(model, seed), nz_y(as<arma::vec>(model["y"])) {
  nz_y(arma::find(abs(y) < 1e-4)).fill(1e-4);
}

// from List, with gkl
svm::svm(const List& model, unsigned int seed, bool gkl) :
  ngssm(model, seed), nz_y(as<arma::vec>(model["y"])), gkl(gkl) {
  nz_y(arma::find(abs(y) < 1e-4)).fill(1e-4);
}

//general constructor
svm::svm(arma::vec y, arma::mat Z, arma::cube T,
  arma::cube R, arma::vec a1, arma::mat P1, double phi,
  arma::mat xreg, arma::vec beta, unsigned int seed) :
  ngssm(y, Z, T, R, a1, P1, phi, arma::vec(1), xreg, beta, 0, seed, true),
  nz_y(y) {
  
  nz_y(arma::find(abs(y) < 1e-4)).fill(1e-4);
}

void svm::update_model(arma::vec theta) {
  
  if(gkl) {
    theta(0) = 2.0 * theta(0) - 1.0;
    theta(1) = sqrt(theta(1));
    theta(2) = exp(theta(2));
  }
  T(0, 0, 0) = theta(0);
  R(0, 0, 0) = theta(1);
  compute_RR();
  P1(0, 0) = theta(1) * theta(1) / (1 - theta(0) * theta(0));
  phi = theta(2);
  
  if(xreg.n_cols > 0) {
    beta = theta.subvec(theta.n_elem - xreg.n_cols, theta.n_elem - 1);
    compute_xbeta();
  }
}

arma::vec svm::get_theta(void) {
  arma::vec theta(3 + xreg.n_cols);
  
  theta(0) = T(0, 0, 0);
  theta(1) = R(0, 0, 0);
  theta(2) = phi;
  
  if(gkl) {
    theta(0) = 0.5 * (theta(0) + 1.0);
    theta(1) = theta(1) * theta(1);
    theta(2) = log(theta(2));
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


// log[p(y | signal)]
double svm::logp_y(arma::vec& signal) {
  
  double logp = 0.0;
  
  for (unsigned int t = 0; t < n; t++) {
    if (arma::is_finite(y(t))) {
      logp -= 0.5 * (LOG2PI + 2.0 * log(phi) + signal(t) + pow((ng_y(t) - xbeta(t))/phi, 2) * exp(-signal(t)));
    }
  }
  
  return logp;
}



double svm::prior_pdf(const arma::vec& theta, const arma::uvec& prior_types,
  const arma::mat& params) {
  
  double q = 0.0;
  if(gkl) {
    if(theta(1) <= 0.0) {
      q = -arma::datum::inf;
    } else {
      double v = 5.0;
      double tau2 = 0.01;
      q = R::dbeta(theta(0), 7.768667, 1.264667, 1);
      q += v * log(v) - log(std::lgamma(v)) + v * log(tau2) - 
        (v + 1.0) * log(theta(1)) - (v * tau2 / theta(1));
    }
  } else {
    for(unsigned int i = 0; i < theta.n_elem; i++) {
      switch(prior_types(i)) {
      case 0  :
        q += R::dunif(theta(i), params(0, i), params(1, i), 1);
        break;
      case 1  :
        if (theta(i) < 0) {
          return -arma::datum::inf;
        } else {
          q += log(2.0) + R::dnorm(theta(i), 0, params(0, i), 1);
        }
        break;
      case 2  :
        q += R::dnorm(theta(i), params(0, i), params(1, i), 1);
        break;
      }
    }
  }
  return q;
}

