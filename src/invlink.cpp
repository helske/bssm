#include "bssm.h"

arma::cube invlink(const arma::cube& alpha, const unsigned int distribution,
  const arma::mat& Z) {
  
  unsigned int nsim = alpha.n_slices;
  unsigned int n = alpha.n_cols;
  unsigned int Ztv(Z.n_cols > 1);
  
  arma::cube y_mean(1, n, nsim);
  
  switch(distribution) {
  case 0  :
    y_mean.zeros();
    break;
  case 1  :
    for (unsigned int i = 0; i < nsim; i++) {
      for (unsigned int t = 0; t < n; t++) {
        y_mean(0, t, i) = arma::as_scalar(
          exp(Z.col(Ztv * t).t() * alpha.slice(i).col(t)));
      }
    }
    break;
  case 2  :
    for (unsigned int i = 0; i < nsim; i++) {
      for (unsigned int t = 0; t < n; t++) {
        double tmp = arma::as_scalar(exp(Z.col(Ztv * t).t() * alpha.slice(i).col(t)));
        y_mean(0, t, i) = tmp / (1.0 + tmp);
      }
    }
    break;
  case 3  :
    for (unsigned int i = 0; i < nsim; i++) {
      for (unsigned int t = 0; t < n; t++) {
        y_mean(0, t, i) = arma::as_scalar(exp(Z.col(Ztv * t).t() * alpha.slice(i).col(t)));
      }
    }
    break;
  }
  return y_mean;
}


arma::cube invlink(const arma::cube& alpha, const unsigned int distribution,
  const arma::mat& Z, const arma::vec& xbeta) {
  
  unsigned int nsim = alpha.n_slices;
  unsigned int n = alpha.n_cols;
  unsigned int Ztv(Z.n_cols > 1);
  
  arma::cube y_mean(1, n, nsim);
  switch(distribution) {
  case 0  :
    for (unsigned int i = 0; i < nsim; i++) {
      for (unsigned int t = 0; t < n; t++) {
        y_mean(0, t, i) = xbeta(t);
      }
    }
    break;
  case 1  :
    for (unsigned int i = 0; i < nsim; i++) {
      for (unsigned int t = 0; t < n; t++) {
        y_mean(0, t, i) = arma::as_scalar(
          exp(xbeta(t) + Z.col(Ztv * t).t() * alpha.slice(i).col(t)));
      }
    }
    break;
  case 2  :
    for (unsigned int i = 0; i < nsim; i++) {
      for (unsigned int t = 0; t < n; t++) {
        double tmp = arma::as_scalar(exp(xbeta(t) + Z.col(Ztv * t).t() * alpha.slice(i).col(t)));
        y_mean(0, t, i) = tmp / (1.0 + tmp);
      }
    }
    break;
  case 3  :
    for (unsigned int i = 0; i < nsim; i++) {
      for (unsigned int t = 0; t < n; t++) {
        y_mean(0, t, i) = arma::as_scalar(exp(xbeta(t) + Z.col(Ztv * t).t() * alpha.slice(i).col(t)));
      }
    }
    break;
  }
  return y_mean;
}


