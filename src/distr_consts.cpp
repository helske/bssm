#include <RcppArmadillo.h>
// constants (independent of states) for particle filter
// use same notation as in bssm models expect norm/svm

using namespace Rcpp;

double norm_log_const(const double sd) {
  return -0.5 * log(2.0 * M_PI) - log(sd);
}

double poisson_log_const(const double y, const double u) {
  return -lgamma(y + 1) + y * log(u);
}

double binomial_log_const(const double y, const double u) {
  return R::lchoose(u, y);
}

double negbin_log_const(const double y, const double u, const double phi) {
  return R::lchoose(y + phi - 1, y) + phi * log(phi) + y * log(u);
}

double poisson_log_const(const arma::vec& y, const arma::vec& u) {
  double res = 0.0;
  for(unsigned int i = 0; i < y.n_elem; i++) {
    res += y(i) * log(u(i)) - lgamma(y(i) + 1) ;
  }
  return res;
}

double binomial_log_const(const arma::vec& y, const arma::vec& u) {
  double res = 0.0;
  for(unsigned int i = 0; i < y.n_elem; i++) {
    res += R::lchoose(u(i), y(i));
  }
  return res;
}

double negbin_log_const(const arma::vec&  y, const arma::vec& u, const double phi) {
  double res = 0.0;
  for(unsigned int i = 0; i < y.n_elem; i++) {
    res += R::lchoose(y(i) + phi - 1, y(i)) + phi*log(phi) + y(i) * log(u(i));
  }
  return res;
}
