// constants (independent of states) parts of distributions
// use same notation as in bssm models expect norm/svm

#include "distr_consts.h"

// thread-safe (does not call R::)
double lchoose(double n, double k) {
  return -std::log(n + 1) - std::lgamma(n - k + 1) - std::lgamma(k + 1) + std::lgamma(n + 2);
};

double norm_log_const(double sd) {
  return -0.5 * std::log(2.0 * M_PI) - std::log(sd);
}

double poisson_log_const(double y, double u) {
  return -std::lgamma(y + 1) + y * std::log(u);
}

double binomial_log_const(double y, double u) {
  // was R::lchoose(u, y);, not necessarily thread safe!
  return lchoose(u, y);
}

double negbin_log_const(double y, double u, double phi) {
  return lchoose(y + phi - 1, y) + phi * std::log(phi) + y * std::log(u);
}

double gamma_log_const(double y, double u, double phi) {
  return phi * std::log(phi) - std::lgamma(phi) + (phi - 1) * std::log(y) - phi * std::log(u);
}


double norm_log_const(const arma::vec& y, const arma::vec& sd) {
  return -0.5 * y.n_elem * std::log(2 * M_PI) - arma::accu(log(sd));
}

double poisson_log_const(const arma::vec& y, const arma::vec& u) {
  double res = 0;
  for(unsigned int i = 0; i < y.n_elem; i++) {
    res += y(i) * std::log(u(i)) - std::lgamma(y(i) + 1) ;
  }
  return res;
}

double binomial_log_const(const arma::vec& y, const arma::vec& u) {
  double res = 0;
  for(unsigned int i = 0; i < y.n_elem; i++) {
    res += lchoose(u(i), y(i));
  }
  return res;
}

double negbin_log_const(const arma::vec&  y, const arma::vec& u, double phi) {
  double res = 0;
  for(unsigned int i = 0; i < y.n_elem; i++) {
    res += lchoose(y(i) + phi - 1, y(i)) + phi * std::log(phi) + y(i) * std::log(u(i));
  }
  return res;
}

double gamma_log_const(const arma::vec&  y, const arma::vec& u, double phi) {
  double res = 0;
  for(unsigned int i = 0; i < y.n_elem; i++) {
    res += phi * std::log(phi) - std::lgamma(phi) + (phi - 1) * std::log(y(i)) - phi * std::log(u(i));
  }
  return res;
}
