// constants (independent of states) parts of distributions
// use same notation as in bssm models expect norm/svm

#include "bssm.h"
#include "distr_consts.h"

double norm_log_const(double sd) {
  return -0.5 * log(2.0 * M_PI) - log(sd);
}

double poisson_log_const(double y, double u) {
  return -lgamma(y + 1) + y * log(u);
}

double binomial_log_const(double y, double u) {
  return R::lchoose(u, y);
}

double negbin_log_const(double y, double u, double phi) {
  return R::lchoose(y + phi - 1, y) + phi * log(phi) + y * log(u);
}


double norm_log_const(const arma::vec& y, const arma::vec& sd) {
  return -0.5 * y.n_elem * log(2.0 * M_PI) - arma::accu(log(sd));
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

double negbin_log_const(const arma::vec&  y, const arma::vec& u, double phi) {
  double res = 0.0;
  for(unsigned int i = 0; i < y.n_elem; i++) {
    res += R::lchoose(y(i) + phi - 1, y(i)) + phi*log(phi) + y(i) * log(u(i));
  }
  return res;
}

double compute_const_term(const ung_ssm& model, const ugg_ssm& approx_model) {
  
  double const_term = 0.0;
  switch(model.distribution) {
  case 0 :
    const_term = arma::uvec(arma::find_finite(model.y)).n_elem * norm_log_const(model.phi);
    break;
  case 1 : {
      arma::uvec y_ind(find_finite(model.y));
      const_term = poisson_log_const(model.y(y_ind), model.u(y_ind));
    } break;
  case 2 : {
    arma::uvec y_ind(find_finite(model.y));
    const_term = binomial_log_const(model.y(y_ind), model.u(y_ind));
  } break;
  case 3 : {
    arma::uvec y_ind(find_finite(model.y));
    const_term = negbin_log_const(model.y(y_ind), model.u(y_ind), model.phi);
  } break;
  }
  return const_term - norm_log_const(approx_model.y, approx_model.H);
}