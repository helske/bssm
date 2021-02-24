#ifndef BSM_LG_H
#define BSM_LG_H

#include "model_ssm_ulg.h"

extern Rcpp::Function default_prior_fn;

class bsm_lg: public ssm_ulg {

public:

  bsm_lg(const Rcpp::List model, const unsigned int seed);

  // update model given the parameters theta
  void update_model(const arma::vec& new_theta);
  void update_model(const arma::vec& new_theta, const Rcpp::Function update_fn);
  double log_prior_pdf(const arma::vec& x, const Rcpp::Function prior_fn = default_prior_fn) const;

private:
  const arma::uvec prior_distributions;
  const arma::mat prior_parameters;
  const bool slope;
  const bool seasonal;
  const arma::uvec fixed;
  const bool y_est;
  const bool level_est;
  const bool slope_est;
  const bool seasonal_est;
};

#endif
