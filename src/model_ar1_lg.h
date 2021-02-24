#ifndef AR1_LG_H
#define AR1_LG_H

#include "model_ssm_ulg.h"

extern Rcpp::Function default_prior_fn;

class ar1_lg: public ssm_ulg {
  
public:
  
  ar1_lg(const Rcpp::List model, const unsigned int seed);
  
  // update model given the parameters theta
  void update_model(const arma::vec& new_theta);
  void update_model(const arma::vec& new_theta, const Rcpp::Function update_fn);
  
  double log_prior_pdf(const arma::vec& x, const Rcpp::Function prior_fn = default_prior_fn)  const;
  
private:
  const arma::uvec prior_distributions;
  const arma::mat prior_parameters;
  const bool mu_est;
  const bool sd_y_est;
  
};

#endif
