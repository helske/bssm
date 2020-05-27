#ifndef UGG_AR1_H
#define UGG_AR1_H

#include "model_ugg_ssm.h"

class ugg_ar1: public ugg_ssm {
  
public:
  
  ugg_ar1(const Rcpp::List& model, const unsigned int seed);
  
  // update model given the parameters theta
  void update_model(const arma::vec& new_theta);
  
  double log_prior_pdf(const arma::vec& x) const;
  
private:
  const arma::uvec prior_distributions;
  const arma::mat prior_parameters;
  const bool mu_est;
  const bool sd_y_est;
  
};

#endif
