#ifndef ar1_lg_H
#define ar1_lg_H

#include "model_ssm_ulg.h"

class ar1_lg: public ssm_ulg {
  
public:
  
  ar1_lg(const Rcpp::List& model, const unsigned int seed);
  
  // update model given the parameters theta
  void update_model(const arma::vec& new_theta);
  
  double log_prior_pdf(const arma::vec& x);
  
private:
  const arma::uvec prior_distributions;
  const arma::mat prior_parameters;
  const bool mu_est;
  const bool sd_y_est;
  
};

#endif
