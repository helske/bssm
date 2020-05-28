#ifndef ar1_ng_H
#define ar1_ng_H

#include "model_ssm_ung.h"

class ar1_ng: public ssm_ung {
  
public:
  
  ar1_ng(const Rcpp::List& model, const unsigned int seed);
  
  // update model given the parameters theta
  void update_model(const arma::vec& new_theta);  
  double log_prior_pdf(const arma::vec& x);
  
private:
  const arma::uvec prior_distributions;
  const arma::mat prior_parameters;
  const bool mu_est;
  const bool phi_est;
};

#endif
