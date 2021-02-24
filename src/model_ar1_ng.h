#ifndef AR1_NG_H
#define AR1_NG_H

#include "model_ssm_ung.h"
extern Rcpp::Function default_prior_fn;

class ar1_ng: public ssm_ung {
  
public:
  
  ar1_ng(const Rcpp::List model, const unsigned int seed);
  
  // update model given the parameters theta
  void update_model(const arma::vec& new_theta); 
  void update_model(const arma::vec& new_theta, const Rcpp::Function update_fn);  
  double log_prior_pdf(const arma::vec& x, const Rcpp::Function prior_fn = default_prior_fn) const;
  
private:
  const arma::uvec prior_distributions;
  const arma::mat prior_parameters;
  const bool mu_est;
  const bool phi_est;
};

#endif
