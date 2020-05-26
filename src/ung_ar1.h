#ifndef UNG_AR1_H
#define UNG_AR1_H

#include "ung_ssm.h"

class ung_ar1: public ung_ssm {
  
public:
  
  ung_ar1(const Rcpp::List& model, const unsigned int seed);
  
  // update model given the parameters theta
  void update_model(const arma::vec& new_theta);
  double log_prior_pdf(const arma::vec& x) const;
  
private:
  const bool mu_est;
};

#endif
