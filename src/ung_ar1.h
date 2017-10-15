#ifndef UNG_AR1_H
#define UNG_AR1_H

#include "ung_ssm.h"

class ung_ar1: public ung_ssm {
  
public:
  
  ung_ar1(const Rcpp::List& model, const unsigned int seed);
  
  // update model given the parameters theta
  void set_theta(const arma::vec& theta);
  // extract theta from the model
  arma::vec get_theta() const;
  
private:
  const bool mu_est;
};

#endif
