#ifndef UNG_BSM_H
#define UNG_BSM_H

#include "ung_ssm.h"

class ung_bsm: public ung_ssm {

public:

  ung_bsm(const Rcpp::List& model, unsigned int seed);

  // update model given the parameters theta
  void set_theta(const arma::vec& theta);
  // extract theta from the model
  arma::vec get_theta() const;

private:
  const bool slope;
  const bool seasonal;
  const bool noise;
  const arma::uvec fixed;
  const bool level_est;
  const bool slope_est;
  const bool seasonal_est;
};

#endif
