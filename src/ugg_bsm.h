#ifndef UGG_BSM_H
#define UGG_BSM_H

#include "ugg_ssm.h"

class ugg_bsm: public ugg_ssm {

public:

  ugg_bsm(const Rcpp::List& model, const unsigned int seed);

  // update model given the parameters theta
  void set_theta(const arma::vec& theta);
  // extract theta from the model
  arma::vec get_theta() const;

private:
  const bool slope;
  const bool seasonal;
  const arma::uvec fixed;
  const bool y_est;
  const bool level_est;
  const bool slope_est;
  const bool seasonal_est;

};

#endif
