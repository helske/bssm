#ifndef UGG_BSM_H
#define UGG_BSM_H

#include "ugg_ssm.h"

class ugg_bsm: public ugg_ssm {

public:

  ugg_bsm(const Rcpp::List& model, const unsigned int seed);

  // update model given the parameters theta
  void update_model(const arma::vec& new_theta);
  double log_prior_pdf(const arma::vec& x) const;
  double log_proposal_ratio(const arma::vec& new_theta, const arma::vec& old_theta) const;

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
