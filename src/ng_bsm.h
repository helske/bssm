#ifndef NGBSM_H
#define NGBSM_H

#include "ngssm.h"

class ng_bsm: public ngssm {

public:

  ng_bsm(const Rcpp::List&, unsigned int, bool);

  double proposal(const arma::vec&, const arma::vec&);
  void update_model(arma::vec);
  arma::vec get_theta(void);

private:
  const bool slope;
  const bool seasonal;
  const bool noise;
  const arma::uvec fixed;
  const bool level_est;
  const bool slope_est;
  const bool seasonal_est;
  const bool log_space;
  const arma::vec noise_const;

};

#endif
