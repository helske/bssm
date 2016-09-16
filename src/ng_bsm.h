#ifndef NGBSM_H
#define NGBSM_H

#include "ngssm.h"

class ng_bsm: public ngssm {

public:


  ng_bsm(const List, unsigned int, bool);

  ng_bsm(arma::vec, arma::mat, arma::cube, arma::cube, arma::vec,
    arma::mat, arma::vec, bool, bool, bool, arma::uvec, arma::mat, arma::vec,
    unsigned int, unsigned int);

  ng_bsm(arma::vec, arma::mat, arma::cube, arma::cube, arma::vec,
    arma::mat, arma::vec, bool, bool, bool, arma::uvec, arma::mat, arma::vec,
    unsigned int, unsigned int, bool);

  double proposal(const arma::vec&, const arma::vec&);
  void update_model(arma::vec);
  arma::vec get_theta(void);

  // log-likelihood of the approximating Gaussian model
  double log_likelihood(bool);

private:
  const bool slope;
  const bool seasonal;
  const bool noise;
  const arma::uvec fixed;
  const bool level_est;
  const bool slope_est;
  const bool seasonal_est;
  const bool log_space;

};

#endif
