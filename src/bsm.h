#ifndef BSM_H
#define BSM_H

#include "gssm.h"

class bsm: public gssm {

public:

  bsm(arma::vec, arma::mat, arma::vec, arma::cube, arma::cube, arma::vec,
    arma::mat, bool, bool, arma::uvec, arma::mat, arma::vec, unsigned int);

  bsm(arma::vec, arma::mat, arma::vec, arma::cube, arma::cube, arma::vec,
    arma::mat, bool, bool, arma::uvec, arma::mat, arma::vec, unsigned int, bool);

  double proposal(const arma::vec&, const arma::vec&);
  void update_model(arma::vec);
  arma::vec get_theta(void);

  double log_likelihood(void);
  double filter(arma::mat&, arma::mat&, arma::cube&, arma::cube&);

private:
  const bool slope;
  const bool seasonal;
  const arma::uvec fixed;
  const bool level_est;
  const bool slope_est;
  const bool seasonal_est;
  const bool log_space;

};

arma::cube sample_states(bsm mod, const arma::mat& theta,
  unsigned int nsim_states, unsigned int n_threads, arma::uvec seeds);

#endif
