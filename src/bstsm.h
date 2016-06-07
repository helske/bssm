#ifndef BSTSM_H
#define BSTSM_H

#include "guvssm.h"

class bstsm: public guvssm {

public:

  bstsm(arma::vec, arma::mat, arma::vec, arma::cube, arma::cube, arma::vec,
    arma::mat, bool, bool, arma::uvec, arma::mat, arma::vec, unsigned int);

  bstsm(arma::vec, arma::mat, arma::vec, arma::cube, arma::cube, arma::vec,
    arma::mat, bool, bool, arma::uvec, arma::mat, arma::vec, unsigned int, bool);

  double proposal(const arma::vec&, const arma::vec&);
  void update_model(arma::vec);
  arma::vec get_theta(void);

  virtual double log_likelihood(void);
  virtual double filter(arma::mat&, arma::mat&, arma::cube&, arma::cube&);

private:
  const bool slope;
  const bool seasonal;
  const arma::uvec fixed;
  const bool level_est;
  const bool slope_est;
  const bool seasonal_est;
  const bool log_space;

};

#endif
