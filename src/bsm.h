#ifndef BSM_H
#define BSM_H

#include "gssm.h"

class bsm: public gssm {

public:
  
  bsm(List, unsigned int, bool);
  // constructor
  bsm(arma::vec, arma::mat, arma::vec, arma::cube, arma::cube, arma::vec,
    arma::mat, bool, bool, arma::uvec, arma::mat, arma::vec, unsigned int);

  // constructor with log_space argument
  bsm(arma::vec, arma::mat, arma::vec, arma::cube, arma::cube, arma::vec,
    arma::mat, bool, bool, arma::uvec, arma::mat, arma::vec, unsigned int, bool);
  
  // log[q(y,x)/q(x,y)]
  double proposal(const arma::vec&, const arma::vec&);

  // update model given the parameters theta
  void update_model(arma::vec);
  // extract theta from the model
  arma::vec get_theta(void);

  // log-likelihood
  double log_likelihood(bool);
  // Kalman filtering
  double filter(arma::mat&, arma::mat&, arma::cube&, arma::cube&, bool);

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
