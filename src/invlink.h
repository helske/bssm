#ifndef INVLINK_H
#define INVLINK_H


#include <RcppArmadillo.h>

arma::cube invlink(const arma::cube& alpha, const unsigned int distribution,
  const arma::mat& Z);

arma::cube invlink(const arma::cube& alpha, const unsigned int distribution,
  const arma::mat& Z, const arma::vec& xbeta);

#endif
