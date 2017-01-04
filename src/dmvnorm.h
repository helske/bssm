#ifndef DMVNORM_H
#define DMVNORM_H

#include <RcppArmadillo.h>

double dmvnorm(const arma::vec& x, const arma::vec& mean,  
  const arma::mat& sigma, bool lwr, bool logd);

#endif
