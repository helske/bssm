#ifndef DMVNORM_H
#define DMVNORM_H

#include "bssm.h"

double dmvnorm(const arma::vec& x, const arma::vec& mean,  
  const arma::mat& sigma, bool lwr, bool logd);
double precompute_dmvnorm(const arma::mat& sigma, arma::mat& Linv, 
  const arma::uvec& nonzero);
double fast_dmvnorm(const arma::vec& x, const arma::vec& mean, 
  const arma::mat& Linv, const arma::uvec& nonzero, const double constant);
#endif
