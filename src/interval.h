#ifndef INTV_H
#define INTV_H

#include <RcppArmadillo.h>


arma::mat intervals(arma::mat& means, const arma::mat& sds, const arma::vec& probs, 
  unsigned int n_ahead);


#endif
