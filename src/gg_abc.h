// Abstract base class for linear Gaussian state space models
#ifndef GG_ABC_H
#define GG_ABC_H

#include <RcppArmadillo.h>

class gg_abc {
  
public:
  
  // update model matrices
  virtual void set_theta(const arma::vec& theta) = 0;
  // get current value of theta
  virtual arma::vec get_theta() const = 0;
  // compute the log-likelihood
  virtual double log_likelihood() const = 0;
};


#endif
