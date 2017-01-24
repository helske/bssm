// Abstract base class for all univariate non-linear and/or non-Gaussian state space models
// supported in bssm

#ifndef UNN_ABC_H
#define UNN_ABC_H

#include <RcppArmadillo.h>
#include "ugg_ssm.h"

class unn_abc {
  
public:
  // update model given the parameter vector theta
  virtual void set_theta(const arma::vec& theta) = 0;
  // get the current values of theta
  virtual arma::vec get_theta() const = 0;
  
  // find the approximating Gaussian model
  virtual ugg_ssm approximate(arma::vec& signal, const unsigned int max_iter, 
    const double conv_tol) const = 0;
  
  // psi-particle filter
  double psi_filter(const ugg_ssm& approx_model,
    const double approx_loglik, const arma::vec& scales,
    const unsigned int nsim, arma::cube& alpha, arma::mat& weights,
    arma::umat& indices);
  
  // compute logarithms of _unnormalized_ importance weights g(y_t | alpha_t) / ~g(~y_t | alpha_t)
  arma::vec log_weights(const ugg_ssm& approx_model, 
    const unsigned int t, const arma::cube& alphasim) const;
  
  // compute unnormalized mode-based scaling terms
  // log[g(y_t | ^alpha_t) / ~g(y_t | ^alpha_t)]
  arma::vec scaling_factors(const ugg_ssm& approx_model, const arma::vec& mode_estimate) const;
  
};


#endif
