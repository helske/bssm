// Abstract base class for all multivariate non-linear and/or non-Gaussian state space models
// supported in bssm

#ifndef MNN_ABC_H
#define MNN_ABC_H

#include <RcppArmadillo.h>
#include "mgg_ssm.h"

class mnn_abc {
  
public:
  // update model given the parameter vector theta
  virtual void set_theta(const arma::vec& theta) = 0;
  // get the current values of theta
  virtual arma::vec get_theta() const = 0;
 
  // find the approximating Gaussian model
  virtual mgg_ssm approximate(arma::mat& mode_estimate, const unsigned int max_iter, 
    const double conv_tol) const = 0;
  
  // psi-particle filter
  virtual double psi_filter(const mgg_ssm& approx_model,
    const double approx_loglik, const arma::vec& scales,
    const unsigned int nsim, arma::cube& alpha, arma::mat& weights,
    arma::umat& indices) = 0;
  
  // compute logarithms of _unnormalized_ importance weights g(y_t | alpha_t) / ~g(~y_t | alpha_t)
  virtual arma::vec log_weights(const mgg_ssm& approx_model, 
    const unsigned int t, const arma::cube& alphasim) const = 0;
  
  // compute unnormalized mode-based scaling terms
  // log[g(y_t | ^alpha_t) / ~g(y_t | ^alpha_t)]
  virtual arma::vec scaling_factors(const mgg_ssm& approx_model, const arma::mat& mode_estimate) const = 0;
  
};


#endif
