// arbitrary nonlinear gaussian state space model with time-invariant functions

#ifndef NLG_SSM_H
#define NLG_SSM_H

#include <RcppArmadillo.h>
#include "mgg_ssm.h"
#include "nl_func.h"

class nlg_ssm {
  
public:
  
  nlg_ssm(const arma::mat& y, const arma::vec a1, const arma::mat& P1,
    SEXP Z_fn_, SEXP H_fn_, SEXP T_fn_, SEXP R_fn_, 
    SEXP Z_gn_, SEXP H_gn_, SEXP T_gn_, SEXP R_gn_, 
    const arma::vec& theta, const unsigned int seed);
  
  // find the approximating Gaussian model
  mgg_ssm approximate(arma::mat& mode_estimate, const unsigned int max_iter, 
    const double conv_tol) const;
  
  // psi-particle filter
  double psi_filter(const mgg_ssm& approx_model,
    const double approx_loglik, const arma::vec& scales,
    const unsigned int nsim, arma::cube& alpha, arma::mat& weights,
    arma::umat& indices);
  
  // compute logarithms of _unnormalized_ importance weights g(y_t | alpha_t) / ~g(~y_t | alpha_t)
  arma::vec log_weights(const mgg_ssm& approx_model, 
    const unsigned int t, const arma::cube& alphasim) const;
  
  // compute unnormalized mode-based scaling terms
  // log[g(y_t | ^alpha_t) / ~g(y_t | ^alpha_t)]
  arma::vec scaling_factors(const mgg_ssm& approx_model, const arma::mat& mode_estimate) const;
  
  // compute logarithms of _unnormalized_ densities g(y_t | alpha_t)
  arma::vec log_obs_density(const unsigned int t, const arma::cube& alphasim) const;
  
  // bootstrap filter  
  double bsf_filter(const unsigned int nsim, arma::cube& alphasim, 
    arma::mat& weights, arma::umat& indices);
  
  
  arma::mat y;
  arma::vec a1;
  arma::mat P1;
  
  const unsigned int n;
  const unsigned int m;
  const unsigned int p;

  std::mt19937 engine;
  const double zero_tol;
  unsigned int seed;
  
  // Parameter vector used in _all_ nonlinear functions
  arma::vec theta;
  // nonlinear functions of 
  // y_t = Z(alpha_t, theta_t) + H(alpha_t, theta_t)*eps_t, 
  // alpha_t+1 = T(alpha_t, theta_t) + R(alpha_t, theta_t)*eta_t
  nonlinear_fn Z_fn;
  nonlinear_fn H_fn;
  nonlinear_fn T_fn;
  nonlinear_fn R_fn;
  //and the derivatives
  nonlinear_gn Z_gn;
  nonlinear_gn H_gn;
  nonlinear_gn T_gn;
  nonlinear_gn R_gn;
  
};


#endif
