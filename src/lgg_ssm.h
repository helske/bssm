// arbitrary linear gaussian state space model with time-invariant functions

#ifndef LGG_SSM_H
#define LGG_SSM_H

#include <sitmo.h>
#include "bssm.h"
#include "mgg_ssm.h"
#include "function_pointers.h"


class lgg_ssm {
  
public:
  
  lgg_ssm(const arma::mat& y, SEXP Z_fn_, SEXP H_fn_, SEXP T_fn_, SEXP R_fn_, 
    SEXP a1_fn_, SEXP P1_fn_, SEXP D_fn_, SEXP C_fn_, 
    const arma::vec& theta, SEXP log_prior_pdf_, const arma::vec& known_params, 
    const arma::mat& known_tv_params, const unsigned int m, const unsigned int k,
    const unsigned int seed);
  
  mgg_ssm build_mgg();
  
  // linear functions of 
  // y_t = Z(alpha_t, theta,t) + H(theta,t)*eps_t, 
  // alpha_t+1 = T(alpha_t, theta,t) + R(theta, t)*eta_t
  
  arma::mat y;
  mat_fn2 Z_fn;
  mat_varfn H_fn;
  mat_fn2 T_fn;
  mat_varfn R_fn;
  //initial value
  vec_initfn a1_fn;
  mat_initfn P1_fn;
  
  vec_fn2 D_fn;
  vec_fn2 C_fn;
  
  // Parameter vector used in _all_ linear functions
  arma::vec theta;
  //prior log-pdf
  double_fn log_prior_pdf;
  // vector of known parameters
  arma::vec known_params;
  // matrix of known (time-varying) parameters
  arma::mat known_tv_params;
  
  const unsigned int m;
  const unsigned int k;
  const unsigned int n;
  const unsigned int p;
  
  
  unsigned int seed;
  sitmo::prng_engine engine;
  const double zero_tol;
  
};


#endif
