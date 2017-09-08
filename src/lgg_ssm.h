// arbitrary linear gaussian state space model with time-invariant functions

#ifndef LGG_SSM_H
#define LGG_SSM_H

#include <sitmo.h>
#include "bssm.h"
#include "mgg_ssm.h"


// typedef for a pointer of linear function of lgg-model equation returning Z, H, T, and R
typedef arma::mat (*lmat_fnPtr)(const unsigned int t, const arma::vec& theta, 
  const arma::vec& known_params, const arma::mat& known_tv_params);
// typedef for a pointer of linear function of lgg-model equation returning D and C
typedef arma::vec (*lvec_fnPtr)(const unsigned int t, const arma::vec& theta, 
  const arma::vec& known_params, const arma::mat& known_tv_params);

// typedef for a pointer returning a1
typedef arma::vec (*a1_fnPtr)(const arma::vec& theta, const arma::vec& known_params);
// typedef for a pointer returning P1
typedef arma::mat (*P1_fnPtr)(const arma::vec& theta, const arma::vec& known_params);
// typedef for a pointer of log-prior function
typedef double (*prior_fnPtr)(const arma::vec&);


class lgg_ssm {
  
public:
  
  lgg_ssm(const arma::mat& y, lmat_fnPtr Z_fn_, lmat_fnPtr H_fn_, lmat_fnPtr T_fn_, lmat_fnPtr R_fn_, 
    a1_fnPtr a1_fn_, P1_fnPtr P1_fn_, lvec_fnPtr D_fn_, lvec_fnPtr C_fn_, 
    const arma::vec& theta, prior_fnPtr log_prior_pdf_, const arma::vec& known_params, 
    const arma::mat& known_tv_params, const unsigned int m, const unsigned int k,
    const unsigned int seed);
  
  mgg_ssm build_mgg();
  void update_mgg(mgg_ssm& model);
  // linear functions of 
  // y_t = Z(alpha_t, theta,t) + H(theta,t)*eps_t, 
  // alpha_t+1 = T(alpha_t, theta,t) + R(theta, t)*eta_t
  
  arma::mat y;
  lmat_fnPtr Z_fn;
  lmat_fnPtr H_fn;
  lmat_fnPtr T_fn;
  lmat_fnPtr R_fn;
  //initial value
  a1_fnPtr a1_fn;
  P1_fnPtr P1_fn;
  
  lvec_fnPtr D_fn;
  lvec_fnPtr C_fn;
  
  // Parameter vector used in _all_ linear functions
  arma::vec theta;
  //prior log-pdf
  prior_fnPtr log_prior_pdf;
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
