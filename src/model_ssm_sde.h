
#ifndef SSM_SDE_H
#define SSM_SDE_H

#include <sitmo.h>
#include "bssm.h"

typedef double (*fnPtr)(const double x, const arma::vec& theta);
typedef double (*prior_fnPtr)(const arma::vec& theta);
typedef arma::vec (*obs_fnPtr)(const double y, 
  const arma::vec& alpha, const arma::vec& theta);

class ssm_sde {
  
public:
  
  ssm_sde( 
    const arma::vec& y, 
    const arma::vec& theta, 
    const double x0, 
    bool positive, 
    fnPtr drift_, fnPtr diffusion_, fnPtr ddiffusion_,
    obs_fnPtr log_obs_density_, prior_fnPtr log_prior_pdf_, 
    const unsigned int L_f,
    const unsigned int L_c,
    const unsigned int seed = 1);
  
  arma::vec y;
  // Parameter vector used in _all_ functions
  arma::vec theta;
  
  const double x0;
  const unsigned int n;
  static const unsigned int m = 1; // number of states
  bool positive;
  
  fnPtr drift;
  fnPtr diffusion;
  fnPtr ddiffusion;
  //log-pdf for observational level
  obs_fnPtr log_obs_density;
  //prior log-pdf
  prior_fnPtr log_prior_pdf;
  
  // PRNG used for simulating Brownian motion on coarse scale
  sitmo::prng_engine coarse_engine;
  // PRNG use for everything else
  sitmo::prng_engine engine;
  
  void update_model(const arma::vec& new_theta) {
    theta = new_theta;
  };
  
  // bootstrap filter  
  double bsf_filter(const unsigned int nsim, const unsigned int L,
    arma::cube& alpha, arma::mat& weights, arma::umat& indices);
  
  const unsigned int L_f;
  const unsigned int L_c; 
  
};


#endif
