
#ifndef SDE_SSM_H
#define SDE_SSM_H

#include <sitmo.h>
#include "bssm.h"

typedef double (*funcPtr)(const double x, const arma::vec& theta);
typedef double (*prior_funcPtr)(const arma::vec& theta);
typedef arma::vec (*obs_funcPtr)(const double y, 
  const arma::vec& alpha, const arma::vec& theta);

class sde_ssm {
  
public:
  
  sde_ssm( 
    const arma::vec& y, 
    const arma::vec& theta, 
    const double x0, 
    bool positive, 
    funcPtr drift_, funcPtr diffusion_, funcPtr ddiffusion_,
    obs_funcPtr log_obs_density_, prior_funcPtr log_prior_pdf_, 
    const unsigned int seed = 1);
  
  arma::vec y;
  // Parameter vector used in _all_ functions
  arma::vec theta;
  
  const double x0;
  const unsigned int n;
  static const unsigned int m = 1; // number of states
  bool positive;
  
  funcPtr drift;
  funcPtr diffusion;
  funcPtr ddiffusion;
  //log-pdf for observational level
  obs_funcPtr log_obs_density;
  //prior log-pdf
  prior_funcPtr log_prior_pdf;
  
  // PRNG used for simulating Brownian motion on coarse scale
  sitmo::prng_engine coarse_engine;
  // PRNG use for everything else
  sitmo::prng_engine engine;
  
  void update_model(const arma::vec& new_theta) {
    theta = new_theta;
  };
  
  arma::vec log_likelihood(
      const unsigned int method, 
      const unsigned int nsim_states, 
      arma::cube& alpha, 
      arma::mat& weights, 
      arma::umat& indices);
  
  // bootstrap filter  
  double bsf_filter(const unsigned int nsim, const unsigned int L, 
    arma::cube& alpha, arma::mat& weights, arma::umat& indices);
  
 
};


#endif
