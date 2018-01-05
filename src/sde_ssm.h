
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
  
  sde_ssm(const arma::vec& y, const arma::vec& theta, 
    const double x0,bool positive, const unsigned int seed,
    funcPtr drift_, funcPtr diffusion_, 
    funcPtr ddiffusion_, prior_funcPtr log_prior_pdf_,
    obs_funcPtr log_obs_density_);
  
  // bootstrap filter  
  double bsf_filter(const unsigned int nsim, const unsigned int L, 
    arma::cube& alpha, arma::mat& weights, arma::umat& indices);
  
  arma::vec y;
  // Parameter vector used in _all_ functions
  arma::vec theta;
  
  const double x0;
  const unsigned int n;
  bool positive;
  unsigned int seed;
  // PRNG used for simulating Brownian motion on coarse scale
  sitmo::prng_engine coarse_engine;
  // PRNG use for everything else
  sitmo::prng_engine engine;
  
  funcPtr drift;
  funcPtr diffusion;
  funcPtr ddiffusion;
  
  //prior log-pdf
  prior_funcPtr log_prior_pdf;
  //log-pdf for observational level
  obs_funcPtr log_obs_density;
};


#endif