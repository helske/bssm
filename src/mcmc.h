#ifndef MCMC_H
#define MCMC_H

#include "bssm.h"

#include "ugg_ssm.h"
#include "ung_ssm.h"

#include "nlg_ssm.h"
#include "sde_ssm.h"
#include "mng_ssm.h"
#include "lgg_ssm.h"

class mcmc {
  
protected:
  
  virtual void trim_storage();
  
  const unsigned int n_iter;
  const unsigned int n_burnin;
  const unsigned int n_thin;
  const unsigned int n_samples;
  const unsigned int n_par;
  const double target_acceptance;
  const double gamma;
  unsigned int n_stored;
  
public:
  
  // constructor
  mcmc(const unsigned int n_iter, const unsigned int n_burnin, 
    const unsigned int n_thin, const unsigned int n, const unsigned int m,
    const double target_acceptance, const double gamma, const arma::mat& S, 
    const unsigned int output_type = 1);

  // sample states given theta
  template <class T>
  void state_posterior(T model, const unsigned int n_threads);
  template <class T>
  void state_summary(T& model, arma::mat& alphahat, arma::cube& Vt);
  template <class T>
  void state_sampler(T& model, const arma::mat& theta, arma::cube& alpha);

  // gaussian mcmc
  template<class T>
  void mcmc_gaussian(T& model, const bool end_ram);

  // pseudo-marginal mcmc
  template<class T>
  void pm_mcmc(T& model, const unsigned int method, const unsigned int nsim_states, const bool end_ram);

  // delayed acceptance mcmc
  template<class T>
  void da_mcmc(T& model, const unsigned int method, const unsigned int nsim_states, const bool end_ram);

  arma::vec posterior_storage;
  arma::mat theta_storage;
  arma::uvec count_storage;
  arma::cube alpha_storage;
  arma::mat alphahat;
  arma::cube Vt;
  arma::mat S;
  double acceptance_rate;
  unsigned int output_type;
  
};


#endif
