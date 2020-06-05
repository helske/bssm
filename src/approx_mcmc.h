#ifndef APPROX_MCMC_H
#define APPROX_MCMC_H

#include "bssm.h"
#include "mcmc.h"
#include "model_ssm_nlg.h"
#include "model_ssm_sde.h"

class approx_mcmc: public mcmc {

public:

  // constructor
  approx_mcmc(const unsigned int iter, const unsigned int burnin, const unsigned int thin,
    const unsigned int n, const unsigned int m, const unsigned int p, 
    const double target_acceptance, const double gamma, const arma::mat& S, 
    const unsigned int output_type = 1, const bool store_modes = true);

  void expand();

  //approximate mcmc
  template<class T>
  void amcmc(T model, const unsigned int method, const bool end_ram);

  void amcmc(ssm_sde model, const unsigned int nsim, const bool end_ram);
    
  template <class T>
  void is_correction_psi(T model, const unsigned int nsim,
    const unsigned int is_type, const unsigned int n_threads);

  template <class T>
  void is_correction_bsf(T model, const unsigned int nsim,
    const unsigned int is_type, const unsigned int n_threads);

  template <class T>
  void is_correction_spdk(T model, const unsigned int nsim,
    const unsigned int is_type, const unsigned int n_threads);
  
  template <class T>
  void approx_state_posterior(T model, const unsigned int n_threads);
  
  template <class T>
  void approx_state_summary(T model);
    
  void ekf_state_summary(ssm_nlg model);
  
  void ekf_state_sample(ssm_nlg model, const unsigned int n_threads);
    
  void ekf_mcmc(ssm_nlg model, const bool end_ram);
  
  arma::vec weight_storage;
  arma::cube mode_storage;

private:

  void trim_storage();
  arma::vec approx_loglik_storage;
  arma::vec prior_storage;
  const bool store_modes;
};


#endif
