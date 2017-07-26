#ifndef SDE_AMCMC_H
#define SDE_AMCMC_H

#include "bssm.h"
#include "mcmc.h"

class sde_amcmc: public mcmc {
  
public:
  
  // constructor
  sde_amcmc(const unsigned int n_iter, const unsigned int n_burnin,
    const unsigned int n_thin, const unsigned int n,
    const double target_acceptance, const double gamma, const arma::mat& S);
  
  void approx_mcmc(sde_ssm model, const bool end_ram, 
    const unsigned int nsim_states, const unsigned int L_c);
  
  void is_correction_bsf(sde_ssm model, const unsigned int nsim_states, 
    const unsigned int L_c, const unsigned int L_f, const bool coupled,
    const bool const_sim, const unsigned int n_threads);
  
  void state_sampler_bsf_is2(sde_ssm& model, const unsigned int nsim_states,
    const unsigned int L_f, const arma::vec& approx_loglik_storage,
    const arma::mat& theta, arma::cube& alpha, arma::vec& weights);
  
  void state_sampler_bsf_is1(sde_ssm& model, const unsigned int nsim_states,
    const unsigned int L_f, const arma::vec& approx_loglik_storage,
    const arma::mat& theta, arma::cube& alpha, arma::vec& weights,
    const arma::uvec& counts);
  
  void state_sampler_cbsf_is2(sde_ssm& model, const unsigned int nsim_states,
    const unsigned int L_c, const unsigned int L_f, const arma::vec& approx_loglik_storage,
    const arma::mat& theta, arma::cube& alpha, arma::vec& weights, const arma::uvec& iter);
  
  void state_sampler_cbsf_is1(sde_ssm& model, const unsigned int nsim_states,
    const unsigned int L_c, const unsigned int L_f, const arma::vec& approx_loglik_storage,
    const arma::mat& theta, arma::cube& alpha, arma::vec& weights, const arma::uvec& counts, 
    const arma::uvec& iter);
  
  arma::vec weight_storage;
  
private:
  
  void trim_storage();
  arma::vec approx_loglik_storage;
  arma::vec prior_storage;
  arma::uvec iter_storage;
};


#endif
