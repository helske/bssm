#ifndef NLG_AMCMC_H
#define NLG_AMCMC_H

#include "bssm.h"
#include "mcmc.h"

class nlg_amcmc: public mcmc {
  
public:
  
  // constructor
  nlg_amcmc(const arma::uvec& prior_distributions, const arma::mat& prior_parameters,
    const unsigned int n_iter, const unsigned int n_burnin, const unsigned int n_thin, 
    const unsigned int n, const unsigned int m, const double target_acceptance, 
    const double gamma, const arma::mat& S);
  
  void approx_mcmc(nlg_ssm model, const unsigned int max_iter, 
    const double conv_tol, const bool end_ram);
  
  void is_correction_bsf(nlg_ssm model, const unsigned int nsim_states, 
    const bool const_sim, const unsigned int n_threads);

  void state_sampler_bsf_is2(nlg_ssm model, const unsigned int nsim_states, 
    const arma::vec& approx_loglik_storage, const arma::mat& theta,
    arma::cube& alpha, arma::vec& weights);

  void state_sampler_bsf_is1(nlg_ssm model, const unsigned int nsim_states, 
    const arma::vec& approx_loglik_storage, const arma::mat& theta,
    arma::cube& alpha, arma::vec& weights, const arma::uvec& counts);
  
  
  arma::vec weight_storage;
  
private:
  
  void trim_storage();
  arma::vec approx_loglik_storage;
  arma::vec scales_storage;
  arma::vec prior_storage;
};


#endif
