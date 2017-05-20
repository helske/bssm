#ifndef UNG_AMCMC_H
#define UNG_AMCMC_H

#include "bssm.h"
#include "mcmc.h"

class ung_amcmc: public mcmc {
  
public:
  
  // constructor
  ung_amcmc(const arma::uvec& prior_distributions, const arma::mat& prior_parameters,
    const unsigned int n_iter, const unsigned int n_burnin, const unsigned int n_thin, 
    const unsigned int n, const unsigned int m, const double target_acceptance, 
    const double gamma, const arma::mat& S, const bool store_states = true);
  
  //approximate mcmc
  template<class T>
  void approx_mcmc(T model, const bool end_ram, const bool local_approx, 
    const arma::vec& initial_mode, const unsigned int max_iter, const double conv_tol);
  
  
  template <class T>
  void is_correction_psi(T model, const unsigned int nsim_states, 
    const bool const_sim, const unsigned int n_threads);
    
  template <class T>
  void state_sampler_psi_is2(T model, const unsigned int nsim_states, 
    const arma::mat& theta, arma::cube& alpha, arma::vec& weights, 
    const arma::mat& y, const arma::mat& H, const arma::mat& scales);
  template <class T>
  void state_sampler_psi_is1(T model, const unsigned int nsim_states, 
    const arma::mat& theta, arma::cube& alpha, arma::vec& weights, 
    const arma::mat& y, const arma::mat& H, const arma::mat& scales,
    const arma::uvec& counts);
  
  
  template <class T>
  void is_correction_bsf(T model, const unsigned int nsim_states, 
    const bool const_sim, const unsigned int n_threads);
  template <class T>
  void state_sampler_bsf_is2(T model, const unsigned int nsim_states, 
    const arma::vec& approx_loglik_storage, const arma::mat& theta,
    arma::cube& alpha, arma::vec& weights);
  template <class T>
  void state_sampler_bsf_is1(T model, const unsigned int nsim_states, 
    const arma::vec& approx_loglik_storage, const arma::mat& theta,
    arma::cube& alpha, arma::vec& weights, const arma::uvec& counts);
  
  template <class T>
  void is_correction_spdk(T model, const unsigned int nsim_states, 
    const bool const_sim, const unsigned int n_threads);
  
  template <class T>
  void state_sampler_spdk_is2(T model, const unsigned int nsim_states,
    const arma::mat& theta, arma::cube& alpha, arma::vec& weights, 
    const arma::mat& y, const arma::mat& H, const arma::vec& scales);
  template <class T>
  void state_sampler_spdk_is1(T model, const unsigned int nsim_states,
    const arma::mat& theta, arma::cube& alpha, arma::vec& weights, 
    const arma::mat& y, const arma::mat& H, const arma::vec& scales,
    const arma::uvec& counts);
  
  arma::vec weight_storage;
  arma::mat y_storage;
  arma::mat H_storage;
  
private:
  
  void trim_storage();
  arma::mat scales_storage;
  arma::vec approx_loglik_storage;
  arma::vec prior_storage;

};


#endif
