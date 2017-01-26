#ifndef UNG_AMCMC_H
#define UNG_AMCMC_H

#include <RcppArmadillo.h>
#include "mcmc.h"

class ung_amcmc: public mcmc {
  
public:
  
  // constructor
  ung_amcmc(const arma::uvec& prior_distributions, const arma::mat& prior_parameters,
    unsigned int n_iter, unsigned int n_burnin, unsigned int n_thin, unsigned int n, unsigned int m,
    double target_acceptance, double gamma, arma::mat& S, bool store_states = true);
  
  //approximate mcmc
  template<class T>
  void approx_mcmc(T model, bool end_ram, bool local_approx, 
    arma::vec& initial_mode, unsigned int max_iter, double conv_tol);
  
  
  template <class T>
  void is_correction_psi(T model, unsigned int nsim_states, bool const_sim, unsigned int n_threads);
    
  template <class T>
  void state_sampler_psi(T model, unsigned int nsim_states, const arma::mat& theta,
    arma::cube& alpha, arma::vec& weights, const arma::mat& y, const arma::mat& H,
    const arma::mat& scales);
  
  template <class T>
  void is_correction_bsf(T model, unsigned int nsim_states, 
    bool const_sim, unsigned int n_threads);
  template <class T>
  void state_sampler_bsf(T model, unsigned int nsim_states, 
    const arma::vec& approx_loglik_storage, const arma::mat& theta,
    arma::cube& alpha, arma::vec& weights);
  
  template <class T>
  void is_correction_spdk(T model, unsigned int nsim_states, bool const_sim, unsigned int n_threads);
  
  template <class T>
  void state_sampler_spdk(T model, unsigned int nsim_states, const arma::mat& theta,
    arma::cube& alpha, arma::vec& weights, const arma::mat& y, const arma::mat& H,
    const arma::vec& scales);
  
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
