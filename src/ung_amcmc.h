#ifndef UNG_AMCMC_H
#define UNG_AMCMC_H

#include "bssm.h"
#include "mcmc.h"

class ung_amcmc: public mcmc {
  
public:
  
  // constructor
  ung_amcmc(const unsigned int n_iter, const unsigned int n_burnin, const unsigned int n_thin, 
    const unsigned int n, const unsigned int m, const double target_acceptance, 
    const double gamma, const arma::mat& S, const unsigned int output_type = 1, 
    const bool store_modes = true);
  
  void expand();
  
  //approximate mcmc
  template<class T>
  void approx_mcmc(T model, const bool end_ram, const bool local_approx, 
    const arma::vec& initial_mode, const unsigned int max_iter, const double conv_tol);
  
  template <class T>
  void is_correction_psi(T model, const unsigned int nsim_states, 
    const unsigned int is_type, const unsigned int n_threads);
  
  template <class T>
  void is_correction_bsf(T model, const unsigned int nsim_states, 
    const unsigned int is_type, const unsigned int n_threads);
  
  template <class T>
  void is_correction_spdk(T model, const unsigned int nsim_states, 
    const unsigned int is_type, const unsigned int n_threads);

  
  template <class T>
  void approx_state_posterior(T model, const unsigned int n_threads);
 
  arma::vec weight_storage;
  arma::mat y_storage;
  arma::mat H_storage;
  
private:
  
  void trim_storage();
  arma::mat scales_storage;
  arma::vec approx_loglik_storage;
  arma::vec prior_storage;
  const bool store_modes;
};


#endif
