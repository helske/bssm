#ifndef MCMC_H
#define MCMC_H

#include "bssm.h"

class nlg_ssm;
class lgg_ssm;
class sde_ssm;

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
  void state_summary(T model, arma::mat& alphahat, arma::cube& Vt);
  template <class T>
  void state_sampler(T& model, const arma::mat& theta, arma::cube& alpha);
  
  // gaussian mcmc
  template<class T>
  void mcmc_gaussian(T model, const bool end_ram);
  
  // pseudo-marginal mcmc
  template<class T>
  void pm_mcmc_spdk(T model, const bool end_ram, const unsigned int nsim_states, 
    const bool local_approx, const arma::vec& initial_mode, const unsigned int max_iter, 
    const double conv_tol);
  template<class T>
  void pm_mcmc_bsf(T model, const bool end_ram, const unsigned int nsim_states);
  template<class T>
  void pm_mcmc_psi(T model, const bool end_ram, const unsigned int nsim_states, 
    const bool local_approx, const arma::vec& initial_mode, 
    const unsigned int max_iter, const double conv_tol);
  
  // delayed acceptance mcmc
  template<class T>
  void da_mcmc_bsf(T model, const bool end_ram, const unsigned int nsim_states, 
    const bool local_approx, const arma::vec& initial_mode, 
    const unsigned int max_iter, const double conv_tol);
  template<class T>
  void da_mcmc_psi(T model, const bool end_ram, const unsigned int nsim_states, 
    const bool local_approx, const arma::vec& initial_mode, 
    const unsigned int max_iter, const double conv_tol);
  template<class T>
  void da_mcmc_spdk(T model, const bool end_ram, const unsigned int nsim_states, 
    const bool local_approx, const arma::vec& initial_mode, 
    const unsigned int max_iter, const double conv_tol);
  
  // using non-linear models
  void pm_mcmc_psi_nlg(nlg_ssm model, const bool end_ram, const unsigned int nsim_states, 
    const unsigned int max_iter, const double conv_tol, const unsigned int iekf_iter);
  void pm_mcmc_bsf_nlg(nlg_ssm model, const bool end_ram, 
    const unsigned int nsim_states);
  void ekf_mcmc_nlg(nlg_ssm model, const bool end_ram, const unsigned int max_iter, 
    const double conv_tol, const unsigned int iekf_iter);
  void da_mcmc_psi_nlg(nlg_ssm model, const bool end_ram, const unsigned int nsim_states,
    const unsigned int max_iter, const double conv_tol, const unsigned int iekf_iter);
  void da_mcmc_bsf_nlg(nlg_ssm model, const bool end_ram, const unsigned int nsim_states,
    const unsigned int max_iter, const double conv_tol, const unsigned int iekf_iter);
  
  // sde models
  void pm_mcmc_bsf_sde(sde_ssm model, const bool end_ram, const unsigned int nsim_states,
    const unsigned int L);
  void da_mcmc_bsf_sde(sde_ssm model, const bool end_ram, const unsigned int nsim_states,
    const unsigned int L_c, const unsigned int L_f, const bool target_full = false);
  
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
