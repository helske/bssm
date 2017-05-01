#ifndef MCMC_H
#define MCMC_H

#include "bssm.h"

class nlg_ssm;

class mcmc {
  
protected:
  
  virtual void trim_storage();
  
  const arma::uvec prior_distributions;
  const arma::mat prior_parameters;
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
  mcmc(const arma::uvec& prior_distributions, const arma::mat& prior_parameters,
    const unsigned int n_iter, const unsigned int n_burnin, 
    const unsigned int n_thin, const unsigned int n, const unsigned int m,
    const double target_acceptance, const double gamma, const arma::mat& S, 
    const bool store_states = true);
  
  // compute the prior pdf
  virtual double log_prior_pdf(const arma::vec& theta) const;
  
  // compute the log-ratio of proposals
  // double proposal(const arma::vec& theta, const arma::vec& theta_proposal) const;

  // sample states given theta
  template <class T>
  void state_posterior(T model, const unsigned int n_threads);
  template <class T>
  void state_summary(T model, arma::mat& alphahat, arma::cube& Vt);
  template <class T>
  void state_sampler(T model, const arma::mat& theta, arma::cube& alpha);
  
  // gaussian mcmc
  template<class T>
  void mcmc_gaussian(T model, const bool end_ram);
  
  // pseudo-marginal mcmc
  template<class T>
  void pm_mcmc_bsf(T model, const bool end_ram, const unsigned int nsim_states);
  template<class T>
  void pm_mcmc_spdk(T model, const bool end_ram, const unsigned int nsim_states);
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
  void da_mcmc_spdk(T model, const bool end_ram, const unsigned int nsim_states);
  template<class T>
  void da_mcmc_psi(T model, const bool end_ram, const unsigned int nsim_states, 
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
  
  arma::vec posterior_storage;
  arma::mat theta_storage;
  arma::cube alpha_storage;
  arma::uvec count_storage;
  arma::mat S;
  double acceptance_rate;
  

  

};


#endif
