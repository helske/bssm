#ifndef MCMC_H
#define MCMC_H

#include <RcppArmadillo.h>

class mcmc {
  
public:
  
  // constructor
  mcmc(const arma::uvec& prior_distributions, const arma::mat& prior_parameters,
    unsigned int n_iter, unsigned int n_burnin, unsigned int n_thin, unsigned int n, unsigned int m,
    double target_acceptance, double gamma, arma::mat& S, 
    unsigned int store_states = 1);
  
  void trim_storage();
  
  // compute the prior pdf
  double log_prior_pdf(const arma::vec& theta) const;
  // compute the log-ratio of proposals
  // double proposal(const arma::vec& theta, const arma::vec& theta_proposal) {
  //   return 0.0;
  // }
  // 
  // sample states given theta
  template <class T>
  void state_posterior(T model, unsigned int n_threads);
  template <class T>
  void state_sampler(T model, const arma::mat& theta, arma::cube& alpha);
  
  // gaussian mcmc
  template<class T>
  void mcmc_gaussian(T model, bool end_ram);
  
  // pseudo-marginal mcmc
  template<class T>
  void pm_mcmc_bsf(T model, bool end_ram, unsigned int nsim_states);
  template<class T>
  void pm_mcmc_spdk(T model, bool end_ram, unsigned int nsim_states);
  template<class T>
  void pm_mcmc_psi(T model, bool end_ram, unsigned int nsim_states, 
    bool local_approx, arma::vec& initial_mode, unsigned int max_iter, double conv_tol);
  
  // delayed acceptance mcmc
  template<class T>
  void da_mcmc_bsf(T model, bool end_ram, unsigned int nsim_states);
  template<class T>
  void da_mcmc_spdk(T model, bool end_ram, unsigned int nsim_states);
  template<class T>
  void da_mcmc_psi(T model, bool end_ram, unsigned int nsim_states);
  
private:
  
  const arma::uvec prior_distributions;
  const arma::mat prior_parameters;
  
  const unsigned int n_iter;
  const unsigned int n_burnin;
  const unsigned int n_thin;
  const unsigned int n_samples;
  const unsigned int n_par;
  const double target_acceptance;
  const double gamma;
  
public:
  arma::vec posterior_storage;
  arma::mat theta_storage;
  arma::cube alpha_storage;
  arma::uvec count_storage;
  unsigned int n_stored;
  arma::mat S;
  double acceptance_rate;
};


#endif
