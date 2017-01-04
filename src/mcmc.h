#ifndef MCMC_H
#define MCMC_H

#include <RcppArmadillo.h>

class gssm;
class ngssm;

class mcmc {
  
public:
  
  mcmc(const arma::uvec& prior_distributions, const arma::mat& prior_parameters,
    unsigned int n_iter, unsigned int n_burnin, unsigned int n_thin, 
    unsigned int n, unsigned int m,
    double target_acceptance, double gamma, arma::mat& S) :
  prior_distributions(prior_distributions), prior_parameters(prior_parameters),
  n_stored(1), n_iter(n_iter), n_burnin(n_burnin), n_thin(n_thin), 
  n_samples(floor((n_iter - n_burnin) / n_thin)), 
  n_par(prior_distributions.n_elem),
  target_acceptance(target_acceptance), gamma(gamma), S(S),
  alpha_store(arma::cube(m, n, n_samples)), 
  theta_store(arma::mat(n_par, n_samples)),
  posterior_store(arma::vec(n_samples)) {
    
  }
  
  void update_storage(unsigned int index, const arma::vec& current_theta, 
    double current_posterior);
  void update_storage(unsigned int index, const arma::mat& current_alpha, 
    const arma::vec& current_theta, double current_posterior);
  void update_storage(unsigned int index, const arma::mat& current_alpha);
  
  double run_mcmc(gssm& model, bool end_ram);
  double run_pm_mcmc_bsf(ngssm& model, bool end_ram, unsigned int nsim_states);
  double run_pm_mcmc_spdk(ngssm& model, bool end_ram, unsigned int nsim_states);
  double run_pm_mcmc_psi(ngssm& model, bool end_ram, unsigned int nsim_states);
  double run_da_mcmc_bsf(ngssm& model, bool end_ram, unsigned int nsim_states);
  double run_da_mcmc_spdk(ngssm& model, bool end_ram, unsigned int nsim_states);
  double run_da_mcmc_psi(ngssm& model, bool end_ram, unsigned int nsim_states);
 
  
  
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
  arma::cube alpha_store;
  arma::mat theta_store;
  arma::vec posterior_store;
  unsigned int n_stored;
  arma::mat S;
};


#endif
