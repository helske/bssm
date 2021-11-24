#ifndef MCMC_H
#define MCMC_H

#include "bssm.h"

class ssm_ulg;
class ssm_mlg;
class ssm_sde;

extern Rcpp::Function default_update_fn;
extern Rcpp::Function default_prior_fn;

class mcmc {
  
protected:
  
  virtual void trim_storage();
  
  const unsigned int iter;
  const unsigned int burnin;
  const unsigned int thin;
  const unsigned int n_samples;
  const unsigned int n_par;
  const double target_acceptance;
  const double gamma;
  
public:
  
  unsigned int n_stored;
  // constructor
  mcmc(const unsigned int iter, const unsigned int burnin, 
    const unsigned int thin, const unsigned int n, const unsigned int m,
    const double target_acceptance, const double gamma, const arma::mat& S, 
    const unsigned int output_type = 1, const bool verbose = true);

  // sample states given theta
  template <class T>
  void state_posterior(T model, const unsigned int n_threads, 
    const Rcpp::Function update_fn = default_update_fn);
  
  template <class T>
  void state_summary(T model, 
    const Rcpp::Function update_fn = default_update_fn);
  
  // linear-gaussian mcmc
  template<class T>
  void mcmc_gaussian(T model, const bool end_ram, 
    const Rcpp::Function update_fn = default_update_fn, 
    const Rcpp::Function prior_fn = default_prior_fn);

  // pseudo-marginal mcmc
  template<class T>
  void pm_mcmc(T model, const unsigned int method, const unsigned int nsim, 
    const bool end_ram, const Rcpp::Function update_fn = default_update_fn, 
    const Rcpp::Function prior_fn = default_prior_fn);

  // delayed acceptance mcmc
  template<class T>
  void da_mcmc(T model, const unsigned int method, const unsigned int nsim, 
    const bool end_ram, const Rcpp::Function update_fn = default_update_fn, 
    const Rcpp::Function prior_fn = default_prior_fn);

  // pseudo-marginal mcmc for SDE
  void pm_mcmc(ssm_sde model, const unsigned int nsim, const bool end_ram);
  
  // delayed acceptance mcmc for SDE
  void da_mcmc(ssm_sde model, const unsigned int nsim, const bool end_ram);
  
  arma::vec posterior_storage;
  arma::mat theta_storage;
  arma::uvec count_storage;
  arma::cube alpha_storage;
  arma::mat alphahat;
  arma::cube Vt;
  arma::mat S;
  double acceptance_rate;
  unsigned int output_type;
  bool verbose;
};


#endif
