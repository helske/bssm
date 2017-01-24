#include "mcmc.h"
#include "ugg_bsm.h"
#include "ung_bsm.h"

// [[Rcpp::export]]
Rcpp::List gaussian_mcmc(const Rcpp::List& model_,
  arma::uvec prior_types, arma::mat prior_pars, bool sim_states,
  unsigned int n_iter, unsigned int n_burnin, unsigned int n_thin,
  double gamma, double target_acceptance, arma::mat S,
  unsigned int seed, bool end_ram, unsigned int n_threads, 
  unsigned int model_type) {
  
  arma::vec a1 = Rcpp::as<arma::vec>(model_["a1"]);
  unsigned int m = a1.n_elem;
  unsigned int n;
  
  if(model_type > 0) {
    arma::vec y = Rcpp::as<arma::vec>(model_["y"]);
    n = y.n_elem;
  } else {
    arma::vec y = Rcpp::as<arma::mat>(model_["y"]);
    n = y.n_rows;
  }
  
  mcmc mcmc_run(prior_types, prior_pars, n_iter, n_burnin, n_thin, n, m,
    target_acceptance, gamma, S, sim_states);
  
  switch (model_type) {
  case 1: {
    ugg_ssm model(clone(model_), seed);
    mcmc_run.mcmc_gaussian(model, end_ram);
    if(sim_states) mcmc_run.state_posterior(model, n_threads);
  } break;
  case 2: {
    ugg_bsm model(clone(model_), seed);
    mcmc_run.mcmc_gaussian(model, end_ram);
    if(sim_states) mcmc_run.state_posterior(model, n_threads);
  } break;
  }
  
  if(sim_states) {
    return Rcpp::List::create(Rcpp::Named("alpha") = mcmc_run.alpha_storage,
      Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
      Rcpp::Named("counts") = mcmc_run.count_storage,
      Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
      Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
  } else {
    return Rcpp::List::create(Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
      Rcpp::Named("counts") = mcmc_run.count_storage,
      Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
      Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
  }
}

// [[Rcpp::export]]
Rcpp::List nongaussian_mcmc(const Rcpp::List& model_,
  arma::uvec prior_types, arma::mat prior_pars, unsigned int nsim_states, 
  unsigned int n_iter, unsigned int n_burnin, unsigned int n_thin,
  double gamma, double target_acceptance, arma::mat S,
  unsigned int seed, bool end_ram, unsigned int n_threads, bool local_approx,
  arma::vec initial_mode, unsigned int max_iter, double conv_tol,
  bool delayed_acceptance, unsigned int simulation_method, 
  unsigned int model_type) {
  
  arma::vec a1 = Rcpp::as<arma::vec>(model_["a1"]);
  unsigned int m = a1.n_elem;
  unsigned int n;
  
  if(model_type > 0) {
    arma::vec y = Rcpp::as<arma::vec>(model_["y"]);
    n = y.n_elem;
  } else {
    arma::vec y = Rcpp::as<arma::mat>(model_["y"]);
    n = y.n_rows;
  }
  
  mcmc mcmc_run(prior_types, prior_pars, n_iter, n_burnin, n_thin, n, m,
    target_acceptance, gamma, S, 1);
  
  switch (model_type) {
  case 1: {
    ung_ssm model(clone(model_), seed);
    mcmc_run.pm_mcmc_psi(model, end_ram, nsim_states, local_approx, initial_mode, 
      max_iter, conv_tol);
    
  } break;
  case 2: {
    ung_bsm model(clone(model_), seed);
    mcmc_run.pm_mcmc_psi(model, end_ram, nsim_states, local_approx, initial_mode, 
      max_iter, conv_tol);
    
  } break;
  }
  
  return Rcpp::List::create(Rcpp::Named("alpha") = mcmc_run.alpha_storage,
    Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
    Rcpp::Named("counts") = mcmc_run.count_storage,
    Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
    Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
}
