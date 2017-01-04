#include "mcmc.h"
#include "bsm.h"

// [[Rcpp::export]]
Rcpp::List bsm_run_mcmc(const Rcpp::List& model_,
  arma::uvec prior_types, arma::mat prior_pars, unsigned int n_iter,
  bool sim_states, unsigned int n_burnin, unsigned int n_thin,
  double gamma, double target_acceptance, arma::mat S,
  unsigned int seed, bool log_space, bool end_ram) {
  
  bsm model(clone(model_), seed, log_space);
  
  mcmc mcmc_run(prior_types, prior_pars, n_iter, n_burnin, n_thin, model.n, model.m, 
    target_acceptance, gamma, S);
  
  double acceptance_rate = mcmc_run.mcmc_gaussian(model, end_ram);
  
  
  if(sim_states) {
  Rcpp::stop("not yet");
  }
  
  return Rcpp::List::create(Rcpp::Named("alpha") = mcmc_run.alpha_store,
    Rcpp::Named("theta") = mcmc_run.theta_store.t(),
    Rcpp::Named("acceptance_rate") = acceptance_rate,
    Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_store);
}
// 
//   bsm model(clone(model_), seed, log_space);
// 
//   unsigned int npar = prior_types.n_elem;
//   unsigned int n_samples = floor((n_iter - n_burnin) / n_thin);
//   arma::mat theta_store(npar, n_samples);
//   arma::cube alpha_store(model.n, model.m, sim_states * n_samples);
//   arma::vec posterior_store(n_samples);
// 
//   double acceptance_rate = model.run_mcmc(prior_types, prior_pars, n_iter,
//     sim_states, n_burnin, n_thin, gamma, target_acceptance, S, end_ram,
//     theta_store, posterior_store, alpha_store);
// 
//   arma::inplace_trans(theta_store);
// 
//   if(sim_states) {
//     return Rcpp::List::create(Rcpp::Named("alpha") = alpha_store,
//       Rcpp::Named("theta") = theta_store,
//       Rcpp::Named("acceptance_rate") = acceptance_rate,
//       Rcpp::Named("S") = S,  Rcpp::Named("posterior") = posterior_store);
//   } else {
//     return Rcpp::List::create(
//       Rcpp::Named("theta") = theta_store,
//       Rcpp::Named("acceptance_rate") = acceptance_rate,
//       Rcpp::Named("S") = S,  Rcpp::Named("posterior") = posterior_store);
//   }
// 
// }
// 
// 
// // [[Rcpp::export]]
// Rcpp::List bsm_run_mcmc_summary(const Rcpp::List& model_, arma::uvec& prior_types,
//   arma::mat& prior_pars, unsigned int n_iter, unsigned int n_burnin,
//   unsigned int n_thin, double gamma, double target_acceptance, arma::mat& S,
//   unsigned int seed, bool log_space, bool end_ram) {
// 
//   bsm model(clone(model_), seed, log_space);
// 
//   unsigned int npar = prior_types.n_elem;
//   unsigned int n_samples = floor((n_iter - n_burnin) / n_thin);
//   arma::mat theta_store(npar, n_samples);
//   arma::vec posterior_store(n_samples);
//   arma::mat alphahat(model.m, model.n, arma::fill::zeros);
//   arma::cube Vt(model.m, model.m, model.n, arma::fill::zeros);
// 
//   double acceptance_rate = model.mcmc_summary(prior_types, prior_pars, n_iter, n_burnin, n_thin,
//     gamma, target_acceptance, S,  end_ram, theta_store, posterior_store, alphahat, Vt);
// 
//   arma::inplace_trans(alphahat);
//   arma::inplace_trans(theta_store);
//   return Rcpp::List::create(Rcpp::Named("alphahat") = alphahat,
//     Rcpp::Named("Vt") = Vt, Rcpp::Named("theta") = theta_store,
//     Rcpp::Named("acceptance_rate") = acceptance_rate,
//     Rcpp::Named("S") = S,  Rcpp::Named("posterior") = posterior_store);
// }
// 
// 
