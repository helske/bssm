#include "ngssm.h"

// [[Rcpp::export]]
double ngssm_loglik(arma::vec& y, arma::mat& Z, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, arma::vec phi,
  arma::mat& xreg, arma::vec& beta, unsigned int distribution,
  arma::vec init_signal, unsigned int nsim_states,
  unsigned int seed) {

  ngssm model(y, Z, T, R, a1, P1, phi, xreg, beta, distribution, seed);

  if (nsim_states < 2) {
    model.conv_tol = 1.0e-12;
    model.max_iter = 1000;
  }

  double ll = model.approx(init_signal, model.max_iter, model.conv_tol);
  double ll_w = 0;
  if (nsim_states > 1) {
    arma::cube alpha = model.sim_smoother(nsim_states, true);
    arma::vec weights = exp(model.importance_weights(alpha) - model.scaling_factor(init_signal));
    ll_w = log(sum(weights) / nsim_states);
  }
  return model.log_likelihood(true) + ll + ll_w;
}

// [[Rcpp::export]]
List ngssm_filter(arma::vec& y, arma::mat& Z, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, arma::vec phi,
  arma::mat& xreg, arma::vec& beta, unsigned int distribution,
  arma::vec init_signal) {

  ngssm model(y, Z, T, R, a1, P1, phi, xreg, beta, distribution,1);

  double logLik = model.approx(init_signal, 1000, 1e-12);

  arma::mat at(a1.n_elem, y.n_elem + 1);
  arma::mat att(a1.n_elem, y.n_elem);
  arma::cube Pt(a1.n_elem, a1.n_elem, y.n_elem + 1);
  arma::cube Ptt(a1.n_elem, a1.n_elem, y.n_elem);

  logLik += model.filter(at, att, Pt, Ptt, true);

  arma::inplace_trans(at);
  arma::inplace_trans(att);

  return List::create(
    Named("at") = at,
    Named("att") = att,
    Named("Pt") = Pt,
    Named("Ptt") = Ptt,
    Named("logLik") = logLik);
}

// [[Rcpp::export]]
List ngssm_mcmc_full(arma::vec& y, arma::mat& Z, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, arma::vec phi,
  unsigned int distribution,
  arma::vec& theta_lwr, arma::vec& theta_upr, unsigned int n_iter,
  unsigned int nsim_states, unsigned int n_burnin,
  unsigned int n_thin, double gamma, double target_acceptance, arma::mat& S,
  arma::uvec Z_ind, arma::uvec T_ind, arma::uvec R_ind, arma::mat& xreg,
  arma::vec& beta, arma::vec& init_signal, unsigned int seed) {

  ngssm model(y, Z, T, R, a1, P1, phi, xreg, beta, distribution, Z_ind,
    T_ind, R_ind, seed);

  return model.mcmc_da(theta_lwr, theta_upr, n_iter, nsim_states, n_burnin,
    n_thin, gamma, target_acceptance, S, init_signal);
}


// [[Rcpp::export]]
arma::mat ngssm_predict2(arma::vec& y, arma::mat& Z, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, arma::vec phi,
  unsigned int distribution, arma::vec& theta_lwr,
  arma::vec& theta_upr, unsigned int n_iter, unsigned int nsim_states,
  unsigned int n_burnin, unsigned int n_thin, double gamma,
  double target_acceptance, arma::mat& S, unsigned int n_ahead,
  unsigned int interval, arma::uvec Z_ind, arma::uvec T_ind,
  arma::uvec R_ind, arma::mat& xreg, arma::vec& beta, arma::vec& init_signal,
  unsigned int seed) {


  ngssm model(y, Z, T, R, a1, P1, phi, xreg, beta, distribution, Z_ind,
    T_ind, R_ind, seed);

  return model.predict2(theta_lwr, theta_upr, n_iter, nsim_states, n_burnin,
    n_thin, gamma, target_acceptance, S, n_ahead, interval, init_signal);
}


// [[Rcpp::export]]
List ngssm_importance_sample(arma::vec& y, arma::mat& Z, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, arma::vec phi,
  unsigned int distribution, arma::mat& xreg, arma::vec& beta, arma::vec init_signal,
  unsigned int nsim_states,  unsigned int seed) {
  
  ngssm model(y, Z, T, R, a1, P1, phi, xreg, beta, distribution, 1);
  
  
  double ll = model.approx(init_signal, model.max_iter, model.conv_tol);
  
  arma::cube alpha = model.sim_smoother(nsim_states, true);
  arma::vec weights = exp(model.importance_weights(alpha) -
    model.scaling_factor(init_signal));
  
  return List::create(
    Named("alpha") = alpha,
    Named("weights") = weights);
}

// [[Rcpp::export]]
List ngssm_approx_model(arma::vec& y, arma::mat& Z, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, arma::vec phi,
  unsigned int distribution, arma::mat& xreg, arma::vec& beta, arma::vec init_signal,
  unsigned int max_iter, double conv_tol) {
  
  ngssm model(y, Z, T, R, a1, P1, phi, xreg, beta, distribution, 1);
  
  double ll = model.approx(init_signal, max_iter, conv_tol);
  
  return List::create(
    Named("y") = model.y,
    Named("H") = model.H,
    Named("logLik") = ll,
    Named("signal") = init_signal);
}

