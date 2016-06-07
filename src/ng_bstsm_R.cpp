#include "ng_bstsm.h"

// [[Rcpp::export]]
double ng_bstsm_loglik(arma::vec& y, arma::mat& Z, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, arma::vec& phi, bool slope,
  bool seasonal, bool noise, arma::uvec fixed, arma::mat& xreg, arma::vec& beta,
  unsigned int distribution, arma::vec init_signal) {

  ng_bstsm model(y, Z, T, R, a1, P1, phi, slope, seasonal, noise, fixed, xreg, beta,
    distribution,1);

  double ll = model.approx(init_signal, model.max_iter, model.conv_tol);
  return model.log_likelihood() + ll;
}

// [[Rcpp::export]]
List ng_bstsm_filter(arma::vec& y, arma::mat& Z, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, arma::vec &phi, bool slope,
  bool seasonal, bool noise, arma::uvec fixed, arma::mat& xreg, arma::vec& beta,
  unsigned int distribution, arma::vec init_signal) {

  ng_bstsm model(y, Z, T, R, a1, P1, phi, slope, seasonal, noise, fixed, xreg, beta,
    distribution,1);

  double logLik = model.approx(init_signal, model.max_iter, model.conv_tol);

  arma::mat at(a1.n_elem, y.n_elem + 1);
  arma::mat att(a1.n_elem, y.n_elem);
  arma::cube Pt(a1.n_elem, a1.n_elem, y.n_elem + 1);
  arma::cube Ptt(a1.n_elem, a1.n_elem, y.n_elem);

  logLik += model.filter(at, att, Pt, Ptt);

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
List ng_bstsm_mcmc_full(arma::vec& y, arma::mat& Z, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, arma::vec& phi,
  unsigned int distribution,
  arma::vec& theta_lwr, arma::vec& theta_upr, unsigned int n_iter,
  unsigned int nsim_states, unsigned int n_burnin, unsigned int n_thin,
  double gamma, double target_acceptance, arma::mat& S, bool slope,
  bool seasonal, bool noise, arma::uvec fixed, arma::mat& xreg, arma::vec& beta,
  arma::vec& init_signal, unsigned int method, unsigned int seed, bool log_space) {


  ng_bstsm model(y, Z, T, R, a1, P1, phi, slope, seasonal, noise, fixed, xreg, beta,
    distribution, seed, log_space);

  if (method == 1) {
    return model.mcmc_full(theta_lwr, theta_upr, n_iter,
      nsim_states, n_burnin, n_thin, gamma, target_acceptance, S, init_signal);
  } else {
    return model.mcmc_da(theta_lwr, theta_upr, n_iter,
      nsim_states, n_burnin, n_thin, gamma, target_acceptance, S, init_signal);
  }

}


// [[Rcpp::export]]
arma::mat ng_bstsm_predict2(arma::vec& y, arma::mat& Z, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, arma::vec& phi,
  unsigned int distribution, arma::vec& theta_lwr,
  arma::vec& theta_upr, unsigned int n_iter, unsigned int nsim_states,
  unsigned int n_burnin, unsigned int n_thin, double gamma,
  double target_acceptance, arma::mat& S, unsigned int n_ahead,
  unsigned int interval, bool slope, bool seasonal, bool noise, arma::uvec fixed,
  arma::mat& xreg, arma::vec& beta, arma::vec& init_signal, unsigned int seed,
  bool log_space) {

  ng_bstsm model(y, Z, T, R, a1, P1, phi, slope, seasonal, noise, fixed, xreg, beta,
    distribution, seed, log_space);

  return model.predict2(theta_lwr, theta_upr, n_iter, nsim_states, n_burnin,
    n_thin, gamma, target_acceptance, S, n_ahead, interval, init_signal);

}
