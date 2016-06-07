#include "bstsm.h"

// [[Rcpp::export]]
double bstsm_loglik(arma::vec& y, arma::mat& Z, arma::vec& H, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, bool slope, bool seasonal,
  arma::uvec fixed, arma::mat& xreg, arma::vec& beta) {

  bstsm model(y, Z, H, T, R, a1, P1, slope, seasonal, fixed, xreg, beta, 1);

  return model.log_likelihood();
}

// [[Rcpp::export]]
List bstsm_filter(arma::vec& y, arma::mat& Z, arma::vec& H, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, bool slope, bool seasonal,
  arma::uvec fixed, arma::mat& xreg, arma::vec& beta) {


  bstsm model(y, Z, H, T, R, a1, P1, slope, seasonal, fixed, xreg, beta, 1);

  arma::mat at(a1.n_elem, y.n_elem + 1);
  arma::mat att(a1.n_elem, y.n_elem);
  arma::cube Pt(a1.n_elem, a1.n_elem, y.n_elem + 1);
  arma::cube Ptt(a1.n_elem, a1.n_elem, y.n_elem);

  double logLik = model.filter(at, att, Pt, Ptt);

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
arma::mat bstsm_fast_smoother(arma::vec& y, arma::mat& Z, arma::vec& H, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, bool slope, bool seasonal,
  arma::uvec fixed, arma::mat& xreg, arma::vec& beta) {

  bstsm model(y, Z, H, T, R, a1, P1, slope, seasonal, fixed, xreg, beta, 1);
  return model.fast_smoother().t();
}

// [[Rcpp::export]]
arma::cube bstsm_sim_smoother(arma::vec& y, arma::mat& Z, arma::vec& H, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, unsigned int nsim, bool slope,
  bool seasonal,arma::uvec fixed, arma::mat& xreg, arma::vec& beta, unsigned int seed) {

  bstsm model(y, Z, H, T, R, a1, P1, slope, seasonal, fixed, xreg, beta, seed);
  return model.sim_smoother(nsim);
}


// [[Rcpp::export]]
List bstsm_smoother(arma::vec& y, arma::mat& Z, arma::vec& H, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, bool slope, bool seasonal,
  arma::uvec fixed, arma::mat& xreg, arma::vec& beta) {

  bstsm model(y, Z, H, T, R, a1, P1, slope, seasonal, fixed, xreg, beta, 1);

  arma::mat alphahat(a1.n_elem, y.n_elem);
  arma::cube Vt(a1.n_elem, a1.n_elem, y.n_elem);

  model.smoother(alphahat, Vt);
  arma::inplace_trans(alphahat);

  return List::create(
    Named("alphahat") = alphahat,
    Named("Vt") = Vt);
}

// [[Rcpp::export]]
List bstsm_mcmc_full(arma::vec& y, arma::mat& Z, arma::vec& H, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1,
  arma::vec& theta_lwr, arma::vec& theta_upr, unsigned int n_iter,
  unsigned int nsim_states, unsigned int n_burnin, unsigned int n_thin,
  double gamma, double target_acceptance, arma::mat& S, bool slope,
  bool seasonal,arma::uvec fixed, arma::mat& xreg, arma::vec& beta,
  unsigned int seed, bool log_space) {

  bstsm model(y, Z, H, T, R, a1, P1, slope, seasonal, fixed, xreg, beta, seed, log_space);

  return model.mcmc_full(theta_lwr, theta_upr, n_iter, nsim_states, n_burnin,
    n_thin, gamma, target_acceptance, S);
}

// [[Rcpp::export]]
List bstsm_mcmc_param(arma::vec& y, arma::mat& Z, arma::vec& H, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1,
  arma::vec& theta_lwr, arma::vec& theta_upr, unsigned int n_iter,
  unsigned int n_burnin, unsigned int n_thin,
  double gamma, double target_acceptance, arma::mat& S, bool slope,
  bool seasonal, arma::uvec fixed, arma::mat& xreg, arma::vec& beta,
  unsigned int seed, bool log_space) {

  bstsm model(y, Z, H, T, R, a1, P1, slope, seasonal, fixed, xreg, beta, seed, log_space);
  return model.mcmc_param(theta_lwr, theta_upr, n_iter, n_burnin,
    n_thin, gamma, target_acceptance, S);
}


// [[Rcpp::export]]
List bstsm_mcmc_summary(arma::vec& y, arma::mat& Z, arma::vec& H, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, arma::vec& theta_lwr,
  arma::vec& theta_upr, unsigned int n_iter, unsigned int n_thin,
  unsigned int n_burnin, double gamma, double target_acceptance, arma::mat& S,
  bool slope, bool seasonal,arma::uvec fixed, arma::mat& xreg, arma::vec& beta,
  unsigned int seed, bool log_space) {

  bstsm model(y, Z, H, T, R, a1, P1, slope, seasonal, fixed, xreg, beta, seed, log_space);
  return model.mcmc_summary(theta_lwr, theta_upr, n_iter, n_burnin, n_thin,
    gamma, target_acceptance, S);
}


// [[Rcpp::export]]
arma::mat bstsm_predict2(arma::vec& y, arma::mat& Z, arma::vec& H, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, arma::vec& theta_lwr,
  arma::vec& theta_upr, unsigned int n_iter, unsigned int nsim_states,
  unsigned int n_burnin, unsigned int n_thin, double gamma,
  double target_acceptance, arma::mat& S, unsigned int n_ahead,
  unsigned int interval, bool slope, bool seasonal,arma::uvec fixed,
  arma::mat& xreg, arma::vec& beta, unsigned int seed, bool log_space) {

  bstsm model(y, Z, H, T, R, a1, P1, slope, seasonal, fixed, xreg, beta, seed, log_space);

  return model.predict2(theta_lwr, theta_upr, n_iter, nsim_states, n_burnin,
    n_thin, gamma, target_acceptance, S, n_ahead, interval);
}


// [[Rcpp::export]]
List bstsm_predict(arma::vec& y, arma::mat& Z, arma::vec& H, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, arma::vec& theta_lwr,
  arma::vec& theta_upr, unsigned int n_iter,
  unsigned int n_burnin, unsigned int n_thin, double gamma,
  double target_acceptance, arma::mat& S, unsigned int n_ahead,
  unsigned int interval, bool slope, bool seasonal,arma::uvec fixed,
  arma::mat& xreg, arma::vec& beta, arma::vec probs, unsigned int seed, bool log_space) {

  bstsm model(y, Z, H, T, R, a1, P1, slope, seasonal, fixed, xreg, beta, seed, log_space);

  return model.predict(theta_lwr, theta_upr, n_iter, n_burnin,
    n_thin, gamma, target_acceptance, S, n_ahead, interval, probs);
}
