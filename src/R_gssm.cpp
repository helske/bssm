#include "gssm.h"

// [[Rcpp::export]]
double gssm_loglik(arma::vec& y, arma::mat& Z, arma::vec& H, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, arma::mat& xreg, arma::vec& beta) {

  gssm model(y, Z, H, T, R, a1, P1, xreg, beta, 1);

  return model.log_likelihood(true);
}

// [[Rcpp::export]]
List gssm_filter(arma::vec& y, arma::mat& Z, arma::vec& H, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, arma::mat& xreg, arma::vec& beta) {


  gssm model(y, Z, H, T, R, a1, P1, xreg, beta, 1);
  arma::mat at(a1.n_elem, y.n_elem + 1);
  arma::mat att(a1.n_elem, y.n_elem);
  arma::cube Pt(a1.n_elem, a1.n_elem, y.n_elem + 1);
  arma::cube Ptt(a1.n_elem, a1.n_elem, y.n_elem);

  double logLik = model.filter(at, att, Pt, Ptt, true);
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
arma::mat gssm_fast_smoother(arma::vec& y, arma::mat& Z, arma::vec& H, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, arma::mat& xreg, arma::vec& beta) {

  gssm model(y, Z, H, T, R, a1, P1, xreg, beta, 1);
  return model.fast_smoother(true).t();
}

// [[Rcpp::export]]
arma::cube gssm_sim_smoother(arma::vec& y, arma::mat& Z, arma::vec& H,
  arma::cube& T, arma::cube& R, arma::vec& a1, arma::mat& P1, unsigned int nsim,
  arma::mat& xreg, arma::vec& beta, unsigned int seed) {

  gssm model(y, Z, H, T, R, a1, P1, xreg, beta, seed);
  return model.sim_smoother(nsim, true);
}


// [[Rcpp::export]]
List gssm_smoother(arma::vec& y, arma::mat& Z, arma::vec& H, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, arma::mat& xreg, arma::vec& beta) {

  gssm model(y, Z, H, T, R, a1, P1, xreg, beta,1);
  arma::mat alphahat(a1.n_elem, y.n_elem);
  arma::cube Vt(a1.n_elem, a1.n_elem, y.n_elem);

  model.smoother(alphahat, Vt,true);
  arma::inplace_trans(alphahat);

  return List::create(
    Named("alphahat") = alphahat,
    Named("Vt") = Vt);
}

// [[Rcpp::export]]
List gssm_mcmc_full(arma::vec& y, arma::mat& Z, arma::vec& H, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1,
  arma::vec& theta_lwr, arma::vec& theta_upr, unsigned int n_iter,
  unsigned int nsim_states, unsigned int n_burnin, unsigned int n_thin,
  double gamma, double target_acceptance, arma::mat S, arma::uvec Z_ind,
  arma::uvec H_ind, arma::uvec T_ind, arma::uvec R_ind, arma::mat& xreg,
  arma::vec& beta, unsigned int seed, bool end_ram) {

  gssm model(y, Z, H, T, R, a1, P1, xreg, beta, Z_ind, H_ind, T_ind, R_ind, seed);

  unsigned int npar = theta_lwr.n_elem;
  unsigned int n_samples = floor((n_iter - n_burnin) / n_thin);
  arma::mat theta_store(npar, n_samples);
  arma::cube alpha_store(model.m, model.n, nsim_states * n_samples);
  arma::vec ll_store(n_samples);

  double acceptance_rate = model.mcmc_full(theta_lwr, theta_upr, n_iter,
    nsim_states, n_burnin, n_thin, gamma, target_acceptance, S, alpha_store,
    theta_store, ll_store, end_ram);

  arma::inplace_trans(theta_store);

  return List::create(Named("alpha") = alpha_store,
    Named("theta") = theta_store,
    Named("acceptance_rate") = acceptance_rate,
    Named("S") = S,  Named("logLik") = ll_store);

}

// [[Rcpp::export]]
List gssm_mcmc_param(arma::vec& y, arma::mat& Z, arma::vec& H, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1,
  arma::vec& theta_lwr, arma::vec& theta_upr, unsigned int n_iter,
  unsigned int n_burnin, unsigned int n_thin,
  double gamma, double target_acceptance, arma::mat S, arma::uvec Z_ind,
  arma::uvec H_ind, arma::uvec T_ind, arma::uvec R_ind, arma::mat& xreg,
  arma::vec& beta, unsigned int seed, bool end_ram) {

  gssm model(y, Z, H, T, R, a1, P1, xreg, beta, Z_ind, H_ind, T_ind, R_ind, seed);

  unsigned int npar = theta_lwr.n_elem;
  unsigned int n_samples = floor((n_iter - n_burnin) / n_thin);
  arma::mat theta_store(npar, n_samples);
  arma::vec ll_store(n_samples);

  double acceptance_rate = model.mcmc_param(theta_lwr, theta_upr, n_iter,
    n_burnin, n_thin, gamma, target_acceptance, S, theta_store, ll_store, end_ram);

  arma::inplace_trans(theta_store);

  return List::create(
    Named("theta") = theta_store,
    Named("acceptance_rate") = acceptance_rate,
    Named("S") = S,  Named("logLik") = ll_store);
}

// [[Rcpp::export]]
List gssm_mcmc_summary(arma::vec& y, arma::mat& Z, arma::vec& H, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, arma::vec& theta_lwr,
  arma::vec& theta_upr, unsigned int n_iter, unsigned int n_thin,
  unsigned int n_burnin, double gamma, double target_acceptance, arma::mat S,
  arma::uvec Z_ind, arma::uvec H_ind, arma::uvec T_ind, arma::uvec R_ind,
  arma::mat& xreg, arma::vec& beta, unsigned int seed, bool end_ram) {

  gssm model(y, Z, H, T, R, a1, P1, xreg, beta, Z_ind, H_ind, T_ind, R_ind, seed);

  unsigned int npar = theta_lwr.n_elem;
  unsigned int n_samples = floor((n_iter - n_burnin) / n_thin);
  arma::mat theta_store(npar, n_samples);
  arma::vec ll_store(n_samples);
  arma::mat alphahat(model.m, model.n, arma::fill::zeros);
  arma::cube Vt(model.m, model.m, model.n, arma::fill::zeros);

  double acceptance_rate = model.mcmc_summary(theta_lwr, theta_upr, n_iter, n_burnin, n_thin,
    gamma, target_acceptance, S, alphahat, Vt, theta_store, ll_store, end_ram);

  arma::inplace_trans(alphahat);
  return List::create(Named("alphahat") = alphahat,
    Named("Vt") = Vt, Named("theta") = theta_store,
    Named("acceptance_rate") = acceptance_rate,
    Named("S") = S,  Named("logLik") = ll_store);

}


// [[Rcpp::export]]
List gssm_predict(arma::vec& y, arma::mat& Z, arma::vec& H, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, arma::vec& theta_lwr,
  arma::vec& theta_upr, unsigned int n_iter,
  unsigned int n_burnin, unsigned int n_thin, double gamma,
  double target_acceptance, arma::mat& S, unsigned int n_ahead,
  unsigned int interval, arma::uvec Z_ind, arma::uvec H_ind, arma::uvec T_ind,
  arma::uvec R_ind, arma::mat& xreg, arma::vec& beta, arma::vec& probs, unsigned int seed) {

  gssm model(y, Z, H, T, R, a1, P1, xreg, beta, Z_ind, H_ind, T_ind, R_ind, seed);

  return model.predict(theta_lwr, theta_upr, n_iter, n_burnin,
    n_thin, gamma, target_acceptance, S, n_ahead, interval, probs);
}

// [[Rcpp::export]]
arma::mat gssm_predict2(arma::vec& y, arma::mat& Z, arma::vec& H, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, arma::vec& theta_lwr,
  arma::vec& theta_upr, unsigned int n_iter, unsigned int nsim_states,
  unsigned int n_burnin, unsigned int n_thin, double gamma,
  double target_acceptance, arma::mat& S, unsigned int n_ahead,
  unsigned int interval, arma::uvec Z_ind, arma::uvec H_ind, arma::uvec T_ind,
  arma::uvec R_ind, arma::mat& xreg, arma::vec& beta, unsigned int seed) {

  gssm model(y, Z, H, T, R, a1, P1, xreg, beta, Z_ind, H_ind, T_ind, R_ind, seed);

  return model.predict2(theta_lwr, theta_upr, n_iter, nsim_states, n_burnin,
    n_thin, gamma, target_acceptance, S, n_ahead, interval);
}

