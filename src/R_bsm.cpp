#include "bsm.h"

// [[Rcpp::export]]
double bsm_loglik(arma::vec& y, arma::mat& Z, arma::vec& H, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, bool slope, bool seasonal,
  arma::uvec fixed, arma::mat& xreg, arma::vec& beta) {
  
  bsm model(y, Z, H, T, R, a1, P1, slope, seasonal, fixed, xreg, beta, 1);
  
  return model.log_likelihood(true);
}

// [[Rcpp::export]]
List bsm_filter(arma::vec& y, arma::mat& Z, arma::vec& H, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, bool slope, bool seasonal,
  arma::uvec fixed, arma::mat& xreg, arma::vec& beta) {
  
  
  bsm model(y, Z, H, T, R, a1, P1, slope, seasonal, fixed, xreg, beta, 1);
  
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
arma::mat bsm_fast_smoother(arma::vec& y, arma::mat& Z, arma::vec& H, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, bool slope, bool seasonal,
  arma::uvec fixed, arma::mat& xreg, arma::vec& beta) {
  
  bsm model(y, Z, H, T, R, a1, P1, slope, seasonal, fixed, xreg, beta, 1);
  return model.fast_smoother(true).t();
}

// [[Rcpp::export]]
arma::cube bsm_sim_smoother(arma::vec& y, arma::mat& Z, arma::vec& H, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, unsigned int nsim, bool slope,
  bool seasonal,arma::uvec fixed, arma::mat& xreg, arma::vec& beta, unsigned int seed) {
  
  bsm model(y, Z, H, T, R, a1, P1, slope, seasonal, fixed, xreg, beta, seed);
  return model.sim_smoother(nsim, true);
}


// [[Rcpp::export]]
List bsm_smoother(arma::vec& y, arma::mat& Z, arma::vec& H, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, bool slope, bool seasonal,
  arma::uvec fixed, arma::mat& xreg, arma::vec& beta) {
  
  bsm model(y, Z, H, T, R, a1, P1, slope, seasonal, fixed, xreg, beta, 1);
  
  arma::mat alphahat(a1.n_elem, y.n_elem);
  arma::cube Vt(a1.n_elem, a1.n_elem, y.n_elem);
  
  model.smoother(alphahat, Vt, true);
  arma::inplace_trans(alphahat);
  
  return List::create(
    Named("alphahat") = alphahat,
    Named("Vt") = Vt);
}

// [[Rcpp::export]]
List bsm_mcmc_full(arma::vec& y, arma::mat& Z, arma::vec& H, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1,
  arma::vec& theta_lwr, arma::vec& theta_upr, unsigned int n_iter,
  unsigned int nsim_states, unsigned int n_burnin, unsigned int n_thin,
  double gamma, double target_acceptance, arma::mat& S, bool slope,
  bool seasonal,arma::uvec fixed, arma::mat& xreg, arma::vec& beta,
  unsigned int seed, bool log_space, bool end_ram) {
  
  bsm model(y, Z, H, T, R, a1, P1, slope, seasonal, fixed, xreg, beta, seed, log_space);
  
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
List bsm_mcmc_param(arma::vec& y, arma::mat& Z, arma::vec& H, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1,
  arma::vec& theta_lwr, arma::vec& theta_upr, unsigned int n_iter,
  unsigned int n_burnin, unsigned int n_thin,
  double gamma, double target_acceptance, arma::mat& S, bool slope,
  bool seasonal, arma::uvec fixed, arma::mat& xreg, arma::vec& beta,
  unsigned int seed, bool log_space, bool sample_states, bool end_ram) {
  
  bsm model(y, Z, H, T, R, a1, P1, slope, seasonal, fixed, xreg, beta, seed, log_space);
  
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
List bsm_mcmc_parallel_full(arma::vec& y, arma::mat& Z, arma::vec& H, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1,
  arma::vec& theta_lwr, arma::vec& theta_upr, unsigned int n_iter,
  unsigned int n_burnin, unsigned int n_thin,
  double gamma, double target_acceptance, arma::mat& S, bool slope,
  bool seasonal, arma::uvec fixed, arma::mat& xreg, arma::vec& beta,
  unsigned int seed, bool log_space, unsigned int nsim_states,
  unsigned int n_threads, arma::uvec seeds, bool end_ram) {
  
  bsm model(y, Z, H, T, R, a1, P1, slope, seasonal, fixed, xreg, beta, seed, log_space);
  
  n_thin = 1; // no thinning allowed, make check in R
  unsigned int npar = theta_lwr.n_elem;
  unsigned int n_samples = floor((n_iter - n_burnin));
  arma::mat theta_store(npar, n_samples);
  arma::vec ll_store(n_samples);
  arma::uvec counts(n_samples, arma::fill::zeros);
  
  double acceptance_rate = model.mcmc_param2(theta_lwr, theta_upr, n_iter,
    n_burnin, n_thin, gamma, target_acceptance, S, theta_store, ll_store, counts, end_ram);
  
  arma::cube alpha = sample_states(model, theta_store, counts, nsim_states, n_threads, seeds);
  
  arma::inplace_trans(theta_store);
  
  return List::create(Named("alpha") = alpha,
    Named("theta") = theta_store, Named("counts") = counts,
    Named("acceptance_rate") = acceptance_rate,
    Named("S") = S,  Named("logLik") = ll_store);
}

// [[Rcpp::export]]
List bsm_mcmc_summary(arma::vec& y, arma::mat& Z, arma::vec& H, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, arma::vec& theta_lwr,
  arma::vec& theta_upr, unsigned int n_iter, unsigned int n_burnin,
  unsigned int n_thin, double gamma, double target_acceptance, arma::mat& S,
  bool slope, bool seasonal,arma::uvec fixed, arma::mat& xreg, arma::vec& beta,
  unsigned int seed, bool log_space, bool end_ram) {
  
  bsm model(y, Z, H, T, R, a1, P1, slope, seasonal, fixed, xreg, beta, seed, log_space);
  
  unsigned int npar = theta_lwr.n_elem;
  unsigned int n_samples = floor((n_iter - n_burnin) / n_thin);
  arma::mat theta_store(npar, n_samples);
  arma::vec ll_store(n_samples);
  arma::mat alphahat(model.m, model.n, arma::fill::zeros);
  arma::cube Vt(model.m, model.m, model.n, arma::fill::zeros);
  
  double acceptance_rate = model.mcmc_summary(theta_lwr, theta_upr, n_iter, n_burnin, n_thin,
    gamma, target_acceptance, S, alphahat, Vt, theta_store, ll_store, end_ram);
  
  arma::inplace_trans(alphahat);
  arma::inplace_trans(theta_store);
  return List::create(Named("alphahat") = alphahat,
    Named("Vt") = Vt, Named("theta") = theta_store,
    Named("acceptance_rate") = acceptance_rate,
    Named("S") = S,  Named("logLik") = ll_store);
}


// [[Rcpp::export]]
arma::mat bsm_predict2(arma::vec& y, arma::mat& Z, arma::vec& H, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, arma::vec& theta_lwr,
  arma::vec& theta_upr, unsigned int n_iter, unsigned int nsim_states,
  unsigned int n_burnin, unsigned int n_thin, double gamma,
  double target_acceptance, arma::mat& S, unsigned int n_ahead,
  unsigned int interval, bool slope, bool seasonal,arma::uvec fixed,
  arma::mat& xreg, arma::vec& beta, unsigned int seed, bool log_space) {
  
  bsm model(y, Z, H, T, R, a1, P1, slope, seasonal, fixed, xreg, beta, seed, log_space);
  
  return model.predict2(theta_lwr, theta_upr, n_iter, nsim_states, n_burnin,
    n_thin, gamma, target_acceptance, S, n_ahead, interval);
}


// [[Rcpp::export]]
List bsm_predict(arma::vec& y, arma::mat& Z, arma::vec& H, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, arma::vec& theta_lwr,
  arma::vec& theta_upr, unsigned int n_iter,
  unsigned int n_burnin, unsigned int n_thin, double gamma,
  double target_acceptance, arma::mat& S, unsigned int n_ahead,
  unsigned int interval, bool slope, bool seasonal,arma::uvec fixed,
  arma::mat& xreg, arma::vec& beta, arma::vec probs, unsigned int seed, bool log_space) {
  
  bsm model(y, Z, H, T, R, a1, P1, slope, seasonal, fixed, xreg, beta, seed, log_space);
  
  return model.predict(theta_lwr, theta_upr, n_iter, n_burnin,
    n_thin, gamma, target_acceptance, S, n_ahead, interval, probs);
}

// [[Rcpp::export]]
arma::cube bsm_sample_states(arma::vec& y, arma::mat& Z, arma::vec& H, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1,
  arma::mat& theta, unsigned int nsim_states, bool slope,
  bool seasonal,arma::uvec fixed, arma::mat& xreg, arma::vec& beta,
  unsigned int n_threads, arma::uvec seeds) {
  
  arma::uvec counts(theta.n_cols, arma::fill::ones);
  
  bsm model(y, Z, H, T, R, a1, P1, slope, seasonal, fixed, xreg, beta, 1);
  
  return sample_states(model, theta, counts, nsim_states, n_threads, seeds);
}

// [[Rcpp::export]]
Rcpp::List bsm_particle_filter(arma::vec& y, arma::mat& Z, arma::vec& H, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, unsigned int nsim_states, bool slope,
  bool seasonal,arma::uvec fixed, arma::mat& xreg, arma::vec& beta, unsigned int seed) {
  
  bsm model(y, Z, H, T, R, a1, P1, slope, seasonal, fixed, xreg, beta, seed);
  
  arma::cube alphasim(model.m, model.n, nsim_states);
  arma::mat V(nsim_states, model.n);
  arma::umat ind(nsim_states, model.n - 1);
  double logU = model.particle_filter(nsim_states, alphasim, V, ind);
  
  return List::create(
    Named("alpha") = alphasim, Named("V") = V, Named("A") = ind,
    Named("logU") = logU);
}

// [[Rcpp::export]]
Rcpp::List bsm_particle_smoother(arma::vec& y, arma::mat& Z, arma::vec& H, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, unsigned int nsim_states, bool slope,
  bool seasonal,arma::uvec fixed, arma::mat& xreg, arma::vec& beta, unsigned int seed, 
  unsigned int method) {
  
  bsm model(y, Z, H, T, R, a1, P1, slope, seasonal, fixed, xreg, beta, seed);
  
  arma::cube alphasim(model.m, model.n, nsim_states);
  arma::mat V(nsim_states, model.n);
  arma::umat ind(nsim_states, model.n - 1);
  double logU = model.particle_filter(nsim_states, alphasim, V, ind);
  
  if(method == 1) {
    backtrack_pf(alphasim, ind);
    arma::mat alphahat(model.n, model.m);
    
    arma::vec Vnorm = V.col(model.n - 1)/arma::sum(V.col(model.n - 1));
    for(unsigned int t = 0; t < model.n; t ++){
      for(unsigned k = 0; k < model.m; k++) {
        alphahat(t, k) = arma::dot(arma::vectorise(alphasim.tube(k, t)), Vnorm);
      }
    }
    return List::create(
      Named("alphahat") = alphahat, Named("V") = Vnorm,
      Named("logU") = logU, Named("alpha") = alphasim);
  } else {
    model.backtrack_pf2(alphasim, V, ind);
    
    arma::mat alphahat(model.n, model.m);
    for(unsigned int t = 0; t < model.n; t ++){
      arma::vec Vnorm = V.col(t)/arma::sum(V.col(t));
      for(unsigned k = 0; k < model.m; k++) {
        alphahat(t, k) = arma::dot(arma::vectorise(alphasim.tube(k, t)), Vnorm);
      }
    }
    return List::create(Named("alphahat") = alphahat, Named("V") = V,
      Named("logU") = logU, Named("alpha") = alphasim);
  }
 
}
