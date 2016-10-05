#include "gssm.h"

// [[Rcpp::export]]
double gssm_loglik(const List& model_) {

  gssm model(model_, 1);

  return model.log_likelihood(true);
}

// [[Rcpp::export]]
List gssm_filter(const List& model_) {


  gssm model(model_, 1);
  arma::mat at(model.m, model.n + 1);
  arma::mat att(model.m, model.n);
  arma::cube Pt(model.m, model.m, model.n + 1);
  arma::cube Ptt(model.m, model.m, model.n);

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
arma::mat gssm_fast_smoother(const List& model_) {

  gssm model(model_, 1);
  return model.fast_smoother(true).t();
}

// [[Rcpp::export]]
arma::cube gssm_sim_smoother(const List& model_, unsigned int nsim, unsigned int seed) {

  gssm model(model_, seed);
  return model.sim_smoother(nsim, true);
}


// [[Rcpp::export]]
List gssm_smoother(const List& model_) {

  gssm model(model_, 1);
  arma::mat alphahat(model.m, model.n);
  arma::cube Vt(model.m, model.m, model.n);

  model.smoother(alphahat, Vt,true);
  arma::inplace_trans(alphahat);

  return List::create(
    Named("alphahat") = alphahat,
    Named("Vt") = Vt);
}

// [[Rcpp::export]]
List gssm_run_mcmc(const List& model_,
  arma::uvec& prior_types, arma::mat& prior_pars,  unsigned int n_iter,
  bool sim_states, unsigned int n_burnin, unsigned int n_thin,
  double gamma, double target_acceptance, arma::mat S, 
  unsigned int seed, bool end_ram, arma::uvec Z_ind,
  arma::uvec H_ind, arma::uvec T_ind, arma::uvec R_ind) {

  gssm model(clone(model_), Z_ind, H_ind, T_ind, R_ind, seed);

  unsigned int npar = prior_types.n_elem;
  unsigned int n_samples = floor((n_iter - n_burnin) / n_thin);
  arma::mat theta_store(npar, n_samples);
  arma::cube alpha_store(model.m, model.n, n_samples * sim_states);
  arma::vec posterior_store(n_samples);


  double acceptance_rate = model.run_mcmc(prior_types, prior_pars, n_iter,
    sim_states, n_burnin, n_thin, gamma, target_acceptance, S, end_ram,
    theta_store, posterior_store, alpha_store);

  arma::inplace_trans(theta_store);

  if(sim_states) {
    return List::create(Named("alpha") = alpha_store,
      Named("theta") = theta_store,
      Named("acceptance_rate") = acceptance_rate,
      Named("S") = S,  Named("posterior") = posterior_store);
  } else {
    return List::create(
      Named("theta") = theta_store,
      Named("acceptance_rate") = acceptance_rate,
      Named("S") = S,  Named("posterior") = posterior_store);

  }
}

// [[Rcpp::export]]
List gssm_run_mcmc_summary(const List& model_, arma::uvec& prior_types,
  arma::vec& prior_pars, unsigned int n_iter, unsigned int n_thin,
  unsigned int n_burnin, double gamma, double target_acceptance, arma::mat S,
  unsigned int seed, bool end_ram,
  arma::uvec Z_ind, arma::uvec H_ind, arma::uvec T_ind, arma::uvec R_ind) {

  gssm model(clone(model_), Z_ind, H_ind, T_ind, R_ind, seed);

  unsigned int npar = prior_types.n_elem;
  unsigned int n_samples = floor((n_iter - n_burnin) / n_thin);
  arma::mat theta_store(npar, n_samples);
  arma::vec posterior_store(n_samples);
  arma::mat alphahat(model.m, model.n, arma::fill::zeros);
  arma::cube Vt(model.m, model.m, model.n, arma::fill::zeros);

  double acceptance_rate = model.mcmc_summary(prior_types, prior_pars, n_iter, n_burnin, n_thin,
    gamma, target_acceptance, S,  end_ram, theta_store, posterior_store, alphahat, Vt);

  arma::inplace_trans(alphahat);
  return List::create(Named("alphahat") = alphahat,
    Named("Vt") = Vt, Named("theta") = theta_store,
    Named("acceptance_rate") = acceptance_rate,
    Named("S") = S,  Named("posterior") = posterior_store);

}


// [[Rcpp::export]]
List gssm_predict(const List& model_, arma::uvec& prior_types,
  arma::vec& prior_pars, unsigned int n_iter,
  unsigned int n_burnin, unsigned int n_thin, double gamma,
  double target_acceptance, arma::mat& S, unsigned int n_ahead,
  unsigned int interval, arma::uvec Z_ind, arma::uvec H_ind, arma::uvec T_ind,
  arma::uvec R_ind, arma::vec& probs, unsigned int seed) {

  gssm model(clone(model_), Z_ind, H_ind, T_ind, R_ind, seed);

  return model.predict(prior_types, prior_pars, n_iter, n_burnin,
    n_thin, gamma, target_acceptance, S, n_ahead, interval, probs);
}

// [[Rcpp::export]]
arma::mat gssm_predict2(const List& model_, arma::uvec& prior_types,
  arma::vec& prior_pars, unsigned int n_iter,
  unsigned int n_burnin, unsigned int n_thin, double gamma,
  double target_acceptance, arma::mat& S, unsigned int n_ahead,
  unsigned int interval, arma::uvec Z_ind, arma::uvec H_ind, arma::uvec T_ind,
  arma::uvec R_ind, unsigned int seed) {

  gssm model(clone(model_), Z_ind, H_ind, T_ind, R_ind, seed);

  return model.predict2(prior_types, prior_pars, n_iter, n_burnin,
    n_thin, gamma, target_acceptance, S, n_ahead, interval);
}


// [[Rcpp::export]]
Rcpp::List gssm_particle_filter(const List& model_, unsigned int nsim_states, unsigned int seed) {

  gssm model(model_, seed);
  //fill with zeros in case of zero weights
  arma::cube alphasim(model.m, model.n, nsim_states, arma::fill::zeros);
  arma::mat V(nsim_states, model.n, arma::fill::zeros);
  arma::umat ind(nsim_states, model.n - 1, arma::fill::zeros);
  double ll = model.particle_filter(nsim_states, alphasim, V, ind);
  return List::create(
    Named("alpha") = alphasim, Named("V") = V, Named("A") = ind,
    Named("logLik") = ll);
}

// [[Rcpp::export]]
Rcpp::List gssm_particle_smoother(const List& model_, unsigned int nsim_states, unsigned int seed,
  unsigned int method) {

  gssm model(model_, seed);

  arma::cube alphasim(model.m, model.n, nsim_states);
  arma::mat V(nsim_states, model.n);
  arma::umat ind(nsim_states, model.n - 1);
  double ll = model.particle_filter(nsim_states, alphasim, V, ind);
  if(!arma::is_finite(ll)) {
    stop("Particle filtering returned likelihood value of zero. ");
  }
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
      Named("logLik") = ll, Named("alpha") = alphasim);
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
      Named("logLik") = ll, Named("alpha") = alphasim);
  }

}

// [[Rcpp::export]]
Rcpp::List gssm_backward_simulate(const List& model_, unsigned int nsim_states, unsigned int seed,
  unsigned int nsim_store) {

  gssm model(model_, seed);

  arma::cube alphasim(model.m, model.n, nsim_states);
  arma::mat V(nsim_states, model.n);
  arma::umat ind(nsim_states, model.n - 1);
  double ll = model.particle_filter(nsim_states, alphasim, V, ind);
  if(!arma::is_finite(ll)) {
    stop("Particle filtering returned likelihood value of zero. ");
  }
  arma::cube alpha(model.m, model.n, nsim_store);
  for (unsigned int i = 0; i < nsim_store; i++) {
    alpha.slice(i) = model.backward_simulate(alphasim, V, ind);

  }
  return List::create(Named("alpha") = alpha,
    Named("logLik") = ll);
}

