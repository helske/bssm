#include "bsm.h"

// [[Rcpp::export]]
double bsm_loglik(const List& model_) {
  bsm model(model_, 1, false);
  return model.log_likelihood(true);
}

// [[Rcpp::export]]
List bsm_filter(const List& model_) {

  bsm model(model_, 1, false);

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
arma::mat bsm_fast_smoother(const List& model_) {

  bsm model(model_, 1, false);

  return model.fast_smoother(true).t();
}

// [[Rcpp::export]]
arma::cube bsm_sim_smoother(const List& model_, unsigned int nsim, unsigned int seed) {

  bsm model(model_, seed, false);
  return model.sim_smoother(nsim, true);
}


// [[Rcpp::export]]
List bsm_smoother(const List& model_) {

  bsm model(model_, 1, false);

  arma::mat alphahat(model.m, model.n);
  arma::cube Vt(model.m, model.m, model.n);

  model.smoother(alphahat, Vt, true);
  arma::inplace_trans(alphahat);

  return List::create(
    Named("alphahat") = alphahat,
    Named("Vt") = Vt);
}


// [[Rcpp::export]]
arma::mat bsm_predict2(const List& model_, arma::uvec& prior_types, arma::mat& prior_pars,
  unsigned int n_iter,
  unsigned int n_burnin, unsigned int n_thin, double gamma,
  double target_acceptance, arma::mat& S, unsigned int n_ahead,
  unsigned int interval, unsigned int seed, bool log_space) {

  bsm model(model_, seed, log_space);

  return model.predict2(prior_types, prior_pars, n_iter, n_burnin,
    n_thin, gamma, target_acceptance, S, n_ahead, interval);
}


// [[Rcpp::export]]
List bsm_predict(const List& model_,
  arma::uvec& prior_types, arma::mat& prior_pars, unsigned int n_iter,
  unsigned int n_burnin, unsigned int n_thin, double gamma,
  double target_acceptance, arma::mat& S, unsigned int n_ahead,
  unsigned int interval, arma::vec probs, unsigned int seed, bool log_space) {


  bsm model(model_, seed, log_space);

  return model.predict(prior_types, prior_pars, n_iter, n_burnin,
    n_thin, gamma, target_acceptance, S, n_ahead, interval, probs);
}

// [[Rcpp::export]]
Rcpp::List bsm_particle_filter(const List& model_, unsigned int nsim_states, unsigned int seed) {

  bsm model(model_, seed, false);
  //fill with zeros in case of zero weights
  arma::cube alphasim(model.m, model.n, nsim_states, arma::fill::zeros);
  arma::mat V(nsim_states, model.n, arma::fill::zeros);
  arma::umat ind(nsim_states, model.n - 1, arma::fill::zeros);
  double logU = model.particle_filter(nsim_states, alphasim, V, ind);
  return List::create(
    Named("alpha") = alphasim, Named("V") = V, Named("A") = ind,
    Named("logU") = logU);
}

// [[Rcpp::export]]
Rcpp::List bsm_particle_smoother(const List& model_, unsigned int nsim_states, unsigned int seed,
  unsigned int method) {

  bsm model(model_, seed, false);

  arma::cube alphasim(model.m, model.n, nsim_states);
  arma::mat V(nsim_states, model.n);
  arma::umat ind(nsim_states, model.n - 1);
  double logU = model.particle_filter(nsim_states, alphasim, V, ind);
  if(!arma::is_finite(logU)) {
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

// [[Rcpp::export]]
Rcpp::List bsm_backward_simulate(const List& model_, unsigned int nsim_states, unsigned int seed,
  unsigned int nsim_store) {

  bsm model(model_, seed, false);

  arma::cube alphasim(model.m, model.n, nsim_states);
  arma::mat V(nsim_states, model.n);
  arma::umat ind(nsim_states, model.n - 1);
  double logU = model.particle_filter(nsim_states, alphasim, V, ind);
  if(!arma::is_finite(logU)) {
    stop("Particle filtering returned likelihood value of zero. ");
  }
  arma::cube alpha(model.m, model.n, nsim_store);
  for (unsigned int i = 0; i < nsim_store; i++) {
    alpha.slice(i) = model.backward_simulate(alphasim, V, ind);

  }
  return List::create(Named("alpha") = alpha,
    Named("logU") = logU);
}

// [[Rcpp::export]]
List bsm_run_mcmc(const List& model_,
  arma::uvec& prior_types, arma::mat& prior_pars, unsigned int n_iter,
  bool sim_states, unsigned int n_burnin, unsigned int n_thin,
  double gamma, double target_acceptance, arma::mat& S,
  unsigned int seed, bool log_space, bool end_ram) {

  bsm model(model_, seed, log_space);

  unsigned int npar = prior_types.n_elem;
  unsigned int n_samples = floor((n_iter - n_burnin) / n_thin);
  arma::mat theta_store(npar, n_samples);
  arma::cube alpha_store(model.m, model.n, sim_states * n_samples);
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
List bsm_run_mcmc_summary(const List& model_, arma::uvec& prior_types,
  arma::mat& prior_pars, unsigned int n_iter, unsigned int n_burnin,
  unsigned int n_thin, double gamma, double target_acceptance, arma::mat& S,
  unsigned int seed, bool log_space, bool end_ram) {

  bsm model(model_, seed, log_space);

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


