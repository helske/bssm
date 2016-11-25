#include "gssm.h"
#include "backtrack.h"

// [[Rcpp::export]]
double gssm_loglik(const Rcpp::List& model_) {

  gssm model(model_, 1);

  return model.log_likelihood(true);
}

// [[Rcpp::export]]
Rcpp::List gssm_filter(const Rcpp::List& model_) {


  gssm model(model_, 1);
  arma::mat at(model.m, model.n + 1);
  arma::mat att(model.m, model.n);
  arma::cube Pt(model.m, model.m, model.n + 1);
  arma::cube Ptt(model.m, model.m, model.n);

  double logLik = model.filter(at, att, Pt, Ptt, true);
  arma::inplace_trans(at);
  arma::inplace_trans(att);

  return Rcpp::List::create(
    Rcpp::Named("at") = at,
    Rcpp::Named("att") = att,
    Rcpp::Named("Pt") = Pt,
    Rcpp::Named("Ptt") = Ptt,
    Rcpp::Named("logLik") = logLik);
}


// [[Rcpp::export]]
arma::mat gssm_fast_smoother(const Rcpp::List& model_) {

  gssm model(model_, 1);
  return model.fast_smoother(true).t();
}

// [[Rcpp::export]]
arma::cube gssm_sim_smoother(const Rcpp::List& model_, unsigned int nsim, unsigned int seed) {

  gssm model(model_, seed);
  return model.sim_smoother(nsim, true);
}


// [[Rcpp::export]]
Rcpp::List gssm_smoother(const Rcpp::List& model_) {

  gssm model(model_, 1);
  arma::mat alphahat(model.m, model.n);
  arma::cube Vt(model.m, model.m, model.n);

  model.smoother(alphahat, Vt,true);
  arma::inplace_trans(alphahat);

  return Rcpp::List::create(
    Rcpp::Named("alphahat") = alphahat,
    Rcpp::Named("Vt") = Vt);
}

// [[Rcpp::export]]
Rcpp::List gssm_run_mcmc(const Rcpp::List& model_,
  arma::uvec& prior_types, arma::mat& prior_pars,  unsigned int n_iter,
  bool sim_states, unsigned int n_burnin, unsigned int n_thin,
  double gamma, double target_acceptance, arma::mat S, 
  unsigned int seed, bool end_ram, arma::uvec Z_ind,
  arma::uvec H_ind, arma::uvec T_ind, arma::uvec R_ind) {

  gssm model(clone(model_), Z_ind, H_ind, T_ind, R_ind, seed);

  unsigned int npar = prior_types.n_elem;
  unsigned int n_samples = floor((n_iter - n_burnin) / n_thin);
  arma::mat theta_store(npar, n_samples);
  arma::cube alpha_store(model.n, model.m, n_samples * sim_states);
  arma::vec posterior_store(n_samples);


  double acceptance_rate = model.run_mcmc(prior_types, prior_pars, n_iter,
    sim_states, n_burnin, n_thin, gamma, target_acceptance, S, end_ram,
    theta_store, posterior_store, alpha_store);

  arma::inplace_trans(theta_store);

  if(sim_states) {
    return Rcpp::List::create(Rcpp::Named("alpha") = alpha_store,
      Rcpp::Named("theta") = theta_store,
      Rcpp::Named("acceptance_rate") = acceptance_rate,
      Rcpp::Named("S") = S,  Rcpp::Named("posterior") = posterior_store);
  } else {
    return Rcpp::List::create(
      Rcpp::Named("theta") = theta_store,
      Rcpp::Named("acceptance_rate") = acceptance_rate,
      Rcpp::Named("S") = S,  Rcpp::Named("posterior") = posterior_store);

  }
}

// [[Rcpp::export]]
Rcpp::List gssm_run_mcmc_summary(const Rcpp::List& model_, arma::uvec& prior_types,
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
  arma::inplace_trans(theta_store);
  return Rcpp::List::create(Rcpp::Named("alphahat") = alphahat,
    Rcpp::Named("Vt") = Vt, Rcpp::Named("theta") = theta_store,
    Rcpp::Named("acceptance_rate") = acceptance_rate,
    Rcpp::Named("S") = S,  Rcpp::Named("posterior") = posterior_store);

}


// [[Rcpp::export]]
Rcpp::List gssm_predict(const Rcpp::List& model_, arma::uvec& prior_types,
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
arma::mat gssm_predict2(const Rcpp::List& model_, arma::uvec& prior_types,
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
Rcpp::List gssm_particle_filter(const Rcpp::List& model_, unsigned int nsim_states, unsigned int seed) {

  gssm model(model_, seed);
  //fill with zeros in case of zero weights
  arma::cube alphasim(model.m, model.n, nsim_states, arma::fill::zeros);
  arma::mat w(nsim_states, model.n, arma::fill::zeros);
  arma::umat ind(nsim_states, model.n - 1, arma::fill::zeros);
  double ll = model.particle_filter(nsim_states, alphasim, w, ind);
  return Rcpp::List::create(
    Rcpp::Named("alpha") = alphasim, Rcpp::Named("w") = w, Rcpp::Named("A") = ind,
    Rcpp::Named("logLik") = ll);
}

// [[Rcpp::export]]
Rcpp::List gssm_particle_smoother(const Rcpp::List& model_, unsigned int nsim_states, unsigned int seed,
  unsigned int method) {

  gssm model(model_, seed);

  arma::cube alphasim(model.m, model.n, nsim_states);
  arma::mat w(nsim_states, model.n);
  arma::umat ind(nsim_states, model.n - 1);
  double ll = model.particle_filter(nsim_states, alphasim, w, ind);
  if(!arma::is_finite(ll)) {
    Rcpp::stop("Particle filtering returned likelihood value of zero. ");
  }
  if(method == 1) {
    backtrack_pf(alphasim, ind);

    arma::mat alphahat(model.n, model.m);

    arma::vec wnorm = w.col(model.n - 1)/arma::sum(w.col(model.n - 1));
    for(unsigned int t = 0; t < model.n; t ++){
      for(unsigned k = 0; k < model.m; k++) {
        alphahat(t, k) = arma::dot(arma::vectorise(alphasim.tube(k, t)), wnorm);
      }
    }
    return Rcpp::List::create(
      Rcpp::Named("alphahat") = alphahat, Rcpp::Named("w") = w,
      Rcpp::Named("logLik") = ll, Rcpp::Named("alpha") = alphasim);
  } else {
    model.backtrack_pf2(alphasim, w, ind);

    arma::mat alphahat(model.n, model.m);
    for(unsigned int t = 0; t < model.n; t ++){
      arma::vec wnorm = w.col(t)/arma::sum(w.col(t));
      for(unsigned k = 0; k < model.m; k++) {
        alphahat(t, k) = arma::dot(arma::vectorise(alphasim.tube(k, t)), wnorm);
      }
    }
    return Rcpp::List::create(Rcpp::Named("alphahat") = alphahat, Rcpp::Named("w") = w,
      Rcpp::Named("logLik") = ll, Rcpp::Named("alpha") = alphasim);
  }

}

// [[Rcpp::export]]
Rcpp::List gssm_backward_simulate(const Rcpp::List& model_, unsigned int nsim_states, unsigned int seed,
  unsigned int nsim_store) {

  gssm model(model_, seed);

  arma::cube alphasim(model.m, model.n, nsim_states);
  arma::mat w(nsim_states, model.n);
  arma::umat ind(nsim_states, model.n - 1);
  double ll = model.particle_filter(nsim_states, alphasim, w, ind);
  if(!arma::is_finite(ll)) {
    Rcpp::stop("Particle filtering returned likelihood value of zero. ");
  }
  arma::cube alpha(model.m, model.n, nsim_store);
  for (unsigned int i = 0; i < nsim_store; i++) {
    alpha.slice(i) = model.backward_simulate(alphasim, w, ind);

  }
  return Rcpp::List::create(Rcpp::Named("alpha") = alpha,
    Rcpp::Named("logLik") = ll);
}

