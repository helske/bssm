#include "ngssm.h"
#include "is_correction.h"
#include "backtrack.h"

// [[Rcpp::export]]
double ngssm_loglik(const Rcpp::List& model_, arma::vec init_signal,
  unsigned int nsim_states, unsigned int method, unsigned int seed,
   unsigned int max_iter, double conv_tol) {
  
  ngssm model(model_, seed);
  
  model.conv_tol = conv_tol;
  model.max_iter = max_iter;
  
  
  double ll = 0.0;
  if (method != 3 ){
    ll = model.approx(init_signal, model.max_iter, model.conv_tol);
    ll += model.log_likelihood(true);
  }
  
  if (!std::isfinite(ll)) {
    return -arma::datum::inf;
  } else {
    if (nsim_states > 0) {
      switch(method) {
      case 1:
        ll = model.psi_loglik(nsim_states, ll, model.scaling_factor_vec(init_signal));
        break;
      case 2  : {
          arma::vec weights(nsim_states);
          arma::cube alpha = model.sim_smoother(nsim_states, true);
          weights = model.importance_weights(alpha) - model.scaling_factor(init_signal);
          double maxw = weights.max();
          weights = exp(weights - maxw);
          ll += log(arma::mean(weights)) + maxw;
        }
        break;
      case 3:
        ll = model.bsf_loglik(nsim_states);
        break;
      }
    }
  }
  return ll;
}



// [[Rcpp::export]]
Rcpp::List ngssm_filter(const Rcpp::List& model_, arma::vec init_signal) {
  
  ngssm model(model_, 1);
  
  double logLik = model.approx(init_signal, 1000, 1e-12);
  
  arma::mat at(model.m, model.n + 1);
  arma::mat att(model.m, model.n);
  arma::cube Pt(model.m, model.m, model.n + 1);
  arma::cube Ptt(model.m, model.m, model.n);
  
  logLik += model.filter(at, att, Pt, Ptt, true);
  
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
arma::mat ngssm_fast_smoother(const Rcpp::List& model_, arma::vec init_signal) {
  
  ngssm model(model_, 1);
  model.approx(init_signal, 1000, 1e-12);
  
  return model.fast_smoother(true).t();
}

// [[Rcpp::export]]
arma::cube ngssm_sim_smoother(const Rcpp::List& model_, unsigned nsim,
  arma::vec init_signal, unsigned int seed) {
  
  ngssm model(model_, seed);
  model.approx(init_signal, 1000, 1e-12);
  
  return model.sim_smoother(nsim, true);
}

// [[Rcpp::export]]
Rcpp::List ngssm_smoother(const Rcpp::List& model_, arma::vec init_signal) {
  
  ngssm model(model_, 1);
  
  model.approx(init_signal, 1000, 1e-12);
  
  arma::mat alphahat(model.m, model.n);
  arma::cube Vt(model.m, model.m, model.n);
  
  model.smoother(alphahat, Vt, true);
  arma::inplace_trans(alphahat);
  
  return Rcpp::List::create(
    Rcpp::Named("alphahat") = alphahat,
    Rcpp::Named("Vt") = Vt);
}
// [[Rcpp::export]]
Rcpp::List ngssm_run_mcmc(const Rcpp::List& model_,
  arma::uvec& prior_types, arma::mat& prior_pars, unsigned int n_iter,
  unsigned int nsim_states, unsigned int n_burnin, unsigned int n_thin,
  double gamma, double target_acceptance, arma::mat S,
  arma::vec& init_signal, unsigned int seed, bool end_ram,
  bool adapt_approx, bool da, unsigned int sim_type,
  arma::uvec Z_ind, arma::uvec T_ind, arma::uvec R_ind) {
  
  ngssm model(clone(model_), Z_ind, T_ind, R_ind, seed);
  
  unsigned int npar = prior_types.n_elem;
  unsigned int n_samples = floor((n_iter - n_burnin) / n_thin);
  arma::mat theta_store(npar, n_samples);
  arma::cube alpha_store(model.n, model.m, n_samples);
  arma::vec posterior_store(n_samples);
  
  double acceptance_rate;
  if(sim_type > 1){
    acceptance_rate = model.run_mcmc_pf(prior_types, prior_pars, n_iter, nsim_states, n_burnin,
      n_thin, gamma, target_acceptance, S, init_signal, end_ram, adapt_approx, da,
      theta_store, posterior_store, alpha_store, sim_type == 2);
  } else {
    acceptance_rate = model.run_mcmc(prior_types, prior_pars, n_iter, nsim_states, n_burnin,
      n_thin, gamma, target_acceptance, S, init_signal, end_ram, adapt_approx, da,
      theta_store, posterior_store, alpha_store);
  }
  
  arma::inplace_trans(theta_store);
  return Rcpp::List::create(Rcpp::Named("alpha") = alpha_store,
    Rcpp::Named("theta") = theta_store,
    Rcpp::Named("acceptance_rate") = acceptance_rate,
    Rcpp::Named("S") = S,  Rcpp::Named("posterior") = posterior_store);
}


// [[Rcpp::export]]
Rcpp::List ngssm_run_mcmc_is(const Rcpp::List& model_,
  arma::uvec& prior_types, arma::mat& prior_pars, unsigned int n_iter,
  unsigned int nsim_states, unsigned int n_burnin, unsigned int n_thin,
  double gamma, double target_acceptance, arma::mat S,
  arma::vec& init_signal, unsigned int seed,
  unsigned int n_threads, bool end_ram, bool adapt_approx, 
  unsigned int sim_type, bool const_m,
  arma::uvec Z_ind, arma::uvec T_ind, arma::uvec R_ind, const arma::uvec& seeds) {
  
  ngssm model(clone(model_), Z_ind, T_ind, R_ind, seed);
  
  unsigned int npar = prior_types.n_elem;
  unsigned int n_samples = floor((n_iter - n_burnin) / n_thin);
  
  arma::mat y_store(model.n, n_samples);
  arma::mat H_store(model.n, n_samples);
  arma::mat ll_approx_u_store(model.n, n_samples);
  arma::mat theta_store(npar, n_samples);
  arma::vec ll_store(n_samples);
  arma::vec prior_store(n_samples);
  
  arma::uvec counts(n_samples);
  //no thinning allowed!
  double acceptance_rate = model.mcmc_approx(prior_types, prior_pars, n_iter,
    nsim_states, n_burnin, 1, gamma, target_acceptance, S, init_signal,
    theta_store, ll_store, prior_store, y_store, H_store, ll_approx_u_store, counts,
    end_ram, adapt_approx);
  
  arma::vec weights_store(counts.n_elem);
  arma::cube alpha_store(model.n, model.m, counts.n_elem);
  
  if(sim_type == 1) {
    is_correction(model, theta_store, y_store, H_store, arma::sum(ll_approx_u_store, 0).t(),
      counts, nsim_states, n_threads, weights_store, alpha_store, const_m, seeds);
  } else {
    if (sim_type == 2) {
      is_correction_bsf(model, theta_store, ll_store,
        counts, nsim_states, n_threads, weights_store, alpha_store, const_m, seeds);
    } else {
      is_correction_psif(model, theta_store, y_store, H_store, ll_approx_u_store,
        counts, nsim_states, n_threads, weights_store, alpha_store, const_m, seeds);
    }
  }
  prior_store += ll_store + log(weights_store);
  
  arma::inplace_trans(theta_store);
  return Rcpp::List::create(Rcpp::Named("alpha") = alpha_store,
    Rcpp::Named("theta") = theta_store, Rcpp::Named("counts") = counts,
    Rcpp::Named("acceptance_rate") = acceptance_rate,
    Rcpp::Named("S") = S,  Rcpp::Named("posterior") = prior_store,
    Rcpp::Named("weights") = weights_store);
}

// [[Rcpp::export]]
arma::mat ngssm_predict2(const Rcpp::List& model_, arma::uvec& prior_types,
  arma::mat& prior_pars, unsigned int n_iter, unsigned int nsim_states,
  unsigned int n_burnin, unsigned int n_thin, double gamma,
  double target_acceptance, arma::mat& S, unsigned int n_ahead,
  unsigned int interval, arma::uvec Z_ind, arma::uvec T_ind,
  arma::uvec R_ind, arma::vec init_signal,
  unsigned int seed) {
  
  
  ngssm model(clone(model_), Z_ind, T_ind, R_ind, seed);
  
  return model.predict2(prior_types, prior_pars, n_iter, nsim_states, n_burnin,
    n_thin, gamma, target_acceptance, S, n_ahead, interval, init_signal);
}


// [[Rcpp::export]]
Rcpp::List ngssm_importance_sample(const Rcpp::List& model_, arma::vec init_signal,
  unsigned int nsim_states,  unsigned int seed) {
  
  ngssm model(model_, seed);
  model.approx(init_signal, model.max_iter, model.conv_tol);
  
  arma::cube alpha = model.sim_smoother(nsim_states, true);
  arma::vec weights = exp(model.importance_weights(alpha) -
    model.scaling_factor(init_signal));
  
  return Rcpp::List::create(
    Rcpp::Named("alpha") = alpha,
    Rcpp::Named("weights") = weights);
}

// [[Rcpp::export]]
Rcpp::List ngssm_approx_model(const Rcpp::List& model_, arma::vec init_signal,
  unsigned int max_iter, double conv_tol) {
  
  ngssm model(model_, 1);
  
  double ll = model.approx(init_signal, max_iter, conv_tol);
  
  return Rcpp::List::create(
    Rcpp::Named("y") = model.y,
    Rcpp::Named("H") = model.H,
    Rcpp::Named("logLik") = ll,
    Rcpp::Named("signal") = init_signal);
}

// [[Rcpp::export]]
Rcpp::List ngssm_particle_filter(const Rcpp::List& model_, unsigned int nsim_states, 
  unsigned int seed,bool bootstrap, arma::vec init_signal) {
  
  ngssm model(model_, seed);
  //fill with zeros in case of zero weights
  arma::cube alphasim(model.m, model.n, nsim_states, arma::fill::zeros);
  arma::mat w(nsim_states, model.n, arma::fill::zeros);
  arma::umat ind(nsim_states, model.n - 1, arma::fill::zeros);
  double ll;
  if(bootstrap) {
    ll = model.particle_filter(nsim_states, alphasim, w, ind);
  } else {
    double ll_g = model.approx(init_signal, model.max_iter, model.conv_tol);
    ll_g += model.log_likelihood(model.distribution != 0);
    arma::vec ll_approx_u = model.scaling_factor_vec(init_signal);
    ll = model.psi_filter(nsim_states, alphasim, w, ind, ll_g, ll_approx_u);
  }
  
  return Rcpp::List::create(
    Rcpp::Named("alpha") = alphasim, Rcpp::Named("w") = w, Rcpp::Named("A") = ind,
    Rcpp::Named("logLik") = ll);
}

// [[Rcpp::export]]
Rcpp::List ngssm_particle_smoother(const Rcpp::List& model_, unsigned int nsim_states, 
  unsigned int seed, bool fs, bool bootstrap, arma::vec init_signal) {
  
  ngssm model(model_, seed);
  
  arma::cube alphasim(model.m, model.n, nsim_states);
  arma::mat w(nsim_states, model.n);
  arma::umat ind(nsim_states, model.n - 1);
  double ll;
  if(bootstrap) {
    ll = model.particle_filter(nsim_states, alphasim, w, ind);
  } else {
    double ll_g = model.approx(init_signal, model.max_iter, model.conv_tol);
    ll_g += model.log_likelihood(model.distribution != 0);
    arma::vec ll_approx_u = model.scaling_factor_vec(init_signal);
    ll = model.psi_filter(nsim_states, alphasim, w, ind, ll_g, ll_approx_u); 
  }
  if(!arma::is_finite(ll)) {
    Rcpp::stop("Particle filtering returned likelihood value of zero. ");
  }
  if(fs) {
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
Rcpp::List ngssm_backward_simulate(const Rcpp::List& model_, unsigned int nsim_states, unsigned int seed,
  unsigned int nsim_store) {
  
  ngssm model(model_, seed);
  
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
