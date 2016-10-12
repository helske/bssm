#include "svm.h"
#include "is_correction.h"

// [[Rcpp::export]]
double svm_loglik(const List& model_, arma::vec init_signal, unsigned int nsim_states,
  unsigned int seed) {
  
  svm model(model_, seed);
  
  if (nsim_states < 2) {
    model.conv_tol = 1.0e-12;
    model.max_iter = 1000;
  }
  
  double ll = model.approx(init_signal, model.max_iter, model.conv_tol);
  double ll_w = 0;
  if (!std::isfinite(ll)) {
    return -arma::datum::inf;
  } else {
    if (nsim_states > 1) {
      arma::vec weights(nsim_states);
      arma::cube alpha = model.sim_smoother(nsim_states, false);
      weights = model.importance_weights(alpha) - model.scaling_factor(init_signal);
      double maxw = weights.max();
      weights = exp(weights - maxw);
      ll_w = log(arma::mean(weights)) + maxw;
    }
  }
  return model.log_likelihood(false) + ll + ll_w;
}

// [[Rcpp::export]]
List svm_filter(const List& model_, arma::vec init_signal) {
  
  svm model(model_, 1);
  
  double logLik = model.approx(init_signal, 1000, 1e-12);
  
  arma::mat at(model.m, model.n + 1);
  arma::mat att(model.m, model.n);
  arma::cube Pt(model.m, model.m, model.n + 1);
  arma::cube Ptt(model.m, model.m, model.n);
  
  logLik += model.filter(at, att, Pt, Ptt, false);
  
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
arma::mat svm_fast_smoother(const List& model_, arma::vec init_signal) {
  
  svm model(model_, 1);
  model.approx(init_signal, 1000, 1e-12);
  
  return model.fast_smoother(false).t();
}

// [[Rcpp::export]]
arma::cube svm_sim_smoother(const List& model_, unsigned nsim,
  arma::vec init_signal, unsigned int seed) {
  
  svm model(model_, seed);
  model.approx(init_signal, 1000, 1e-12);
  
  return model.sim_smoother(nsim, false);
}
// [[Rcpp::export]]
List svm_smoother(const List& model_, arma::vec init_signal) {
  
  svm model(model_, 1);
  model.approx(init_signal, 1000, 1e-12);
  
  arma::mat alphahat(model.m, model.n);
  arma::cube Vt(model.m, model.m, model.n);
  
  model.smoother(alphahat, Vt, false);
  arma::inplace_trans(alphahat);
  
  return List::create(
    Named("alphahat") = alphahat,
    Named("Vt") = Vt);
}

// [[Rcpp::export]]
List svm_run_mcmc(const List& model_,
  arma::uvec& prior_types, arma::mat& prior_pars, unsigned int n_iter,
  unsigned int nsim_states, unsigned int n_burnin, unsigned int n_thin,
  double gamma, double target_acceptance, arma::mat S,
  arma::vec& init_signal, unsigned int seed, bool end_ram,
  bool adapt_approx, bool da, unsigned int sim_type, bool gkl) {
  
  svm model(clone(model_), seed, gkl);
  
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
  
  return List::create(Named("alpha") = alpha_store,
    Named("theta") = theta_store,
    Named("acceptance_rate") = acceptance_rate,
    Named("S") = S,  Named("posterior") = posterior_store);
}

// [[Rcpp::export]]
List svm_run_mcmc_is(const List& model_,
  arma::uvec& prior_types, arma::mat& prior_pars, unsigned int n_iter,
  unsigned int nsim_states, unsigned int n_burnin, unsigned int n_thin,
  double gamma, double target_acceptance, arma::mat S,
  arma::vec& init_signal, unsigned int seed,
  unsigned int n_threads, bool end_ram, bool adapt_approx, 
  unsigned int sim_type, bool const_m, bool gkl) {
  
  svm model(clone(model_), seed, gkl);
  
  unsigned int npar = prior_types.n_elem;
  unsigned int n_samples = floor((n_iter - n_burnin) / n_thin);
  
  arma::mat y_store(model.n, n_samples);
  arma::mat H_store(model.n, n_samples);
  arma::vec ll_approx_u_store(n_samples);
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
    is_correction(model, theta_store, y_store, H_store, ll_approx_u_store,
      counts, nsim_states, n_threads, weights_store, alpha_store, const_m);
     prior_store += ll_store + weights_store;
    } else {
      if (sim_type == 2) {
      is_correction_bsf(model, theta_store, ll_store,
        counts, nsim_states, n_threads, weights_store, alpha_store, const_m);
      } else {
      is_correction_psif(model, theta_store, y_store, H_store, ll_store,
        counts, nsim_states, n_threads, weights_store, alpha_store, const_m);
    }
    prior_store += weights_store;
  }
  arma::inplace_trans(theta_store);
  return List::create(Named("alpha") = alpha_store,
    Named("theta") = theta_store, Named("counts") = counts,
    Named("acceptance_rate") = acceptance_rate,
    Named("S") = S,  Named("posterior") = prior_store,
    Named("weights") = weights_store);
}


// [[Rcpp::export]]
List svm_importance_sample(const List& model_, arma::vec init_signal, 
   unsigned int nsim_states, unsigned int seed) {
  
  svm model(model_, seed);
  
  model.approx(init_signal, model.max_iter, model.conv_tol);
  
  arma::cube alpha = model.sim_smoother(nsim_states, false);
  arma::vec weights = exp(model.importance_weights(alpha) -
    model.scaling_factor(init_signal));
  
  return List::create(
    Named("alpha") = alpha,
    Named("weights") = weights);
}

// [[Rcpp::export]]
List svm_approx_model(const List& model_, arma::vec init_signal,
  unsigned int max_iter, double conv_tol) {
  
  svm model(model_, 1);
  
  double ll = model.approx(init_signal, max_iter, conv_tol);
  
  
  return List::create(
    Named("y") = model.y,
    Named("H") = model.H,
    Named("scaling_factor") = ll,
    Named("signal") = init_signal);
}


// [[Rcpp::export]]
List svm_particle_filter(const List& model_, unsigned int nsim_states,
  unsigned int seed, bool bootstrap, arma::vec init_signal) {
  
  svm model(model_, seed);
  
  //fill with zeros in case of zero weights
  arma::cube alphasim(model.m, model.n, nsim_states, arma::fill::zeros);
  arma::mat V(nsim_states, model.n, arma::fill::zeros);
  arma::umat ind(nsim_states, model.n - 1, arma::fill::zeros);
  double ll;
  if(bootstrap) {
    ll = model.particle_filter(nsim_states, alphasim, V, ind);
  } else {
    ll = model.psi_filter(nsim_states, alphasim, V, ind, init_signal);  
  }
  
  return List::create(
    Named("alpha") = alphasim, Named("V") = V, Named("A") = ind,
    Named("logLik") = ll);
}

// [[Rcpp::export]]
Rcpp::List svm_particle_smoother(const List& model_, unsigned int nsim_states,
  unsigned int seed, unsigned int method, unsigned int type, arma::vec init_signal) {
  
  
  svm model(model_, seed);
  
  arma::cube alphasim(model.m, model.n, nsim_states);
  arma::mat V(nsim_states, model.n);
  arma::umat ind(nsim_states, model.n - 1);
  double ll = 0.0;
  if (type == 1) {
    ll = model.particle_filter(nsim_states, alphasim, V, ind);
  } else {
    ll = model.psi_filter(nsim_states, alphasim, V, ind, init_signal);
  }
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
Rcpp::List svm_backward_simulate(const List& model_, unsigned int nsim_states,
  unsigned int seed, unsigned int nsim_store) {
  
  svm model(model_, seed);
  
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
