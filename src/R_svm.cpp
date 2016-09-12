#include "svm.h"


// [[Rcpp::export]]
double svm_loglik(arma::vec& y, arma::mat& Z, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, arma::vec& phi,
  arma::mat& xreg, arma::vec& beta, arma::vec init_signal, unsigned int nsim_states,
  unsigned int seed) {

  svm model(y, Z, T, R, a1, P1, phi, xreg, beta, seed);

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
List svm_smoother(arma::vec& y, arma::mat& Z, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, arma::vec &phi,
  arma::mat& xreg, arma::vec& beta, arma::vec init_signal) {

  svm model(y, Z, T, R, a1, P1, phi, xreg, beta, 1);

  double logLik = model.approx(init_signal, 1000, 1e-12);

  arma::mat alphahat(a1.n_elem, y.n_elem);
  arma::cube Vt(a1.n_elem, a1.n_elem, y.n_elem);

  model.smoother(alphahat, Vt, false);
  arma::inplace_trans(alphahat);

  return List::create(
    Named("alphahat") = alphahat,
    Named("Vt") = Vt);
}

// [[Rcpp::export]]
List svm_run_mcmc(arma::vec& y, arma::mat& Z, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, arma::vec& phi,
  arma::mat& xreg, arma::vec& beta, arma::uvec& prior_types,
  arma::mat& prior_pars, unsigned int n_iter,
  unsigned int nsim_states, unsigned int n_burnin, unsigned int n_thin,
  double gamma, double target_acceptance, arma::mat S,
  arma::vec& init_signal, bool da, unsigned int seed,
  unsigned int n_threads, bool end_ram, bool adapt_approx) {

  svm model(y, Z, T, R, a1, P1, phi, xreg, beta, seed);

  unsigned int npar = prior_types.n_elem;
  unsigned int n_samples = floor((n_iter - n_burnin) / n_thin);
  arma::mat theta_store(npar, n_samples);
  arma::cube alpha_store(model.m, model.n, n_samples);
  arma::vec posterior_store(n_samples);

  double acceptance_rate = model.run_mcmc(prior_types, prior_pars, n_iter, nsim_states, n_burnin,
    n_thin, gamma, target_acceptance, S, init_signal, end_ram, adapt_approx, da,
    theta_store, posterior_store, alpha_store);

  return List::create(Named("alpha") = alpha_store,
    Named("theta") = theta_store,
    Named("acceptance_rate") = acceptance_rate,
    Named("S") = S,  Named("posterior") = posterior_store);
}


// [[Rcpp::export]]
List svm_run_mcmc_is(arma::vec& y, arma::mat& Z, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, arma::vec& phi,
  arma::mat& xreg, arma::vec& beta, arma::uvec& prior_types,
  arma::mat& prior_pars, unsigned int n_iter,
  unsigned int nsim_states, unsigned int n_burnin, unsigned int n_thin,
  double gamma, double target_acceptance, arma::mat S,
  arma::vec& init_signal, bool const_m, unsigned int seed,
  unsigned int n_threads, bool end_ram, bool adapt_approx, bool pf) {

  svm model(y, Z, T, R, a1, P1, phi, xreg, beta, seed);

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
  arma::cube alpha_store(model.m, model.n, counts.n_elem);

  if(pf) {
    is_correction_bsf(model, theta_store, ll_store,
      counts, nsim_states, n_threads, weights_store, alpha_store, const_m);
    prior_store = weights_store;
  } else {
  is_correction(model, theta_store, y_store, H_store, ll_approx_u_store,
    arma::uvec(counts.n_elem, arma::fill::ones),
    nsim_states, n_threads, weights_store, alpha_store, const_m);
    prior_store += ll_store + weights_store;
  }
  arma::inplace_trans(theta_store);
  return List::create(Named("alpha") = alpha_store,
    Named("theta") = theta_store, Named("counts") = counts,
    Named("acceptance_rate") = acceptance_rate,
    Named("S") = S,  Named("posterior") = prior_store,
    Named("weights") = weights_store);
}


// [[Rcpp::export]]
List svm_importance_sample(arma::vec& y, arma::mat& Z, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, arma::vec& phi,
  arma::mat& xreg, arma::vec& beta,
  unsigned int nsim_states,
  arma::vec init_signal, unsigned int seed) {

  svm model(y, Z, T, R, a1, P1, phi, xreg, beta, seed);

  double ll = model.approx(init_signal, model.max_iter, model.conv_tol);

  arma::cube alpha = model.sim_smoother(nsim_states, false);
  arma::vec weights = exp(model.importance_weights(alpha) -
    model.scaling_factor(init_signal));

  return List::create(
    Named("alpha") = alpha,
    Named("weights") = weights);
}

// [[Rcpp::export]]
List svm_approx_model(arma::vec& y, arma::mat& Z, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, arma::vec& phi,
  arma::mat& xreg, arma::vec& beta, arma::vec init_signal, unsigned int max_iter,
  double conv_tol) {

  svm model(y, Z, T, R, a1, P1, phi, xreg, beta, 1);

  double ll = model.approx(init_signal, max_iter, conv_tol);


  return List::create(
    Named("y") = model.y,
    Named("H") = model.H,
    Named("scaling_factor") = ll,
    Named("signal") = init_signal);
}


// [[Rcpp::export]]
List svm_particle_filter(arma::vec& y, arma::mat& Z, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, arma::vec& phi,
  arma::mat& xreg, arma::vec& beta,
  unsigned int nsim_states,
  arma::vec init_signal, unsigned int seed) {

  svm model(y, Z, T, R, a1, P1, phi, xreg, beta, seed);

  arma::cube alphasim(model.m, model.n, nsim_states);
  arma::mat V(nsim_states, model.n);
  arma::umat ind(nsim_states, model.n - 1);
  double logU = model.particle_filter(nsim_states, alphasim, V, ind);

    return List::create(
      Named("alpha") = alphasim, Named("V") = V, Named("A") = ind,
      Named("logU") = logU);
}
