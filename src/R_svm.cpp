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
  arma::vec weights(nsim_states);
  if (nsim_states > 1) {
    arma::cube alpha = model.sim_smoother(nsim_states, false);
    weights = exp(model.importance_weights(alpha) - model.scaling_factor(init_signal));
    ll_w = log(sum(weights) / nsim_states);
  }


  //  return List::create(Named("ll") = ll,
  //    Named("llw") = ll_w, Named("llg") = model.log_likelihood(),
  //    Named("weights") = weights,
  //    Named("y") = model.y,
  //    Named("HH") = model.HH, Named("signal") = init_signal);
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
List svm_mcmc_full(arma::vec& y, arma::mat& Z, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, arma::vec& phi,
  arma::mat& xreg, arma::vec& beta,
  arma::vec& theta_lwr, arma::vec& theta_upr, unsigned int n_iter,
  unsigned int nsim_states, unsigned int n_burnin, unsigned int n_thin,
  double gamma, double target_acceptance, arma::mat S,
  arma::vec& init_signal, unsigned int method, unsigned int seed,
  unsigned int n_threads, arma::uvec seeds) {


  svm model(y, Z, T, R, a1, P1, phi, xreg, beta, seed);

  switch(method) {
  case 1 :
    return model.mcmc_full(theta_lwr, theta_upr, n_iter,
      nsim_states, n_burnin, n_thin, gamma, target_acceptance, S, init_signal);
    break;
  case 2 :
    return model.mcmc_da(theta_lwr, theta_upr, n_iter,
      nsim_states, n_burnin, n_thin, gamma, target_acceptance, S, init_signal);
    break;
  case 3 : {
      unsigned int npar = theta_lwr.n_elem;
      unsigned int n_samples = floor((n_iter - n_burnin) / n_thin);
      arma::mat theta_store(npar, n_samples);
      arma::vec ll_store(n_samples);
      arma::mat y_store(model.n, n_samples);
      arma::mat H_store(model.n, n_samples);
      arma::vec ll_approx_u_store(n_samples);

      arma::uvec counts(n_samples, arma::fill::ones);
      //no thinning allowed!
      double acceptance_rate = model.mcmc_approx(theta_lwr, theta_upr, n_iter,
        nsim_states, n_burnin, 1, gamma, target_acceptance, S, init_signal,
        theta_store, ll_store, y_store, H_store, ll_approx_u_store);

      arma::vec weights_store(n_samples);
      arma::cube alpha_store(model.m, model.n, n_samples);


      is_correction(model, theta_store, y_store, H_store, ll_approx_u_store,
        counts, nsim_states, n_threads, seeds, weights_store, alpha_store);

      arma::inplace_trans(theta_store);
      return List::create(Named("alpha") = alpha_store,
        Named("theta") = theta_store,
        Named("acceptance_rate") = acceptance_rate,
        Named("S") = S,  Named("logLik") = ll_store, Named("weights") = weights_store);
    }
    break;
  case 4 : {
    unsigned int npar = theta_lwr.n_elem;
    unsigned int n_samples = floor(n_iter - n_burnin);
    arma::mat theta_store(npar, n_samples);
    arma::vec ll_store(n_samples);
    arma::mat y_store(model.n, n_samples);
    arma::mat H_store(model.n, n_samples);
    arma::vec ll_approx_u_store(n_samples);

    arma::uvec counts(n_samples);
    //no thinning allowed!
    double acceptance_rate = model.mcmc_approx2(theta_lwr, theta_upr, n_iter,
      nsim_states, n_burnin, 1, gamma, target_acceptance, S, init_signal,
      theta_store, ll_store, y_store, H_store, ll_approx_u_store, counts);

    arma::vec weights_store(counts.n_elem);
    arma::cube alpha_store(model.m, model.n, counts.n_elem);

    is_correction(model, theta_store, y_store, H_store, ll_approx_u_store, counts,
      nsim_states, n_threads, seeds, weights_store, alpha_store);

    arma::inplace_trans(theta_store);
    return List::create(Named("alpha") = alpha_store,
      Named("theta") = theta_store, Named("counts") = counts,
      Named("acceptance_rate") = acceptance_rate,
      Named("S") = S,  Named("logLik") = ll_store, Named("weights") = weights_store);
  }
    break;
  case 5 : {
    unsigned int npar = theta_lwr.n_elem;
    unsigned int n_samples = floor(n_iter - n_burnin);
    arma::mat theta_store(npar, n_samples);
    arma::vec ll_store(n_samples);
    arma::mat y_store(model.n, n_samples);
    arma::mat H_store(model.n, n_samples);
    arma::vec ll_approx_u_store(n_samples);

    arma::uvec counts(n_samples);
    //no thinning allowed!
    double acceptance_rate = model.mcmc_approx2(theta_lwr, theta_upr, n_iter,
      nsim_states, n_burnin, 1, gamma, target_acceptance, S, init_signal,
      theta_store, ll_store, y_store, H_store, ll_approx_u_store, counts);

    arma::vec weights_store(counts.n_elem);
    arma::cube alpha_store(model.m, model.n, counts.n_elem);

    is_correction2(model, theta_store, y_store, H_store, ll_approx_u_store, counts,
      nsim_states, n_threads, seeds, weights_store, alpha_store);

    arma::inplace_trans(theta_store);
    return List::create(Named("alpha") = alpha_store,
      Named("theta") = theta_store, Named("counts") = counts,
      Named("acceptance_rate") = acceptance_rate,
      Named("S") = S,  Named("logLik") = ll_store, Named("weights") = weights_store);
  }
    break;
  }
  return List::create(Named("just_in_case") = "should be impossible to see this... Restructure the function later");
}
