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
    weights = exp(model.importance_weights(alpha, init_signal));
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
  }
return List::create(Named("just_in_case") = "should be impossible to see this... Restructure the function later");
}
