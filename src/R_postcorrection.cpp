#include "approx_mcmc.h"

#include "model_ssm_mlg.h"
#include "model_ssm_mng.h"
#include "model_bsm_lg.h"
#include "model_bsm_ng.h"
#include "model_ar1_lg.h"
#include "model_ar1_ng.h"
#include "model_svm.h"
#include "model_ssm_nlg.h"


// [[Rcpp::export]]
arma::vec suggest_n_nongaussian(const Rcpp::List model_,
  const arma::vec theta, const arma::vec candidates,
  const unsigned int replications, const unsigned int seed,
  const int model_type) {
  
  arma::vec sds(candidates.n_elem);
  switch (model_type) {
  case 0: {
    ssm_mng model(model_, seed);
    model.update_model(theta, model_["update_fn"]);
    for(unsigned int i = 0; i < candidates.n_elem; i++) {
      int nsim = candidates(i);
      arma::cube alpha(model.m, model.n + 1, nsim);
      arma::mat weights(nsim, model.n + 1);
      arma::umat indices(nsim, model.n);
      arma::vec ll(replications);
      for(unsigned int j = 0; j < replications; j ++) {
        arma::vec loglik = model.log_likelihood(1, nsim, alpha, weights, indices);
        ll(j) = loglik(0);
      }
      sds(i) = arma::stddev(ll);
    }
  } break;
  case 1: {
    ssm_ung model(model_, seed);
    model.update_model(theta, model_["update_fn"]);
    for(unsigned int i = 0; i < candidates.n_elem; i++) {
      int nsim = candidates(i);
      arma::cube alpha(model.m, model.n + 1, nsim);
      arma::mat weights(nsim, model.n + 1);
      arma::umat indices(nsim, model.n);
      arma::vec ll(replications);
      for(unsigned int j = 0; j < replications; j ++) {
        arma::vec loglik = model.log_likelihood(1, nsim, alpha, weights, indices);
        ll(j) = loglik(0);
      }
      sds(i) = arma::stddev(ll);
    }
  } break;
  case 2: {
    bsm_ng model(model_, seed);
    model.update_model(theta);
    for(unsigned int i = 0; i < candidates.n_elem; i++) {
      int nsim = candidates(i);
      arma::cube alpha(model.m, model.n + 1, nsim);
      arma::mat weights(nsim, model.n + 1);
      arma::umat indices(nsim, model.n);
      arma::vec ll(replications);
      for(unsigned int j = 0; j < replications; j ++) {
        arma::vec loglik = model.log_likelihood(1, nsim, alpha, weights, indices);
        ll(j) = loglik(0);
      }
      sds(i) = arma::stddev(ll);
    }
  } break;
  case 3: {
    svm model(model_, seed);
    model.update_model(theta);
    for(unsigned int i = 0; i < candidates.n_elem; i++) {
      int nsim = candidates(i);
      arma::cube alpha(model.m, model.n + 1, nsim);
      arma::mat weights(nsim, model.n + 1);
      arma::umat indices(nsim, model.n);
      arma::vec ll(replications);
      for(unsigned int j = 0; j < replications; j ++) {
        arma::vec loglik = model.log_likelihood(1, nsim, alpha, weights, indices);
        ll(j) = loglik(0);
      }
      sds(i) = arma::stddev(ll);
    }
  } break;
  case 4: {
    ar1_ng model(model_, seed);
    model.update_model(theta);
    for(unsigned int i = 0; i < candidates.n_elem; i++) {
      int nsim = candidates(i);
      arma::cube alpha(model.m, model.n + 1, nsim);
      arma::mat weights(nsim, model.n + 1);
      arma::umat indices(nsim, model.n);
      arma::vec ll(replications);
      for(unsigned int j = 0; j < replications; j ++) {
        arma::vec loglik = model.log_likelihood(1, nsim, alpha, weights, indices);
        ll(j) = loglik(0);
      }
      sds(i) = arma::stddev(ll);
    }
  } break;
  }
  
  return sds;
}


// [[Rcpp::export]]
arma::vec suggest_n_nonlinear(const arma::mat& y, SEXP Z, SEXP H,
  SEXP T, SEXP R, SEXP Zg, SEXP Tg, SEXP a1, SEXP P1,
  const arma::vec& theta, SEXP log_prior_pdf, const arma::vec& known_params,
  const arma::mat& known_tv_params, const unsigned int n_states,
  const unsigned int n_etas,  const arma::uvec& time_varying,
  const arma::vec theta_map, const arma::vec candidates,
  const unsigned int replications, const unsigned int seed) {
  
  Rcpp::XPtr<nvec_fnPtr> xpfun_Z(Z);
  Rcpp::XPtr<nmat_fnPtr> xpfun_H(H);
  Rcpp::XPtr<nvec_fnPtr> xpfun_T(T);
  Rcpp::XPtr<nmat_fnPtr> xpfun_R(R);
  Rcpp::XPtr<nmat_fnPtr> xpfun_Zg(Zg);
  Rcpp::XPtr<nmat_fnPtr> xpfun_Tg(Tg);
  Rcpp::XPtr<a1_fnPtr> xpfun_a1(a1);
  Rcpp::XPtr<P1_fnPtr> xpfun_P1(P1);
  Rcpp::XPtr<prior_fnPtr> xpfun_prior(log_prior_pdf);
  
  ssm_nlg model(y, *xpfun_Z, *xpfun_H, *xpfun_T, *xpfun_R, *xpfun_Zg, *xpfun_Tg,
    *xpfun_a1, *xpfun_P1,  theta, *xpfun_prior, known_params, known_tv_params,
    n_states, n_etas,
    time_varying, seed);
  
  model.update_model(theta_map);
  arma::vec sds(candidates.n_elem);
  for(unsigned int i = 0; i < candidates.n_elem; i++) {
    int nsim = candidates(i);
    arma::cube alpha(model.m, model.n + 1, nsim);
    arma::mat weights(nsim, model.n + 1);
    arma::umat indices(nsim, model.n);
    arma::vec ll(replications);
    for(unsigned int j = 0; j < replications; j ++) {
      arma::vec loglik = model.log_likelihood(1, nsim, alpha, weights, indices);
      ll(j) = loglik(0);
    }
    sds(i) = arma::stddev(ll);
  }
  return sds;
}


// [[Rcpp::export]]
Rcpp::List postcorrection_nongaussian(const Rcpp::List model_, 
  const int model_type,
  const unsigned int output_type,
  const unsigned int nsim,
  const unsigned int seed,
  const unsigned int n_threads,
  const unsigned int is_type, 
  const arma::uvec counts, const arma::mat theta,
  const arma::cube modes) {

  arma::vec a1 = Rcpp::as<arma::vec>(model_["a1"]);
  unsigned int m = a1.n_elem;
  unsigned int n;
  unsigned int p;
  if(model_type > 0) {
    arma::vec y = Rcpp::as<arma::vec>(model_["y"]);
    n = y.n_elem;
    p = 1;
  } else {
    arma::mat y = Rcpp::as<arma::mat>(model_["y"]);
    n = y.n_rows;
    p = y.n_cols;
  }

  approx_mcmc mcmc_run(counts.n_elem, 0, 1, n, m, p,
    0.234, 1, arma::mat(theta.n_rows, theta.n_rows), output_type, false);
  
  mcmc_run.n_stored = counts.n_elem;
  // mcmc_run.trim_storage();
  mcmc_run.count_storage = counts;
  mcmc_run.theta_storage = theta;
  mcmc_run.mode_storage = modes;
  mcmc_run.prior_storage.zeros();
  mcmc_run.approx_loglik_storage.zeros();
  
  switch (model_type) {
    case 0: {
      ssm_mng model(model_, seed);
    
      if(is_type == 3) {
        mcmc_run.expand();
      }
      mcmc_run.is_correction_psi(model, nsim, is_type, n_threads, model_["update_fn"]);
      
    } break;
    case 1: {
      ssm_ung model(model_, seed);
      if(is_type == 3) {
        mcmc_run.expand();
      }
      mcmc_run.is_correction_psi(model, nsim, is_type, n_threads, model_["update_fn"]);
     
    } break;
    case 2: {
      bsm_ng model(model_, seed);
      if(is_type == 3) {
        mcmc_run.expand();
      }
      mcmc_run.is_correction_psi(model, nsim, is_type, n_threads);
    } break;
    case 3: {
      svm model(model_, seed);
      if(is_type == 3) {
        mcmc_run.expand();
      }
      mcmc_run.is_correction_psi(model, nsim, is_type, n_threads);
    } break;
    case 4: {
      ar1_ng model(model_, seed);
      if(is_type == 3) {
        mcmc_run.expand();
      }
      mcmc_run.is_correction_psi(model, nsim, is_type, n_threads);
    } break;
  }

  switch (output_type) {
  case 1: {
    return Rcpp::List::create(
      Rcpp::Named("alpha") = mcmc_run.alpha_storage,
      Rcpp::Named("weights") = mcmc_run.weight_storage,
      Rcpp::Named("posterior") = mcmc_run.posterior_storage);
  } break;
  case 2: {
    return Rcpp::List::create(
      Rcpp::Named("alphahat") = mcmc_run.alphahat.t(), 
      Rcpp::Named("Vt") = mcmc_run.Vt,
      Rcpp::Named("weights") = mcmc_run.weight_storage,
      Rcpp::Named("posterior") = mcmc_run.posterior_storage);
  } break;
  case 3: {
    return Rcpp::List::create(
      Rcpp::Named("weights") = mcmc_run.weight_storage,
      Rcpp::Named("posterior") = mcmc_run.posterior_storage);
  } break;
  }

  return Rcpp::List::create(Rcpp::Named("error") = "error");
}


// [[Rcpp::export]]
Rcpp::List postcorrection_nonlinear(const arma::mat& y, SEXP Z, SEXP H,
  SEXP T, SEXP R, SEXP Zg, SEXP Tg, SEXP a1, SEXP P1,
  const arma::vec& theta_init, SEXP log_prior_pdf, const arma::vec& known_params,
  const arma::mat& known_tv_params, const unsigned int n_states,
  const unsigned int n_etas,  const arma::uvec& time_varying,
  const unsigned int output_type,
  const unsigned int nsim,
  const unsigned int seed,
  const unsigned int n_threads,
  const unsigned int is_type, 
  const arma::uvec counts, const arma::mat theta,
  const arma::cube modes) {
  
  Rcpp::XPtr<nvec_fnPtr> xpfun_Z(Z);
  Rcpp::XPtr<nmat_fnPtr> xpfun_H(H);
  Rcpp::XPtr<nvec_fnPtr> xpfun_T(T);
  Rcpp::XPtr<nmat_fnPtr> xpfun_R(R);
  Rcpp::XPtr<nmat_fnPtr> xpfun_Zg(Zg);
  Rcpp::XPtr<nmat_fnPtr> xpfun_Tg(Tg);
  Rcpp::XPtr<a1_fnPtr> xpfun_a1(a1);
  Rcpp::XPtr<P1_fnPtr> xpfun_P1(P1);
  Rcpp::XPtr<prior_fnPtr> xpfun_prior(log_prior_pdf);
  
  ssm_nlg model(y, *xpfun_Z, *xpfun_H, *xpfun_T, *xpfun_R, *xpfun_Zg, *xpfun_Tg,
    *xpfun_a1, *xpfun_P1, theta_init, *xpfun_prior, known_params, known_tv_params,
    n_states, n_etas, time_varying, seed);
  
  approx_mcmc mcmc_run(counts.n_elem, 0, 1, model.n, model.m, model.m,
    0.234, 1, arma::mat(theta.n_rows, theta.n_rows), output_type, false);
  mcmc_run.n_stored = counts.n_elem;
  // mcmc_run.trim_storage();
  mcmc_run.count_storage = counts;
  mcmc_run.theta_storage = theta;
  mcmc_run.mode_storage = modes;
  mcmc_run.prior_storage.zeros();
  mcmc_run.approx_loglik_storage.zeros();
  
  if(is_type == 3) {
    mcmc_run.expand();
  }
  mcmc_run.is_correction_psi(model, nsim, is_type, n_threads);

  
  switch (output_type) {
  case 1: {
    return Rcpp::List::create(
      Rcpp::Named("alpha") = mcmc_run.alpha_storage,
      Rcpp::Named("weights") = mcmc_run.weight_storage,
      Rcpp::Named("posterior") = mcmc_run.posterior_storage);
  } break;
  case 2: {
    return Rcpp::List::create(
      Rcpp::Named("alphahat") = mcmc_run.alphahat.t(), 
      Rcpp::Named("Vt") = mcmc_run.Vt,
      Rcpp::Named("weights") = mcmc_run.weight_storage,
      Rcpp::Named("posterior") = mcmc_run.posterior_storage);
  } break;
  case 3: {
    return Rcpp::List::create(
      Rcpp::Named("weights") = mcmc_run.weight_storage,
      Rcpp::Named("posterior") = mcmc_run.posterior_storage);
  } break;
  }
  
  return Rcpp::List::create(Rcpp::Named("error") = "error");
}
