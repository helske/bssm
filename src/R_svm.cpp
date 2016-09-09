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
List svm_mcmc_full(arma::vec& y, arma::mat& Z, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, arma::vec& phi,
  arma::mat& xreg, arma::vec& beta,
  arma::vec& theta_lwr, arma::vec& theta_upr, unsigned int n_iter,
  unsigned int nsim_states, unsigned int n_burnin, unsigned int n_thin,
  double gamma, double target_acceptance, arma::mat S,
  arma::vec& init_signal, unsigned int method, unsigned int seed,double sd_prior,
  unsigned int n_threads, arma::uvec seeds, bool end_ram, bool adapt_approx) {



  svm model(y, Z, T, R, a1, P1, phi, xreg, beta, seed, sd_prior);

  switch(method) {
  case 1 :
    return model.mcmc_full(theta_lwr, theta_upr, n_iter,
      nsim_states, n_burnin, n_thin, gamma, target_acceptance, S, init_signal, end_ram, adapt_approx);
    break;
  case 2 :
    return model.mcmc_da(theta_lwr, theta_upr, n_iter,
      nsim_states, n_burnin, n_thin, gamma, target_acceptance, S, init_signal, end_ram, adapt_approx);
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
        theta_store, ll_store, y_store, H_store, ll_approx_u_store, end_ram, adapt_approx);

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
      theta_store, ll_store, y_store, H_store, ll_approx_u_store, counts, end_ram, adapt_approx);

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
      theta_store, ll_store, y_store, H_store, ll_approx_u_store, counts, end_ram, adapt_approx);

    arma::vec weights_store(counts.n_elem);
    arma::cube alpha_store(model.m, model.n, counts.n_elem);

    is_correction(model, theta_store, y_store, H_store, ll_approx_u_store,
      arma::uvec(counts.n_elem, arma::fill::ones),
      nsim_states, n_threads, seeds, weights_store, alpha_store);

    arma::inplace_trans(theta_store);
    return List::create(Named("alpha") = alpha_store,
      Named("theta") = theta_store, Named("counts") = counts,
      Named("acceptance_rate") = acceptance_rate,
      Named("S") = S,  Named("logLik") = ll_store, Named("weights") = weights_store);
  }
    break;
  // case 6 :
  //   return model.mcmc_da_bsf(theta_lwr, theta_upr, n_iter,
  //     nsim_states, n_burnin, n_thin, gamma, target_acceptance, S, init_signal, end_ram, adapt_approx);
  //   break;
  case 7 : {
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
      theta_store, ll_store, y_store, H_store, ll_approx_u_store, counts, end_ram, adapt_approx);

    arma::vec weights_store(counts.n_elem);
    arma::cube alpha_store(model.m, model.n, counts.n_elem);

    is_correction_bsf(model, theta_store, ll_store,
      arma::uvec(counts.n_elem, arma::fill::ones),
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

// [[Rcpp::export]]
List svm_mcmc_param(arma::vec& y, arma::mat& Z, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, arma::vec& phi,
  arma::mat& xreg, arma::vec& beta,
  arma::vec& theta_lwr, arma::vec& theta_upr, unsigned int n_iter,
  unsigned int nsim_states, unsigned int n_burnin, unsigned int n_thin,
  double gamma, double target_acceptance, arma::mat S,
  arma::vec& init_signal, unsigned int method, unsigned int seed,double sd_prior,
  unsigned int n_threads, arma::uvec seeds, bool end_ram, bool adapt_approx, 
  double ess_treshold) {
  
  svm model(y, Z, T, R, a1, P1, phi, xreg, beta, seed, sd_prior);
  
  switch(method) {
  case 1 :
    return model.mcmc_param(theta_lwr, theta_upr, n_iter,
      nsim_states, n_burnin, n_thin, gamma, target_acceptance, S, init_signal,
      end_ram, adapt_approx);
    break;
  case 2 :
    return model.mcmc_da_param(theta_lwr, theta_upr, n_iter,
      nsim_states, n_burnin, n_thin, gamma, target_acceptance, S, init_signal,
      end_ram, adapt_approx);
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
        theta_store, ll_store, y_store, H_store, ll_approx_u_store, end_ram, adapt_approx);
      
      arma::vec weights_store(n_samples);
      
      is_correction_param(model, theta_store, y_store, H_store, ll_approx_u_store,
        counts, nsim_states, n_threads, seeds, weights_store);
      
      arma::inplace_trans(theta_store);
      return List::create(
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
      theta_store, ll_store, y_store, H_store, ll_approx_u_store, counts,
      end_ram, adapt_approx);
    
    arma::vec weights_store(counts.n_elem);
  
    is_correction_param(model, theta_store, y_store, H_store, ll_approx_u_store, counts,
      nsim_states, n_threads, seeds, weights_store);
    
    arma::inplace_trans(theta_store);
    return List::create(
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
      theta_store, ll_store, y_store, H_store, ll_approx_u_store, counts,
      end_ram, adapt_approx);
    
    arma::vec weights_store(counts.n_elem);
    
    is_correction_param(model, theta_store, y_store, H_store, ll_approx_u_store,
      arma::uvec(counts.n_elem, arma::fill::ones),
      nsim_states, n_threads, seeds, weights_store);
    
    arma::inplace_trans(theta_store);
    return List::create(
      Named("theta") = theta_store, Named("counts") = counts,
      Named("acceptance_rate") = acceptance_rate,
      Named("S") = S,  Named("logLik") = ll_store, Named("weights") = weights_store);
  }
    break;
  // case 7 : {
  //   unsigned int npar = theta_lwr.n_elem;
  //   unsigned int n_samples = floor(n_iter - n_burnin);
  //   arma::mat theta_store(npar, n_samples);
  //   arma::vec ll_store(n_samples);
  //   arma::mat y_store(model.n, n_samples);
  //   arma::mat H_store(model.n, n_samples);
  //   arma::vec ll_approx_u_store(n_samples);
  // 
  //   arma::uvec counts(n_samples);
  //   //no thinning allowed!
  //   double acceptance_rate = model.mcmc_approx2(theta_lwr, theta_upr, n_iter,
  //     nsim_states, n_burnin, 1, gamma, target_acceptance, S, init_signal,
  //     theta_store, ll_store, y_store, H_store, ll_approx_u_store, counts,
  //     end_ram, adapt_approx);
  // 
  //   arma::vec weights_store(counts.n_elem);
  // 
  //   is_correction_bsf_param(model, theta_store, ll_store,
  //     arma::uvec(counts.n_elem, arma::fill::ones),
  //     nsim_states, n_threads, seeds, weights_store, ess_treshold);
  // 
  //   arma::inplace_trans(theta_store);
  //   return List::create(
  //     Named("theta") = theta_store, Named("counts") = counts,
  //     Named("acceptance_rate") = acceptance_rate,
  //     Named("S") = S,  Named("logLik") = ll_store, Named("weights") = weights_store);
  // }
  //   break;
  }
  return List::create(Named("just_in_case") = "should be impossible to see this... Restructure the function later");
}


// [[Rcpp::export]]
List svm_importance_sample(arma::vec& y, arma::mat& Z, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, arma::vec& phi,
  arma::mat& xreg, arma::vec& beta,
  unsigned int nsim_states,
  arma::vec init_signal, unsigned int seed) {

  svm model(y, Z, T, R, a1, P1, phi, xreg, beta, seed, false);

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
  
  svm model(y, Z, T, R, a1, P1, phi, xreg, beta, seed, false);
  
  arma::cube alphasim(model.m, model.n, nsim_states);
  arma::mat V(nsim_states, model.n);
  arma::umat ind(nsim_states, model.n - 1);
  double logU = model.particle_filter(nsim_states, alphasim, V, ind);
  
    return List::create(
      Named("alpha") = alphasim, Named("V") = V, Named("A") = ind,
      Named("logU") = logU);
}


// [[Rcpp::export]]
List svm_particle_filter2(arma::vec& y, arma::mat& Z, arma::cube& T,
  arma::cube& R, arma::vec& a1, arma::mat& P1, arma::vec& phi,
  arma::mat& xreg, arma::vec& beta,
  unsigned int nsim_states,
  arma::vec init_signal, unsigned int seed, double q) {
  
  svm model(y, Z, T, R, a1, P1, phi, xreg, beta, seed, false);
  
  arma::cube alphasim(model.m, model.n, nsim_states);
  arma::mat V(nsim_states, model.n);
  arma::umat ind(nsim_states, model.n - 1);
  double logU = model.particle_filter2(nsim_states, alphasim, V, ind, init_signal, q);
  
  return List::create(
    Named("alpha") = alphasim, Named("V") = V, Named("A") = ind,
    Named("logU") = logU);
}