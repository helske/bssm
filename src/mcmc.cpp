#ifdef _OPENMP
#include <omp.h>
#endif
#include <ramcmc.h>
#include "mcmc.h"
#include "ugg_ssm.h"
#include "ung_ssm.h"
#include "ugg_bsm.h"
#include "ung_bsm.h"
#include "ung_svm.h"
#include "nlg_ssm.h"

#include "distr_consts.h"
#include "filter_smoother.h"

mcmc::mcmc(const arma::uvec& prior_distributions, const arma::mat& prior_parameters,
  const unsigned int n_iter, const unsigned int n_burnin, 
  const unsigned int n_thin, const unsigned int n, const unsigned int m,
  const double target_acceptance, const double gamma, const arma::mat& S, 
  const bool store_states) :
  prior_distributions(prior_distributions), prior_parameters(prior_parameters),
  n_iter(n_iter), n_burnin(n_burnin), n_thin(n_thin),
  n_samples(std::floor(static_cast <double> (n_iter - n_burnin) / n_thin)), 
  n_par(prior_distributions.n_elem),
  target_acceptance(target_acceptance), gamma(gamma), n_stored(0),
  posterior_storage(arma::vec(n_samples)), 
  theta_storage(arma::mat(n_par, n_samples)),
  count_storage(arma::uvec(n_samples, arma::fill::zeros)),
  alpha_storage(arma::cube(n, m, store_states * n_samples)), S(S),
  acceptance_rate(0.0) {
}

void mcmc::trim_storage() {
  theta_storage.resize(n_par, n_stored);
  posterior_storage.resize(n_stored);
  count_storage.resize(n_stored);
  alpha_storage.resize(alpha_storage.n_rows, alpha_storage.n_cols, n_stored);
}


template void mcmc::state_posterior(ugg_ssm model, const unsigned int n_threads);
template void mcmc::state_posterior(ugg_bsm model, const unsigned int n_threads);

template <class T>
void mcmc::state_posterior(T model, const unsigned int n_threads) {
  
  if(n_threads > 1) {
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads) default(none) firstprivate(model)
    {
      model.engine = sitmo::prng_engine(omp_get_thread_num() + 1);
      unsigned thread_size = 
        static_cast <unsigned int>(std::floor(static_cast <double> (n_stored) / n_threads));
      unsigned int start = omp_get_thread_num() * thread_size;
      unsigned int end = (omp_get_thread_num() + 1) * thread_size - 1;
      if(omp_get_thread_num() == static_cast<int>(n_threads - 1)) {
        end = n_stored - 1;
      }
      
      arma::mat theta_piece = theta_storage(arma::span::all, arma::span(start, end));
      arma::cube alpha_piece = alpha_storage.slices(start, end);
      state_sampler(model, theta_piece, alpha_piece);
      alpha_storage.slices(start, end) = alpha_piece;
    }
#else
    state_sampler(model, theta_storage, alpha_storage);
#endif
  } else {
    state_sampler(model, theta_storage, alpha_storage);
  }
}


// should parallelize at some point
template void mcmc::state_summary(ugg_ssm model, arma::mat& alphahat, 
  arma::cube& Vt);
template void mcmc::state_summary(ugg_bsm model, arma::mat& alphahat, 
  arma::cube& Vt);

template <class T>
void mcmc::state_summary(T model, arma::mat& alphahat, arma::cube& Vt) {
  
  state_sampler(model, theta_storage, alpha_storage);
  arma::cube Valpha(model.m, model.m, model.n, arma::fill::zeros);
  
  arma::vec theta = theta_storage.col(0);
  model.set_theta(theta);
  model.smoother(alphahat, Vt);
  
  double sum_w = count_storage(0);
  arma::mat alphahat_i = alphahat;
  arma::cube Vt_i = Vt;
  
  for (unsigned int i = 1; i < n_stored; i++) {
    arma::vec theta = theta_storage.col(i);
    model.set_theta(theta);
    model.smoother(alphahat_i, Vt_i);
    
    arma::mat diff = alphahat_i - alphahat;
    double tmp = count_storage(i) + sum_w;
    alphahat = (alphahat * sum_w + alphahat_i * count_storage(i)) / tmp;
    
    for (unsigned int t = 0; t < model.n; t++) {
      Valpha.slice(t) += diff.col(t) * (alphahat_i.col(t) - alphahat.col(t)).t();
    }
    Vt = (Vt * sum_w + Vt_i * count_storage(i)) / tmp;
    sum_w = tmp;
  }
  Vt += Valpha / n_samples; // Var[E(alpha)] + E[Var(alpha)]
  
  
}


template void mcmc::state_sampler(ugg_ssm& model, const arma::mat& theta, arma::cube& alpha);
template void mcmc::state_sampler(ugg_bsm& model, const arma::mat& theta, arma::cube& alpha);

template <class T>
void mcmc::state_sampler(T& model, const arma::mat& theta, arma::cube& alpha) {
  for (unsigned int i = 0; i < theta.n_cols; i++) {
    arma::vec theta_i = theta.col(i);
    model.set_theta(theta_i);
    alpha.slice(i) = model.simulate_states(1).slice(0).t();
  }
}

///// TODO: Change to enums later!!!!
//log-prior_pdf
// type 0 = uniform distribution
// type 1 = half-normal
// type 2 = normal
//
double mcmc::log_prior_pdf(const arma::vec& theta) const {
  
  double q = 0.0;
  for(unsigned int i = 0; i < theta.n_elem; i++) {
    switch(prior_distributions(i)) {
    case 0  :
      q += R::dunif(theta(i), prior_parameters(0, i), prior_parameters(1, i), 1);
      break;
    case 1  :
      if (theta(i) < 0) {
        return -arma::datum::inf;
      } else {
        q += std::log(2.0) + R::dnorm(theta(i), 0, prior_parameters(0, i), 1);
      }
      break;
    case 2  :
      q += R::dnorm(theta(i), prior_parameters(0, i), prior_parameters(1, i), 1);
      break;
    }
  }
  return q;
}

// double mcmc::proposal(const arma::vec& theta, const arma::vec& theta_proposal) const {
//   return arma::accu(theta_prop - theta);
// }
// run MCMC for linear-Gaussian state space model
// target the marginal p(theta | y)
// sample states separately given the posterior sample of theta
template void mcmc::mcmc_gaussian(ugg_ssm model, const bool end_ram);
template void mcmc::mcmc_gaussian(ugg_bsm model, const bool end_ram);

template<class T>
void mcmc::mcmc_gaussian(T model, const bool end_ram) {
  
  arma::vec theta = model.get_theta();
  double logprior = log_prior_pdf(theta);
  double loglik = model.log_likelihood();
  
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  
  double acceptance_prob = 0.0;
  bool new_value = true;
  unsigned int n_values = 0;
  for (unsigned int i = 1; i <= n_iter; i++) {
    
    if (i % 16 == 0) {
      Rcpp::checkUserInterrupt();
    }
    
    // sample from standard normal distribution
    arma::vec u(n_par);
    for(unsigned int j = 0; j < n_par; j++) {
      u(j) = normal(model.engine);
    }
    // propose new theta
    arma::vec theta_prop = theta + S * u;
    // compute prior
    double logprior_prop = log_prior_pdf(theta_prop);
    
    if (logprior_prop > -arma::datum::inf) {
      // update parameters
      model.set_theta(theta_prop);
      // compute log-likelihood with proposed theta
      double loglik_prop = model.log_likelihood();
      //compute the acceptance probability
      // use explicit min(...) as we need this value later
      // double q = proposal(theta, theta_prop);
      acceptance_prob = std::min(1.0, 
        std::exp(loglik_prop - loglik + logprior_prop - logprior));
      //accept
      if (unif(model.engine) < acceptance_prob) {
        if (i > n_burnin) {
          acceptance_rate++;
          n_values++;
        }
        loglik = loglik_prop;
        logprior = logprior_prop;
        theta = theta_prop;
        new_value = true;
        
      }
    } else acceptance_prob = 0.0;
    
    if (i > n_burnin && n_values % n_thin == 0) {
      //new block
      if (new_value) {
        posterior_storage(n_stored) = logprior + loglik;
        theta_storage.col(n_stored) = theta;
        count_storage(n_stored) = 1;
        n_stored++;
        new_value = false;
      } else {
        count_storage(n_stored - 1)++;
      }
    }
    if (!end_ram || i <= n_burnin) {
      ramcmc::adapt_S(S, u, acceptance_prob, target_acceptance, i, gamma);
    }
    
  }
  
  trim_storage();
  acceptance_rate /= (n_iter - n_burnin);
}

// run pseudo-marginal MCMC for non-linear and/or non-Gaussian state space model
// using psi-PF
template void mcmc::pm_mcmc_spdk(ung_ssm model, const bool end_ram, 
  const unsigned int nsim_states, const bool local_approx, 
  const arma::vec& initial_mode, const unsigned int max_iter, const double conv_tol);
template void mcmc::pm_mcmc_spdk(ung_bsm model, const bool end_ram, 
  const unsigned int nsim_states, const bool local_approx, 
  const arma::vec& initial_mode, const unsigned int max_iter, const double conv_tol);
template void mcmc::pm_mcmc_spdk(ung_svm model, const bool end_ram, 
  const unsigned int nsim_states, const bool local_approx, 
  const arma::vec& initial_mode, const unsigned int max_iter, const double conv_tol);

template<class T>
void mcmc::pm_mcmc_spdk(T model, const bool end_ram, const unsigned int nsim_states, 
  const bool local_approx, const arma::vec& initial_mode, const unsigned int max_iter, 
  const double conv_tol) {
  
  // get the current values of theta
  arma::vec theta = model.get_theta();
  // compute the log[p(theta)]
  double logprior = log_prior_pdf(theta);
  
  // construct the approximate Gaussian model
  arma::vec mode_estimate = initial_mode;
  ugg_ssm approx_model = model.approximate(mode_estimate, max_iter, conv_tol);
  
  // compute the log-likelihood of the approximate model
  double gaussian_loglik = approx_model.log_likelihood();
  
  
  // compute unnormalized mode-based correction terms 
  // log[g(y_t | ^alpha_t) / ~g(y_t | ^alpha_t)]
  arma::vec scales = model.scaling_factors(approx_model, mode_estimate);
  
  double sum_scales = arma::accu(scales);
  // compute the constant term
  double const_term = compute_const_term(model, approx_model); 
  
  arma::cube alpha = approx_model.simulate_states(nsim_states, true);
  
  arma::vec weights = arma::exp(model.importance_weights(approx_model, alpha) - sum_scales);
  std::discrete_distribution<unsigned int> sample(weights.begin(), weights.end());
  unsigned int ind = sample(model.engine);
  arma::mat sampled_alpha = alpha.slice(ind);
  double ll_w = std::log(arma::accu(weights) / nsim_states);
  
  double loglik = gaussian_loglik + const_term + sum_scales + ll_w;
  double acceptance_prob = 0.0;
  bool new_value = true;
  unsigned int n_values = 0;
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  for (unsigned int i = 1; i <= n_iter; i++) {
    
    if (i % 16 == 0) {
      Rcpp::checkUserInterrupt();
    }
    
    // sample from standard normal distribution
    arma::vec u(n_par);
    for(unsigned int j = 0; j < n_par; j++) {
      u(j) = normal(model.engine);
    }
    
    // propose new theta
    arma::vec theta_prop = theta + S * u;
    // compute prior
    double logprior_prop = log_prior_pdf(theta_prop);
    
    if (logprior_prop > -arma::datum::inf) {
      // update parameters
      model.set_theta(theta_prop);
      
      if (local_approx) {
        // construct the approximate Gaussian model
        mode_estimate = initial_mode;
        model.approximate(approx_model, mode_estimate, max_iter, conv_tol);
      } else {
        model.approximate(approx_model, mode_estimate, 0, conv_tol);
      }
      // compute unnormalized mode-based correction terms 
      // log[g(y_t | ^alpha_t) / ~g(y_t | ^alpha_t)]
      scales = model.scaling_factors(approx_model, mode_estimate);
      sum_scales = arma::accu(scales);
      // compute the constant term
      const_term = compute_const_term(model, approx_model);
      
      alpha = approx_model.simulate_states(nsim_states, true);
      weights = arma::exp(model.importance_weights(approx_model, alpha) - sum_scales);
      ll_w = std::log(arma::accu(weights) / nsim_states);
      
      
      double loglik_prop = 
        approx_model.log_likelihood() + const_term + sum_scales + ll_w;
      
      //compute the acceptance probability
      // use explicit min(...) as we need this value later
      // double q = proposal(theta, theta_prop);
      
      acceptance_prob = std::min(1.0, std::exp(loglik_prop - loglik + 
        logprior_prop - logprior));
      
      //accept
      if (unif(model.engine) < acceptance_prob) {
        if (i > n_burnin) {
          acceptance_rate++;
          n_values++;
        }
        std::discrete_distribution<unsigned int> sample(weights.begin(), weights.end());
        sampled_alpha = alpha.slice(ind);
        loglik = loglik_prop;
        logprior = logprior_prop;
        theta = theta_prop;
        new_value = true;
        
      }
    } else acceptance_prob = 0.0;
    
    if (i > n_burnin && n_values % n_thin == 0) {
      //new block
      if (new_value) {
        posterior_storage(n_stored) = logprior + loglik;
        theta_storage.col(n_stored) = theta;
        count_storage(n_stored) = 1;
        alpha_storage.slice(n_stored) = sampled_alpha.t();
        n_stored++;
        new_value = false;
      } else {
        count_storage(n_stored - 1)++;
      }
    }
    
    if (!end_ram || i <= n_burnin) {
      ramcmc::adapt_S(S, u, acceptance_prob, target_acceptance, i, gamma);
    }
  }
  
  trim_storage();
  acceptance_rate /= (n_iter - n_burnin);
}

// run pseudo-marginal MCMC for non-linear and/or non-Gaussian state space model
// using psi-PF
template void mcmc::pm_mcmc_psi(ung_ssm model, const bool end_ram, 
  const unsigned int nsim_states, const bool local_approx, 
  const arma::vec& initial_mode, const unsigned int max_iter, const double conv_tol);
template void mcmc::pm_mcmc_psi(ung_bsm model, const bool end_ram, 
  const unsigned int nsim_states, const bool local_approx, 
  const arma::vec& initial_mode, const unsigned int max_iter, const double conv_tol);
template void mcmc::pm_mcmc_psi(ung_svm model, const bool end_ram, 
  const unsigned int nsim_states, const bool local_approx, 
  const arma::vec& initial_mode, const unsigned int max_iter, const double conv_tol);

template<class T>
void mcmc::pm_mcmc_psi(T model, const bool end_ram, const unsigned int nsim_states, 
  const bool local_approx, const arma::vec& initial_mode, const unsigned int max_iter, 
  const double conv_tol) {
  
  unsigned int m = model.m;
  unsigned n = model.n;
  
  // get the current values of theta
  arma::vec theta = model.get_theta();
  // compute the log[p(theta)]
  double logprior = log_prior_pdf(theta);
  
  // construct the approximate Gaussian model
  arma::vec mode_estimate = initial_mode;
  ugg_ssm approx_model = model.approximate(mode_estimate, max_iter, conv_tol);
  
  // compute the log-likelihood of the approximate model
  double gaussian_loglik = approx_model.log_likelihood();
  
  
  // compute unnormalized mode-based correction terms 
  // log[g(y_t | ^alpha_t) / ~g(y_t | ^alpha_t)]
  arma::vec scales = model.scaling_factors(approx_model, mode_estimate);
  
  double sum_scales = arma::accu(scales);
  // compute the constant term
  double const_term = compute_const_term(model, approx_model); 
  // log-likelihood approximation
  double approx_loglik = gaussian_loglik + const_term + sum_scales;
  
  arma::cube alpha(m, n, nsim_states);
  arma::mat weights(nsim_states, n);
  arma::umat indices(nsim_states, n - 1);
  double loglik = model.psi_filter(approx_model, approx_loglik, scales, 
    nsim_states, alpha, weights, indices);
  
  filter_smoother(alpha, indices);
  arma::vec w = weights.col(n - 1);
  std::discrete_distribution<unsigned int> sample0(w.begin(), w.end());
  arma::mat sampled_alpha = alpha.slice(sample0(model.engine));
  
  double acceptance_prob = 0.0;
  bool new_value = true;
  unsigned int n_values = 0;
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  for (unsigned int i = 1; i <= n_iter; i++) {
    
    if (i % 16 == 0) {
      Rcpp::checkUserInterrupt();
    }
    
    // sample from standard normal distribution
    arma::vec u(n_par);
    for(unsigned int j = 0; j < n_par; j++) {
      u(j) = normal(model.engine);
    }
    
    // propose new theta
    arma::vec theta_prop = theta + S * u;
    // compute prior
    double logprior_prop = log_prior_pdf(theta_prop);
    
    if (logprior_prop > -arma::datum::inf) {
      // update parameters
      model.set_theta(theta_prop);
      
      if (local_approx) {
        // construct the approximate Gaussian model
        mode_estimate = initial_mode;
        model.approximate(approx_model, mode_estimate, max_iter, conv_tol);
      } else {
        model.approximate(approx_model, mode_estimate, 0, conv_tol);
      }
      // compute unnormalized mode-based correction terms 
      // log[g(y_t | ^alpha_t) / ~g(y_t | ^alpha_t)]
      scales = model.scaling_factors(approx_model, mode_estimate);
      sum_scales = arma::accu(scales);
      // compute the constant term
      const_term = compute_const_term(model, approx_model);
      // compute the log-likelihood of the approximate model
      gaussian_loglik = approx_model.log_likelihood();
      approx_loglik = gaussian_loglik + const_term + sum_scales;
      
      double loglik_prop = model.psi_filter(approx_model, approx_loglik, scales, 
        nsim_states, alpha, weights, indices);
      
      //compute the acceptance probability
      // use explicit min(...) as we need this value later
      // double q = proposal(theta, theta_prop);
      
      acceptance_prob = std::min(1.0, std::exp(loglik_prop - loglik + 
        logprior_prop - logprior));
      
      //accept
      if (unif(model.engine) < acceptance_prob) {
        if (i > n_burnin) {
          acceptance_rate++;
          n_values++;
        }
        filter_smoother(alpha, indices);
        w = weights.col(n - 1);
        std::discrete_distribution<unsigned int> sample(w.begin(), w.end());
        sampled_alpha = alpha.slice(sample(model.engine));
        loglik = loglik_prop;
        logprior = logprior_prop;
        theta = theta_prop;
        new_value = true;
        
      }
    } else acceptance_prob = 0.0;
    
    if (i > n_burnin && n_values % n_thin == 0) {
      //new block
      if (new_value) {
        posterior_storage(n_stored) = logprior + loglik;
        theta_storage.col(n_stored) = theta;
        count_storage(n_stored) = 1;
        alpha_storage.slice(n_stored) = sampled_alpha.t();
        n_stored++;
        new_value = false;
      } else {
        count_storage(n_stored - 1)++;
      }
    }
    
    if (!end_ram || i <= n_burnin) {
      ramcmc::adapt_S(S, u, acceptance_prob, target_acceptance, i, gamma);
    }
  }
  
  trim_storage();
  acceptance_rate /= (n_iter - n_burnin);
}

//run pseudo-marginal MCMC for non-linear and/or non-Gaussian state space model
//using bsf-PF

template void mcmc::pm_mcmc_bsf(ung_ssm model, const bool end_ram,
  const unsigned int nsim_states);
template void mcmc::pm_mcmc_bsf(ung_bsm model, const bool end_ram,
  const unsigned int nsim_states);
template void mcmc::pm_mcmc_bsf(ung_svm model, const bool end_ram,
  const unsigned int nsim_states);

template<class T>
void mcmc::pm_mcmc_bsf(T model, const bool end_ram, const unsigned int nsim_states) {
  
  unsigned int m = model.m;
  unsigned n = model.n;
  
  // get the current values of theta
  arma::vec theta = model.get_theta();
  
  // compute the log[p(theta)]
  double logprior = log_prior_pdf(theta);
  
  arma::cube alpha(m, n, nsim_states);
  arma::mat weights(nsim_states, n);
  arma::umat indices(nsim_states, n - 1);
  double loglik = model.bsf_filter(nsim_states, alpha, weights, indices);
  
  filter_smoother(alpha, indices);
  arma::vec w = weights.col(n - 1);
  std::discrete_distribution<unsigned int> sample0(w.begin(), w.end());
  arma::mat sampled_alpha = alpha.slice(sample0(model.engine));
  
  double acceptance_prob = 0.0;
  bool new_value = true;
  unsigned int n_values = 0;
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  
  for (unsigned int i = 1; i <= n_iter; i++) {
    
    if (i % 16 == 0) {
      Rcpp::checkUserInterrupt();
    }
    
    // sample from standard normal distribution
    arma::vec u(n_par);
    for(unsigned int j = 0; j < n_par; j++) {
      u(j) = normal(model.engine);
    }
    
    // propose new theta
    arma::vec theta_prop = theta + S * u;
    // compute prior
    double logprior_prop = log_prior_pdf(theta_prop);
    
    if (logprior_prop > -arma::datum::inf) {
      // update parameters
      model.set_theta(theta_prop);
      
      double loglik_prop = model.bsf_filter(nsim_states, alpha, weights, indices);
      
      //compute the acceptance probability
      // use explicit min(...) as we need this value later
      //  double q = proposal(theta, theta_prop);
      
      acceptance_prob = std::min(1.0, std::exp(loglik_prop - loglik +
        logprior_prop - logprior));
      //accept
      
      if (unif(model.engine) < acceptance_prob) {
        if (i > n_burnin) {
          acceptance_rate++;
          n_values++;
        }
        filter_smoother(alpha, indices);
        w = weights.col(n - 1);
        std::discrete_distribution<unsigned int> sample(w.begin(), w.end());
        sampled_alpha = alpha.slice(sample(model.engine));
        loglik = loglik_prop;
        logprior = logprior_prop;
        theta = theta_prop;
        new_value = true;
      }
    } else acceptance_prob = 0.0;
    
    if (i > n_burnin && n_values % n_thin == 0) {
      //new block
      if (new_value) {
        posterior_storage(n_stored) = logprior + loglik;
        theta_storage.col(n_stored) = theta;
        count_storage(n_stored) = 1;
        alpha_storage.slice(n_stored) = sampled_alpha.t();
        n_stored++;
        new_value = false;
      } else {
        count_storage(n_stored - 1)++;
      }
    }
    
    if (!end_ram || i <= n_burnin) {
      ramcmc::adapt_S(S, u, acceptance_prob, target_acceptance, i, gamma);
    }
  }
  
  trim_storage();
  acceptance_rate /= (n_iter - n_burnin);
}

// run delayed acceptance pseudo-marginal MCMC for 
// non-linear and/or non-Gaussian state space model
// using SPDK importance sampling
template void mcmc::da_mcmc_spdk(ung_ssm model, const bool end_ram, 
  const unsigned int nsim_states, const bool local_approx, 
  const arma::vec& initial_mode, const unsigned int max_iter, const double conv_tol);
template void mcmc::da_mcmc_spdk(ung_bsm model, const bool end_ram, 
  const unsigned int nsim_states, const bool local_approx, 
  const arma::vec& initial_mode, const unsigned int max_iter, const double conv_tol);
template void mcmc::da_mcmc_spdk(ung_svm model, const bool end_ram, 
  const unsigned int nsim_states, const bool local_approx, 
  const arma::vec& initial_mode, const unsigned int max_iter, const double conv_tol);

template<class T>
void mcmc::da_mcmc_spdk(T model, const bool end_ram, const unsigned int nsim_states, 
  const bool local_approx, const arma::vec& initial_mode, const unsigned int max_iter, 
  const double conv_tol) {
  
  
  // get the current values of theta
  arma::vec theta = model.get_theta();
  // compute the log[p(theta)]
  double logprior = log_prior_pdf(theta);
  
  // construct the approximate Gaussian model
  arma::vec mode_estimate = initial_mode;
  ugg_ssm approx_model = model.approximate(mode_estimate, max_iter, conv_tol);
  
  // compute the log-likelihood of the approximate model
  double gaussian_loglik = approx_model.log_likelihood();
  
  // compute unnormalized mode-based correction terms 
  // log[g(y_t | ^alpha_t) / ~g(y_t | ^alpha_t)]
  arma::vec scales = model.scaling_factors(approx_model, mode_estimate);
  
  // compute the constant term
  double const_term = compute_const_term(model, approx_model); 
  // log-likelihood approximation
  double sum_scales = arma::accu(scales);
  double approx_loglik = gaussian_loglik + const_term + sum_scales;
  
  arma::cube alpha = approx_model.simulate_states(nsim_states, true);
  arma::vec weights = arma::exp(model.importance_weights(approx_model, alpha) - sum_scales);
  std::discrete_distribution<unsigned int> sample(weights.begin(), weights.end());
  unsigned int ind = sample(model.engine);
  arma::mat sampled_alpha = alpha.slice(ind);
  double ll_w = std::log(arma::accu(weights) / nsim_states);
  
  double loglik = gaussian_loglik + const_term + sum_scales + ll_w;
  
  double acceptance_prob = 0.0;
  bool new_value = true;
  unsigned int n_values = 0;
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  for (unsigned int i = 1; i <= n_iter; i++) {
    
    if (i % 16 == 0) {
      Rcpp::checkUserInterrupt();
    }
    
    // sample from standard normal distribution
    arma::vec u(n_par);
    for(unsigned int j = 0; j < n_par; j++) {
      u(j) = normal(model.engine);
    }
    
    // propose new theta
    arma::vec theta_prop = theta + S * u;
    // compute prior
    double logprior_prop = log_prior_pdf(theta_prop);
    
    if (logprior_prop > -arma::datum::inf) {
      // update parameters
      model.set_theta(theta_prop);
      
      if (local_approx) {
        // construct the approximate Gaussian model
        mode_estimate = initial_mode;
        model.approximate(approx_model, mode_estimate, max_iter, conv_tol);
      } else {
        model.approximate(approx_model, mode_estimate, 0, conv_tol);
      }
      // compute unnormalized mode-based correction terms 
      // log[g(y_t | ^alpha_t) / ~g(y_t | ^alpha_t)]
      scales = model.scaling_factors(approx_model, mode_estimate);
      sum_scales = arma::accu(scales);
      // compute the constant term
      const_term = compute_const_term(model, approx_model);
      // compute the log-likelihood of the approximate model
      gaussian_loglik = approx_model.log_likelihood();
      double approx_loglik_prop = gaussian_loglik + const_term + sum_scales;
      
      // stage 1 acceptance probability, used in RAM as well
      acceptance_prob = std::min(1.0, std::exp(approx_loglik_prop - approx_loglik + 
        logprior_prop - logprior));
      
      // initial acceptance
      if (unif(model.engine) < acceptance_prob) {
        
        alpha = approx_model.simulate_states(nsim_states, true);
        weights = arma::exp(model.importance_weights(approx_model, alpha) - sum_scales);
        double ll_w_prop = std::log(arma::accu(weights) / nsim_states);
        
        //just in case
        if(std::isfinite(ll_w_prop)) {
          // delayed acceptance ratio, in log-scale
          double acceptance_prob2 = ll_w_prop - ll_w;
          if (std::log(unif(model.engine)) < acceptance_prob2) {
            
            if (i > n_burnin) {
              acceptance_rate++;
              n_values++;
            }
            std::discrete_distribution<unsigned int> sample(weights.begin(), weights.end());
            sampled_alpha = alpha.slice(sample(model.engine));
            approx_loglik = approx_loglik_prop;
            loglik = approx_loglik + ll_w_prop;
            logprior = logprior_prop;
            ll_w = ll_w_prop;
            theta = theta_prop;
            new_value = true;
          }
        }
      }
    } else acceptance_prob = 0.0;
    
    if (i > n_burnin && n_values % n_thin == 0) {
      //new block
      if (new_value) {
        posterior_storage(n_stored) = logprior + loglik;
        theta_storage.col(n_stored) = theta;
        count_storage(n_stored) = 1;
        alpha_storage.slice(n_stored) = sampled_alpha.t();
        n_stored++;
        new_value = false;
      } else {
        count_storage(n_stored - 1)++;
      }
    }
    
    if (!end_ram || i <= n_burnin) {
      ramcmc::adapt_S(S, u, acceptance_prob, target_acceptance, i, gamma);
    }
  }
  
  trim_storage();
  acceptance_rate /= (n_iter - n_burnin);
}

// run delayed acceptance pseudo-marginal MCMC for 
// non-linear and/or non-Gaussian state space model
// using psi-PF
template void mcmc::da_mcmc_psi(ung_ssm model, const bool end_ram, 
  const unsigned int nsim_states, const bool local_approx, 
  const arma::vec& initial_mode, const unsigned int max_iter, const double conv_tol);
template void mcmc::da_mcmc_psi(ung_bsm model, const bool end_ram, 
  const unsigned int nsim_states, const bool local_approx, 
  const arma::vec& initial_mode, const unsigned int max_iter, const double conv_tol);
template void mcmc::da_mcmc_psi(ung_svm model, const bool end_ram, 
  const unsigned int nsim_states, const bool local_approx, 
  const arma::vec& initial_mode, const unsigned int max_iter, const double conv_tol);

template<class T>
void mcmc::da_mcmc_psi(T model, const bool end_ram, const unsigned int nsim_states, 
  const bool local_approx, const arma::vec& initial_mode, const unsigned int max_iter, 
  const double conv_tol) {
  
  unsigned int m = model.m;
  unsigned n = model.n;
  
  // get the current values of theta
  arma::vec theta = model.get_theta();
  // compute the log[p(theta)]
  double logprior = log_prior_pdf(theta);
  
  // construct the approximate Gaussian model
  arma::vec mode_estimate = initial_mode;
  ugg_ssm approx_model = model.approximate(mode_estimate, max_iter, conv_tol);
  
  // compute the log-likelihood of the approximate model
  double gaussian_loglik = approx_model.log_likelihood();
  
  // compute unnormalized mode-based correction terms 
  // log[g(y_t | ^alpha_t) / ~g(y_t | ^alpha_t)]
  arma::vec scales = model.scaling_factors(approx_model, mode_estimate);
  double sum_scales = arma::accu(scales);
  // compute the constant term
  double const_term = compute_const_term(model, approx_model); 
  // log-likelihood approximation
  double approx_loglik = gaussian_loglik + const_term + sum_scales;
  
  arma::cube alpha(m, n, nsim_states);
  arma::mat weights(nsim_states, n);
  arma::umat indices(nsim_states, n - 1);
  double loglik = model.psi_filter(approx_model, approx_loglik, scales, 
    nsim_states, alpha, weights, indices);
  
  filter_smoother(alpha, indices);
  arma::vec w = weights.col(n - 1);
  std::discrete_distribution<unsigned int> sample0(w.begin(), w.end());
  arma::mat sampled_alpha = alpha.slice(sample0(model.engine));
  
  double acceptance_prob = 0.0;
  bool new_value = true;
  unsigned int n_values = 0;
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  for (unsigned int i = 1; i <= n_iter; i++) {
    
    if (i % 16 == 0) {
      Rcpp::checkUserInterrupt();
    }
    
    // sample from standard normal distribution
    arma::vec u(n_par);
    for(unsigned int j = 0; j < n_par; j++) {
      u(j) = normal(model.engine);
    }
    
    // propose new theta
    arma::vec theta_prop = theta + S * u;
    // compute prior
    double logprior_prop = log_prior_pdf(theta_prop);
    
    if (logprior_prop > -arma::datum::inf) {
      // update parameters
      model.set_theta(theta_prop);
      
      if (local_approx) {
        // construct the approximate Gaussian model
        mode_estimate = initial_mode;
        model.approximate(approx_model, mode_estimate, max_iter, conv_tol);
      } else {
        model.approximate(approx_model, mode_estimate, 0, conv_tol);
      }
      // compute unnormalized mode-based correction terms 
      // log[g(y_t | ^alpha_t) / ~g(y_t | ^alpha_t)]
      scales = model.scaling_factors(approx_model, mode_estimate);
      sum_scales = arma::accu(scales);
      // compute the constant term
      const_term = compute_const_term(model, approx_model);
      // compute the log-likelihood of the approximate model
      gaussian_loglik = approx_model.log_likelihood();
      double approx_loglik_prop = gaussian_loglik + const_term + sum_scales;
      
      // stage 1 acceptance probability, used in RAM as well
      acceptance_prob = std::min(1.0, std::exp(approx_loglik_prop - approx_loglik + 
        logprior_prop - logprior));
      
      // initial acceptance
      if (unif(model.engine) < acceptance_prob) {
        
        double loglik_prop = model.psi_filter(approx_model, approx_loglik_prop, scales, 
          nsim_states, alpha, weights, indices);
        
        //just in case
        if(std::isfinite(loglik_prop)) {
          // delayed acceptance ratio, in log-scale
          double acceptance_prob2 = loglik_prop + approx_loglik - loglik - approx_loglik_prop;
          if (std::log(unif(model.engine)) < acceptance_prob2) {
            
            if (i > n_burnin) {
              acceptance_rate++;
              n_values++;
            }
            filter_smoother(alpha, indices);
            w = weights.col(n - 1);
            std::discrete_distribution<unsigned int> sample(w.begin(), w.end());
            sampled_alpha = alpha.slice(sample(model.engine));
            approx_loglik = approx_loglik_prop;
            loglik = loglik_prop;
            logprior = logprior_prop;
            theta = theta_prop;
            new_value = true;
          }
        }
      }
    } else acceptance_prob = 0.0;
    
    if (i > n_burnin && n_values % n_thin == 0) {
      //new block
      if (new_value) {
        posterior_storage(n_stored) = logprior + loglik;
        theta_storage.col(n_stored) = theta;
        count_storage(n_stored) = 1;
        alpha_storage.slice(n_stored) = sampled_alpha.t();
        n_stored++;
        new_value = false;
      } else {
        count_storage(n_stored - 1)++;
      }
    }
    
    if (!end_ram || i <= n_burnin) {
      ramcmc::adapt_S(S, u, acceptance_prob, target_acceptance, i, gamma);
    }
  }
  
  trim_storage();
  acceptance_rate /= (n_iter - n_burnin);
}
// run delayed acceptance pseudo-marginal MCMC for 
// non-linear and/or non-Gaussian state space model
// using bootstrap filter
template void mcmc::da_mcmc_bsf(ung_ssm model, const bool end_ram, 
  const unsigned int nsim_states, const bool local_approx, 
  const arma::vec& initial_mode, const unsigned int max_iter, const double conv_tol);
template void mcmc::da_mcmc_bsf(ung_bsm model, const bool end_ram, 
  const unsigned int nsim_states, const bool local_approx, 
  const arma::vec& initial_mode, const unsigned int max_iter, const double conv_tol);
template void mcmc::da_mcmc_bsf(ung_svm model, const bool end_ram, 
  const unsigned int nsim_states, const bool local_approx, 
  const arma::vec& initial_mode, const unsigned int max_iter, const double conv_tol);

template<class T>
void mcmc::da_mcmc_bsf(T model, const bool end_ram, const unsigned int nsim_states, 
  const bool local_approx, const arma::vec& initial_mode, const unsigned int max_iter, 
  const double conv_tol) {
  
  unsigned int m = model.m;
  unsigned n = model.n;
  
  // get the current values of theta
  arma::vec theta = model.get_theta();
  // compute the log[p(theta)]
  double logprior = log_prior_pdf(theta);
  
  // construct the approximate Gaussian model
  arma::vec mode_estimate = initial_mode;
  ugg_ssm approx_model = model.approximate(mode_estimate, max_iter, conv_tol);
  
  // compute the log-likelihood of the approximate model
  double gaussian_loglik = approx_model.log_likelihood();
  
  // compute unnormalized mode-based correction terms 
  // log[g(y_t | ^alpha_t) / ~g(y_t | ^alpha_t)]
  arma::vec scales = model.scaling_factors(approx_model, mode_estimate);
  double sum_scales = arma::accu(scales);
  // compute the constant term
  double const_term = compute_const_term(model, approx_model); 
  // log-likelihood approximation
  double approx_loglik = gaussian_loglik + const_term + sum_scales;
  
  arma::cube alpha(m, n, nsim_states);
  arma::mat weights(nsim_states, n);
  arma::umat indices(nsim_states, n - 1);
  double loglik = model.bsf_filter(nsim_states, alpha, weights, indices);
  
  filter_smoother(alpha, indices);
  arma::vec w = weights.col(n - 1);
  std::discrete_distribution<unsigned int> sample0(w.begin(), w.end());
  arma::mat sampled_alpha = alpha.slice(sample0(model.engine));
  
  double acceptance_prob = 0.0;
  bool new_value = true;
  unsigned int n_values = 0;
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  for (unsigned int i = 1; i <= n_iter; i++) {
    
    if (i % 16 == 0) {
      Rcpp::checkUserInterrupt();
    }
    
    // sample from standard normal distribution
    arma::vec u(n_par);
    for(unsigned int j = 0; j < n_par; j++) {
      u(j) = normal(model.engine);
    }
    
    // propose new theta
    arma::vec theta_prop = theta + S * u;
    // compute prior
    double logprior_prop = log_prior_pdf(theta_prop);
    
    if (logprior_prop > -arma::datum::inf) {
      // update parameters
      model.set_theta(theta_prop);
      
      if (local_approx) {
        // construct the approximate Gaussian model
        mode_estimate = initial_mode;
        model.approximate(approx_model, mode_estimate, max_iter, conv_tol);
      } else {
        model.approximate(approx_model, mode_estimate, 0, conv_tol);
      }
      // compute unnormalized mode-based correction terms 
      // log[g(y_t | ^alpha_t) / ~g(y_t | ^alpha_t)]
      scales = model.scaling_factors(approx_model, mode_estimate);
      sum_scales = arma::accu(scales);
      // compute the constant term
      const_term = compute_const_term(model, approx_model);
      // compute the log-likelihood of the approximate model
      gaussian_loglik = approx_model.log_likelihood();
      double approx_loglik_prop = gaussian_loglik + const_term + sum_scales;
      
      // stage 1 acceptance probability, used in RAM as well
      acceptance_prob = std::min(1.0, std::exp(approx_loglik_prop - approx_loglik + 
        logprior_prop - logprior));
      
      // initial acceptance
      if (unif(model.engine) < acceptance_prob) {
        
        double loglik_prop = model.bsf_filter(nsim_states, alpha, weights, indices);
        
        //just in case
        if(std::isfinite(loglik_prop)) {
          // delayed acceptance ratio, in log-scale
          double acceptance_prob2 = loglik_prop + approx_loglik - loglik - approx_loglik_prop;
          if (std::log(unif(model.engine)) < acceptance_prob2) {
            if (i > n_burnin) {
              acceptance_rate++;
              n_values++;
            }
            filter_smoother(alpha, indices);
            w = weights.col(n - 1);
            std::discrete_distribution<unsigned int> sample(w.begin(), w.end());
            sampled_alpha = alpha.slice(sample(model.engine));
            approx_loglik = approx_loglik_prop;
            loglik = loglik_prop;
            logprior = logprior_prop;
            theta = theta_prop;
            new_value = true;
          }
        }
      }
    } else acceptance_prob = 0.0;
    
    if (i > n_burnin && n_values % n_thin == 0) {
      //new block
      if (new_value) {
        posterior_storage(n_stored) = logprior + loglik;
        theta_storage.col(n_stored) = theta;
        count_storage(n_stored) = 1;
        alpha_storage.slice(n_stored) = sampled_alpha.t();
        n_stored++;
        new_value = false;
      } else {
        count_storage(n_stored - 1)++;
      }
    }
    
    if (!end_ram || i <= n_burnin) {
      ramcmc::adapt_S(S, u, acceptance_prob, target_acceptance, i, gamma);
    }
  }
  
  trim_storage();
  acceptance_rate /= (n_iter - n_burnin);
}

// run pseudo-marginal MCMC for non-linear Gaussian state space model
// using psi-PF
void mcmc::pm_mcmc_psi_nlg(nlg_ssm model, const bool end_ram, 
  const unsigned int nsim_states, const unsigned int max_iter, 
  const double conv_tol, const unsigned int iekf_iter) {
  
  unsigned int m = model.m;
  unsigned n = model.n;
  // compute the log[p(theta)]
  double logprior = model.log_prior_pdf.eval(model.theta);
  // construct the approximate Gaussian model
  arma::mat mode_estimate(m, n);
  mgg_ssm approx_model0 = model.approximate(mode_estimate, max_iter, conv_tol, iekf_iter);
  if(!arma::is_finite(mode_estimate)) {
    Rcpp::stop("Approximation did not converge. ");
  }
  // compute the log-likelihood of the gaussian model
  double gaussian_loglik = approx_model0.log_likelihood();
  
  arma::cube alpha(m, n, nsim_states);
  arma::mat weights(nsim_states, n);
  arma::umat indices(nsim_states, n - 1);
  
  double loglik = model.psi_filter(approx_model0, gaussian_loglik, 
    nsim_states, alpha, weights, indices);
  
  filter_smoother(alpha, indices);
  arma::vec w = weights.col(n - 1);
  std::discrete_distribution<unsigned int> sample0(w.begin(), w.end());
  arma::mat sampled_alpha = alpha.slice(sample0(model.engine));
  
  double acceptance_prob = 0.0;
  bool new_value = true;
  unsigned int n_values = 0;
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  arma::vec theta = model.theta;
  for (unsigned int i = 1; i <= n_iter; i++) {
    
    if (i % 16 == 0) {
      Rcpp::checkUserInterrupt();
    }
    
    // sample from standard normal distribution
    arma::vec u(n_par);
    for(unsigned int j = 0; j < n_par; j++) {
      u(j) = normal(model.engine);
    }
    
    // propose new theta
    arma::vec theta_prop = theta + S * u;
    // compute prior
    
    double logprior_prop = model.log_prior_pdf.eval(theta_prop);
    
    if (arma::is_finite(logprior_prop) && logprior_prop > -arma::datum::inf) {
      // update parameters
      model.theta = theta_prop;
      
      
      double loglik_prop;
      // construct the approximate Gaussian model
      mgg_ssm approx_model = model.approximate(mode_estimate, max_iter, conv_tol, iekf_iter);
      if(!arma::is_finite(mode_estimate)) {
        gaussian_loglik = -arma::datum::inf;
        loglik_prop = -arma::datum::inf;
      } else {
        // compute the log-likelihood of the approximate model
        gaussian_loglik = approx_model.log_likelihood();
        loglik_prop = model.psi_filter(approx_model, gaussian_loglik, 
          nsim_states, alpha, weights, indices);
      }
      //compute the acceptance probability
      // use explicit min(...) as we need this value later
      // double q = proposal(theta, theta_prop);
      if(arma::is_finite(loglik_prop)) {
        acceptance_prob = std::min(1.0, std::exp(loglik_prop - loglik + 
          logprior_prop - logprior));
      } else {
        acceptance_prob = 0.0;
      }
      
      //accept
      if (unif(model.engine) < acceptance_prob) {
        if (i > n_burnin) {
          acceptance_rate++;
          n_values++;
        }
        filter_smoother(alpha, indices);
        w = weights.col(n - 1);
        std::discrete_distribution<unsigned int> sample(w.begin(), w.end());
        sampled_alpha = alpha.slice(sample(model.engine));
        loglik = loglik_prop;
        logprior = logprior_prop;
        theta = theta_prop;
        new_value = true;
      }
    } else acceptance_prob = 0.0;
    
    if (i > n_burnin && n_values % n_thin == 0) {
      //new block
      if (new_value) {
        posterior_storage(n_stored) = logprior + loglik;
        theta_storage.col(n_stored) = theta;
        count_storage(n_stored) = 1;
        alpha_storage.slice(n_stored) = sampled_alpha.t();
        n_stored++;
        new_value = false;
      } else {
        count_storage(n_stored - 1)++;
      }
    }
    
    if (!end_ram || i <= n_burnin) {
      ramcmc::adapt_S(S, u, acceptance_prob, target_acceptance, i, gamma);
    }
  }
  
  trim_storage();
  acceptance_rate /= (n_iter - n_burnin);
}

// using BSF
void mcmc::pm_mcmc_bsf_nlg(nlg_ssm model, const bool end_ram, 
  const unsigned int nsim_states) {
  
  unsigned int m = model.m;
  unsigned n = model.n;
  // compute the log[p(theta)]
  double logprior = model.log_prior_pdf.eval(model.theta);
  
  
  arma::cube alpha(m, n, nsim_states);
  arma::mat weights(nsim_states, n);
  arma::umat indices(nsim_states, n - 1);
  double loglik = model.bsf_filter(nsim_states, alpha, weights, indices);
  
  filter_smoother(alpha, indices);
  arma::vec w = weights.col(n - 1);
  std::discrete_distribution<unsigned int> sample0(w.begin(), w.end());
  arma::mat sampled_alpha = alpha.slice(sample0(model.engine));
  
  double acceptance_prob = 0.0;
  bool new_value = true;
  unsigned int n_values = 0;
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  arma::vec theta = model.theta;
  
  for (unsigned int i = 1; i <= n_iter; i++) {
    
    if (i % 16 == 0) {
      Rcpp::checkUserInterrupt();
    }
    
    // sample from standard normal distribution
    arma::vec u(n_par);
    for(unsigned int j = 0; j < n_par; j++) {
      u(j) = normal(model.engine);
    }
    
    // propose new theta
    arma::vec theta_prop = theta + S * u;
    // compute prior
    
    double logprior_prop = model.log_prior_pdf.eval(theta_prop);
    
    if (arma::is_finite(logprior_prop) && logprior_prop > -arma::datum::inf) {
      // update parameters
      model.theta = theta_prop;
      
      double loglik_prop = model.bsf_filter(nsim_states, alpha, weights, indices);
      
      //compute the acceptance probability
      // use explicit min(...) as we need this value later
      //  double q = proposal(theta, theta_prop);
      
      acceptance_prob = std::min(1.0, std::exp(loglik_prop - loglik +
        logprior_prop - logprior));
      //accept
      
      if (unif(model.engine) < acceptance_prob) {
        if (i > n_burnin) {
          acceptance_rate++;
          n_values++;
        }
        filter_smoother(alpha, indices);
        w = weights.col(n - 1);
        std::discrete_distribution<unsigned int> sample(w.begin(), w.end());
        sampled_alpha = alpha.slice(sample(model.engine));
        loglik = loglik_prop;
        logprior = logprior_prop;
        theta = theta_prop;
        new_value = true;
      }
    } else acceptance_prob = 0.0;
    
    if (i > n_burnin && n_values % n_thin == 0) {
      //new block
      if (new_value) {
        posterior_storage(n_stored) = logprior + loglik;
        theta_storage.col(n_stored) = theta;
        count_storage(n_stored) = 1;
        alpha_storage.slice(n_stored) = sampled_alpha.t();
        n_stored++;
        new_value = false;
      } else {
        count_storage(n_stored - 1)++;
      }
    }
    
    if (!end_ram || i <= n_burnin) {
      ramcmc::adapt_S(S, u, acceptance_prob, target_acceptance, i, gamma);
    }
  }
  
  trim_storage();
  acceptance_rate /= (n_iter - n_burnin);
}

// run delayed acceptance MCMC for non-linear Gaussian state space model
// using psi-PF
void mcmc::da_mcmc_psi_nlg(nlg_ssm model, const bool end_ram, 
  const unsigned int nsim_states, const unsigned int max_iter,
  const double conv_tol, const unsigned int iekf_iter) {
  
  unsigned int m = model.m;
  unsigned n = model.n;
  // compute the log[p(theta)]
  double logprior = model.log_prior_pdf.eval(model.theta);
  
  // construct the approximate Gaussian model
  arma::mat mode_estimate(m, n);
  mgg_ssm approx_model0 = model.approximate(mode_estimate, max_iter, conv_tol, iekf_iter);
  if(!arma::is_finite(mode_estimate)) {
    Rcpp::stop("Approximation did not converge.");
  }
  // compute the log-likelihood of the approximate model
  double approx_loglik = approx_model0.log_likelihood();
  
  arma::cube alpha(m, n, nsim_states);
  arma::mat weights(nsim_states, n);
  arma::umat indices(nsim_states, n - 1);
  double loglik = model.psi_filter(approx_model0, approx_loglik, 
    nsim_states, alpha, weights, indices);
  approx_loglik += arma::accu(model.scaling_factors(approx_model0, mode_estimate));
  filter_smoother(alpha, indices);
  arma::vec w = weights.col(n - 1);
  std::discrete_distribution<unsigned int> sample0(w.begin(), w.end());
  arma::mat sampled_alpha = alpha.slice(sample0(model.engine));
  
  double acceptance_prob = 0.0;
  bool new_value = true;
  unsigned int n_values = 0;
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  arma::vec theta = model.theta;
  
  for (unsigned int i = 1; i <= n_iter; i++) {
    
    if (i % 16 == 0) {
      Rcpp::checkUserInterrupt();
    }
    
    // sample from standard normal distribution
    arma::vec u(n_par);
    for(unsigned int j = 0; j < n_par; j++) {
      u(j) = normal(model.engine);
    }
    
    // propose new theta
    arma::vec theta_prop = theta + S * u;
    // compute prior
    double logprior_prop = model.log_prior_pdf.eval(theta_prop);
    
    if (logprior_prop > -arma::datum::inf) {
      // update parameters
      model.theta = theta_prop;
      // construct the approximate Gaussian model
      
      mgg_ssm approx_model = model.approximate(mode_estimate, max_iter, conv_tol, iekf_iter);
      if(!arma::is_finite(mode_estimate)) {
        acceptance_prob = 0;
      } else {
        double sum_scales = arma::accu(model.scaling_factors(approx_model, mode_estimate));
        // compute the log-likelihood of the approximate model
        double approx_loglik_prop = approx_model.log_likelihood() + sum_scales;
        // stage 1 acceptance probability, used in RAM as well
        acceptance_prob = std::min(1.0, std::exp(approx_loglik_prop - approx_loglik + 
          logprior_prop - logprior));
        
        // initial acceptance
        if (unif(model.engine) < acceptance_prob) {
          
          double loglik_prop = model.psi_filter(approx_model, approx_loglik_prop - sum_scales, 
            nsim_states, alpha, weights, indices);
          
          //just in case
          if(std::isfinite(loglik_prop)) {
            // delayed acceptance ratio, in log-scale
            double acceptance_prob2 = loglik_prop + approx_loglik - loglik - approx_loglik_prop;
            
            if (std::log(unif(model.engine)) < acceptance_prob2) {
              
              if (i > n_burnin) {
                acceptance_rate++;
                n_values++;
              }
              filter_smoother(alpha, indices);
              w = weights.col(n - 1);
              std::discrete_distribution<unsigned int> sample(w.begin(), w.end());
              sampled_alpha = alpha.slice(sample(model.engine));
              approx_loglik = approx_loglik_prop;
              loglik = loglik_prop;
              logprior = logprior_prop;
              theta = theta_prop;
              new_value = true;
            }
          }
        }
      }
    } else acceptance_prob = 0.0;
    
    if (i > n_burnin && n_values % n_thin == 0) {
      //new block
      if (new_value) {
        posterior_storage(n_stored) = logprior + loglik;
        theta_storage.col(n_stored) = theta;
        count_storage(n_stored) = 1;
        alpha_storage.slice(n_stored) = sampled_alpha.t();
        n_stored++;
        new_value = false;
      } else {
        count_storage(n_stored - 1)++;
      }
    }
    
    if (!end_ram || i <= n_burnin) {
      ramcmc::adapt_S(S, u, acceptance_prob, target_acceptance, i, gamma);
    }
  }
  
  trim_storage();
  acceptance_rate /= (n_iter - n_burnin);
}

// run delayed acceptance MCMC for non-linear Gaussian state space model
// using BSF
void mcmc::da_mcmc_bsf_nlg(nlg_ssm model, const bool end_ram, const unsigned int nsim_states,
  const unsigned int max_iter, const double conv_tol, const unsigned int iekf_iter) {
  
  unsigned int m = model.m;
  unsigned n = model.n;
  // compute the log[p(theta)]
  double logprior = model.log_prior_pdf.eval(model.theta);
  
  // construct the approximate Gaussian model
  arma::mat mode_estimate(m, n);
  mgg_ssm approx_model0 = model.approximate(mode_estimate, max_iter, conv_tol, iekf_iter);
  if(!arma::is_finite(mode_estimate)) {
    Rcpp::stop("Approximation did not converge. ");
  }
  // compute the log-likelihood of the approximate model
  double sum_scales = arma::accu(model.scaling_factors(approx_model0, mode_estimate));
  double approx_loglik = approx_model0.log_likelihood() + sum_scales;
  
  arma::cube alpha(m, n, nsim_states);
  arma::mat weights(nsim_states, n);
  arma::umat indices(nsim_states, n - 1);
  double loglik = model.bsf_filter(nsim_states, alpha, weights, indices);
  
  filter_smoother(alpha, indices);
  arma::vec w = weights.col(n - 1);
  std::discrete_distribution<unsigned int> sample0(w.begin(), w.end());
  arma::mat sampled_alpha = alpha.slice(sample0(model.engine));
  
  double acceptance_prob = 0.0;
  bool new_value = true;
  unsigned int n_values = 0;
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  arma::vec theta = model.theta;
  
  for (unsigned int i = 1; i <= n_iter; i++) {
    
    if (i % 16 == 0) {
      Rcpp::checkUserInterrupt();
    }
    
    // sample from standard normal distribution
    arma::vec u(n_par);
    for(unsigned int j = 0; j < n_par; j++) {
      u(j) = normal(model.engine);
    }
    
    // propose new theta
    arma::vec theta_prop = theta + S * u;
    // compute prior
    double logprior_prop = model.log_prior_pdf.eval(theta_prop);
    
    if (logprior_prop > -arma::datum::inf) {
      // update parameters
      model.theta = theta_prop;
      
      // construct the approximate Gaussian model
      mgg_ssm approx_model = model.approximate(mode_estimate, max_iter, conv_tol, iekf_iter);
      
      if(!arma::is_finite(mode_estimate)) {
        acceptance_prob = 0;
      } else {
        
        // compute the log-likelihood of the approximate model
        double sum_scales = arma::accu(model.scaling_factors(approx_model, mode_estimate));
        
        // compute the log-likelihood of the approximate model
        double approx_loglik_prop = approx_model.log_likelihood() + sum_scales;
        
        // stage 1 acceptance probability, used in RAM as well
        acceptance_prob = std::min(1.0, std::exp(approx_loglik_prop - approx_loglik + 
          logprior_prop - logprior));
        
        // initial acceptance
        if (unif(model.engine) < acceptance_prob) {
          
          
          double loglik_prop = model.bsf_filter(nsim_states, alpha, weights, indices);
          
          //just in case
          if(std::isfinite(loglik_prop)) {
            // delayed acceptance ratio, in log-scale
            double acceptance_prob2 = loglik_prop + approx_loglik - loglik - approx_loglik_prop;
            if (std::log(unif(model.engine)) < acceptance_prob2) {
              
              if (i > n_burnin) {
                acceptance_rate++;
                n_values++;
              }
              filter_smoother(alpha, indices);
              w = weights.col(n - 1);
              std::discrete_distribution<unsigned int> sample(w.begin(), w.end());
              sampled_alpha = alpha.slice(sample(model.engine));
              approx_loglik = approx_loglik_prop;
              loglik = loglik_prop;
              logprior = logprior_prop;
              theta = theta_prop;
              new_value = true;
            }
          }
        }
      }
    } else acceptance_prob = 0.0;
    
    if (i > n_burnin && n_values % n_thin == 0) {
      //new block
      if (new_value) {
        posterior_storage(n_stored) = logprior + loglik;
        theta_storage.col(n_stored) = theta;
        count_storage(n_stored) = 1;
        alpha_storage.slice(n_stored) = sampled_alpha.t();
        n_stored++;
        new_value = false;
      } else {
        count_storage(n_stored - 1)++;
      }
    }
    
    if (!end_ram || i <= n_burnin) {
      ramcmc::adapt_S(S, u, acceptance_prob, target_acceptance, i, gamma);
    }
  }
  
  trim_storage();
  acceptance_rate /= (n_iter - n_burnin);
}
