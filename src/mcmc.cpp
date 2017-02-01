#ifdef _OPENMP
#include <omp.h>
#endif
#include <ramcmc.h>
#include "mcmc.h"
#include "ugg_ssm.h"
#include "ung_ssm.h"
#include "ugg_bsm.h"
#include "ung_bsm.h"

#include "distr_consts.h"
#include "filter_smoother.h"

mcmc::mcmc(const arma::uvec& prior_distributions, const arma::mat& prior_parameters,
  const unsigned int n_iter, const unsigned int n_burnin, 
  const unsigned int n_thin, const unsigned int n, const unsigned int m,
  const double target_acceptance, const double gamma, const arma::mat& S, 
  const bool store_states) :
  prior_distributions(prior_distributions), prior_parameters(prior_parameters),
  n_stored(0), n_iter(n_iter), n_burnin(n_burnin), n_thin(n_thin),
  n_samples(floor((n_iter - n_burnin) / n_thin)), 
  n_par(prior_distributions.n_elem),
  target_acceptance(target_acceptance), gamma(gamma), S(S),
  alpha_storage(arma::cube(n, m, store_states * n_samples)), 
  theta_storage(arma::mat(n_par, n_samples)),
  posterior_storage(arma::vec(n_samples)), 
  count_storage(arma::uvec(n_samples, arma::fill::zeros)),
  acceptance_rate(0.0) {
}

void mcmc::trim_storage() {
  theta_storage.resize(n_par, n_stored);
  posterior_storage.resize(n_stored);
  count_storage.resize(n_stored);
  alpha_storage.resize(alpha_storage.n_rows, alpha_storage.n_cols, n_stored);
}


template void mcmc::state_posterior(ugg_ssm model, unsigned int n_threads);
template void mcmc::state_posterior(ugg_bsm model, unsigned int n_threads);

template <class T>
void mcmc::state_posterior(T model, unsigned int n_threads) {
  
  if(n_threads > 1) {
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads) default(none) \
    shared(n_threads) firstprivate(model)
    {
      model.engine = std::mt19937(omp_get_thread_num() + 1);
      unsigned thread_size = floor(n_stored / n_threads);
      unsigned int start = omp_get_thread_num() * thread_size;
      unsigned int end = (omp_get_thread_num() + 1) * thread_size - 1;
      if(omp_get_thread_num() == (n_threads - 1)) {
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

template void mcmc::state_sampler(ugg_ssm model, const arma::mat& theta, arma::cube& alpha);
template void mcmc::state_sampler(ugg_bsm model, const arma::mat& theta, arma::cube& alpha);

template <class T>
void mcmc::state_sampler(T model, const arma::mat& theta, arma::cube& alpha) {
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
        q += log(2.0) + R::dnorm(theta(i), 0, prior_parameters(0, i), 1);
      }
      break;
    case 2  :
      q += R::dnorm(theta(i), prior_parameters(0, i), prior_parameters(1, i), 1);
      break;
    }
  }
  return q;
}

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
  unsigned int counts = 0;
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
        exp(loglik_prop - loglik + logprior_prop - logprior));
      //accept
      if (unif(model.engine) < acceptance_prob) {
        if (i > n_burnin) acceptance_rate++;
        loglik = loglik_prop;
        logprior = logprior_prop;
        theta = theta_prop;
        counts = 0;
      }
    } else acceptance_prob = 0.0;
    
    if (i > n_burnin) {
      counts++;
      if ((i - n_burnin - 1) % n_thin == 0) {
        if (counts <= n_thin) {
          posterior_storage(n_stored) = logprior + loglik;
          theta_storage.col(n_stored) = theta;
          count_storage(n_stored) = 1;
          n_stored++;
        } else {
          count_storage(n_stored - 1)++;
        }
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
  double approx_loglik = gaussian_loglik + const_term + 
    arma::accu(scales);
  
  arma::cube alpha(m, n, nsim_states);
  arma::mat weights(nsim_states, n);
  arma::umat indices(nsim_states, n - 1);
  double loglik = model.psi_filter(approx_model, approx_loglik, scales, 
    nsim_states, alpha, weights, indices);
  
  filter_smoother(alpha, indices);
  arma::vec w = weights.col(n - 1);
  std::discrete_distribution<> sample0(w.begin(), w.end());
  arma::mat sampled_alpha = alpha.slice(sample0(model.engine));
  
  double acceptance_prob = 0.0;
  unsigned int counts = 0;
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
        // compute unnormalized mode-based correction terms 
        // log[g(y_t | ^alpha_t) / ~g(y_t | ^alpha_t)]
        scales = model.scaling_factors(approx_model, mode_estimate);
        sum_scales = arma::accu(scales);
        // compute the constant term
        const_term = compute_const_term(model, approx_model);
      } else {
        model.approximate(approx_model, mode_estimate, 0, conv_tol);
      }
      // compute the log-likelihood of the approximate model
      gaussian_loglik = approx_model.log_likelihood();
      approx_loglik = gaussian_loglik + const_term + sum_scales;
      
      double loglik_prop = model.psi_filter(approx_model, approx_loglik, scales, 
        nsim_states, alpha, weights, indices);
      
      //compute the acceptance probability
      // use explicit min(...) as we need this value later
      // double q = proposal(theta, theta_prop);
      
      acceptance_prob = std::min(1.0, exp(loglik_prop - loglik + 
        logprior_prop - logprior));
      //accept
      if (unif(model.engine) < acceptance_prob) {
        if (i > n_burnin) acceptance_rate++;
        filter_smoother(alpha, indices);
        w = weights.col(n - 1);
        std::discrete_distribution<> sample(w.begin(), w.end());
        sampled_alpha = alpha.slice(sample(model.engine));
        loglik = loglik_prop;
        logprior = logprior_prop;
        theta = theta_prop;
        counts = 0;
      }
    } else acceptance_prob = 0.0;
    
    if (i > n_burnin) {
      counts++;
      if ((i - n_burnin - 1) % n_thin == 0) {
        if (counts <= n_thin) {
          posterior_storage(n_stored) = logprior + loglik;
          theta_storage.col(n_stored) = theta;
          alpha_storage.slice(n_stored) = sampled_alpha.t();
          count_storage(n_stored) = 1;
          n_stored++;
        } else {
          count_storage(n_stored - 1)++;
        }
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
  std::discrete_distribution<> sample0(w.begin(), w.end());
  arma::mat sampled_alpha = alpha.slice(sample0(model.engine));
  
  double acceptance_prob = 0.0;
  unsigned int counts = 0;
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
      
      acceptance_prob = std::min(1.0, exp(loglik_prop - loglik +
        logprior_prop - logprior));
      //accept
      
      if (unif(model.engine) < acceptance_prob) {
        if (i > n_burnin) acceptance_rate++;
        filter_smoother(alpha, indices);
        w = weights.col(n - 1);
        std::discrete_distribution<> sample(w.begin(), w.end());
        sampled_alpha = alpha.slice(sample(model.engine));
        loglik = loglik_prop;
        logprior = logprior_prop;
        theta = theta_prop;
        counts = 0;
      }
    } else acceptance_prob = 0.0;
    
    if (i > n_burnin) {
      counts++;
      if ((i - n_burnin - 1) % n_thin == 0) {
        if (counts <= n_thin) {
          posterior_storage(n_stored) = logprior + loglik;
          theta_storage.col(n_stored) = theta;
          alpha_storage.slice(n_stored) = sampled_alpha.t();
          count_storage(n_stored) = 1;
          n_stored++;
        } else {
          count_storage(n_stored - 1)++;
        }
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
  double approx_loglik = gaussian_loglik + const_term + 
    arma::accu(scales);
  
  arma::cube alpha(m, n, nsim_states);
  arma::mat weights(nsim_states, n);
  arma::umat indices(nsim_states, n - 1);
  double loglik = model.psi_filter(approx_model, approx_loglik, scales, 
    nsim_states, alpha, weights, indices);
  
  filter_smoother(alpha, indices);
  arma::vec w = weights.col(n - 1);
  std::discrete_distribution<> sample0(w.begin(), w.end());
  arma::mat sampled_alpha = alpha.slice(sample0(model.engine));
  
  double acceptance_prob = 0.0;
  unsigned int counts = 0;
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
        // compute unnormalized mode-based correction terms 
        // log[g(y_t | ^alpha_t) / ~g(y_t | ^alpha_t)]
        scales = model.scaling_factors(approx_model, mode_estimate);
        sum_scales = arma::accu(scales);
        // compute the constant term
        const_term = compute_const_term(model, approx_model);
      } else {
        model.approximate(approx_model, mode_estimate, 0, conv_tol);
      }
      // compute the log-likelihood of the approximate model
      gaussian_loglik = approx_model.log_likelihood();
      double approx_loglik_prop = gaussian_loglik + const_term + sum_scales;
      
      // stage 1 acceptance probability, used in RAM as well
      acceptance_prob = std::min(1.0, exp(approx_loglik_prop - approx_loglik + 
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
            
            if (i > n_burnin) acceptance_rate++;
            filter_smoother(alpha, indices);
            w = weights.col(n - 1);
            std::discrete_distribution<> sample(w.begin(), w.end());
            sampled_alpha = alpha.slice(sample(model.engine));
            approx_loglik = approx_loglik_prop;
            loglik = loglik_prop;
            logprior = logprior_prop;
            theta = theta_prop;
            counts = 0;
          }
        }
      }
    } else acceptance_prob = 0.0;
    
    if (i > n_burnin) {
      counts++;
      if ((i - n_burnin - 1) % n_thin == 0) {
        if (counts <= n_thin) {
          posterior_storage(n_stored) = logprior + loglik;
          theta_storage.col(n_stored) = theta;
          alpha_storage.slice(n_stored) = sampled_alpha.t();
          count_storage(n_stored) = 1;
          n_stored++;
        } else {
          count_storage(n_stored - 1)++;
        }
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
  double approx_loglik = gaussian_loglik + const_term + 
    arma::accu(scales);
  
  arma::cube alpha(m, n, nsim_states);
  arma::mat weights(nsim_states, n);
  arma::umat indices(nsim_states, n - 1);
  double loglik = model.bsf_filter(nsim_states, alpha, weights, indices);
  
  filter_smoother(alpha, indices);
  arma::vec w = weights.col(n - 1);
  std::discrete_distribution<> sample0(w.begin(), w.end());
  arma::mat sampled_alpha = alpha.slice(sample0(model.engine));
  
  double acceptance_prob = 0.0;
  unsigned int counts = 0;
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
        // compute unnormalized mode-based correction terms 
        // log[g(y_t | ^alpha_t) / ~g(y_t | ^alpha_t)]
        scales = model.scaling_factors(approx_model, mode_estimate);
        sum_scales = arma::accu(scales);
        // compute the constant term
        const_term = compute_const_term(model, approx_model);
      } else {
        model.approximate(approx_model, mode_estimate, 0, conv_tol);
      }
      // compute the log-likelihood of the approximate model
      gaussian_loglik = approx_model.log_likelihood();
      double approx_loglik_prop = gaussian_loglik + const_term + sum_scales;
      
      // stage 1 acceptance probability, used in RAM as well
      acceptance_prob = std::min(1.0, exp(approx_loglik_prop - approx_loglik + 
        logprior_prop - logprior));
      
      // initial acceptance
      if (unif(model.engine) < acceptance_prob) {
        
        double loglik_prop = model.bsf_filter(nsim_states, alpha, weights, indices);
        
        //just in case
        if(std::isfinite(loglik_prop)) {
          // delayed acceptance ratio, in log-scale
          double acceptance_prob2 = loglik_prop + approx_loglik - loglik - approx_loglik_prop;
          if (std::log(unif(model.engine)) < acceptance_prob2) {
            if (i > n_burnin) acceptance_rate++;
            filter_smoother(alpha, indices);
            w = weights.col(n - 1);
            std::discrete_distribution<> sample(w.begin(), w.end());
            sampled_alpha = alpha.slice(sample(model.engine));
            approx_loglik = approx_loglik_prop;
            loglik = loglik_prop;
            logprior = logprior_prop;
            theta = theta_prop;
            counts = 0;
          }
        }
      }
    } else acceptance_prob = 0.0;
    
    if (i > n_burnin) {
      counts++;
      if ((i - n_burnin - 1) % n_thin == 0) {
        if (counts <= n_thin) {
          posterior_storage(n_stored) = logprior + loglik;
          theta_storage.col(n_stored) = theta;
          alpha_storage.slice(n_stored) = sampled_alpha.t();
          count_storage(n_stored) = 1;
          n_stored++;
        } else {
          count_storage(n_stored - 1)++;
        }
      }
    }
    
    if (!end_ram || i <= n_burnin) {
      ramcmc::adapt_S(S, u, acceptance_prob, target_acceptance, i, gamma);
    }
  }
  
  trim_storage();
  acceptance_rate /= (n_iter - n_burnin);
}

// 
// // run delayed acceptance pseudo-marginal MCMC for 
// // non-linear and/or non-Gaussian state space model
// // using psi-PF
// template void mcmc::approx_mcmc(ung_ssm model, const bool end_ram, 
//   const unsigned int nsim_states, const bool local_approx, arma::vec& initial_mode, 
//   unsigned int max_iter, const double conv_tol);
// template void mcmc::approx_mcmc(ung_bsm model, const bool end_ram, 
//   const unsigned int nsim_states, const bool local_approx, arma::vec& initial_mode, 
//   unsigned int max_iter, const double conv_tol);
// 
// template<class T>
// void mcmc::approx_mcmc(T model, const bool end_ram, const bool local_approx, 
//   const arma::vec& initial_mode, const unsigned int max_iter, const double conv_tol) {
//   
//   unsigned int m = model.m;
//   unsigned n = model.n;
//   
//   // get the current values of theta
//   arma::vec theta = model.get_theta();
//   // compute the log[p(theta)]
//   double logprior = log_prior_pdf(theta);
//   
//   // construct the approximate Gaussian model
//   arma::vec mode_estimate = initial_mode;
//   ugg_ssm approx_model = model.approximate(mode_estimate, max_iter, conv_tol);
//   
//   // compute the log-likelihood of the approximate model
//   double gaussian_loglik = approx_model.log_likelihood();
//   
//   // compute unnormalized mode-based correction terms 
//   // log[g(y_t | ^alpha_t) / ~g(y_t | ^alpha_t)]
//   arma::vec scales = model.scaling_factors(approx_model, mode_estimate);
//   double sum_scales = arma::accu(scales);
//   // compute the constant term
//   double const_term = compute_const_term(model, approx_model); 
//   // log-likelihood approximation
//   double approx_loglik = gaussian_loglik + const_term + 
//     arma::accu(scales);
//   
//   double acceptance_prob = 0.0;
//   std::normal_distribution<> normal(0.0, 1.0);
//   std::uniform_real_distribution<> unif(0.0, 1.0);
//   for (unsigned int i = 1; i <= n_iter; i++) {
//     
//     if (i % 16 == 0) {
//       Rcpp::checkUserInterrupt();
//     }
//     
//     // sample from standard normal distribution
//     arma::vec u(n_par);
//     for(unsigned int j = 0; j < n_par; j++) {
//       u(j) = normal(model.engine);
//     }
//     
//     // propose new theta
//     arma::vec theta_prop = theta + S * u;
//     // compute prior
//     double logprior_prop = log_prior_pdf(theta_prop);
//     
//     if (logprior_prop > -arma::datum::inf) {
//       // update parameters
//       model.set_theta(theta_prop);
//       
//       if (local_approx) {
//         // construct the approximate Gaussian model
//         mode_estimate = initial_mode;
//         model.approximate(approx_model, mode_estimate, max_iter, conv_tol);
//         // compute unnormalized mode-based correction terms 
//         // log[g(y_t | ^alpha_t) / ~g(y_t | ^alpha_t)]
//         scales = model.scaling_factors(approx_model, mode_estimate);
//         sum_scales = arma::accu(scales);
//         // compute the constant term
//         const_term = compute_const_term(model, approx_model);
//       } else {
//         model.approximate(approx_model, mode_estimate, 0, conv_tol);
//       }
//       // compute the log-likelihood of the approximate model
//       gaussian_loglik = approx_model.log_likelihood();
//       double approx_loglik_prop = gaussian_loglik + const_term + sum_scales;
//       
//       acceptance_prob = std::min(1.0, exp(approx_loglik_prop - approx_loglik + 
//         logprior_prop - logprior));
//       
//       if (unif(model.engine) < acceptance_prob) {
//         
//         approx_loglik = approx_loglik_prop;
//         logprior = logprior_prop;
//         theta = theta_prop;
//         if (i > n_burnin) {
//           acceptance_rate++;
//           if ((i - 1) % n_thin == 0) {
//             posterior_storage(n_stored) = logprior + approx_loglik;
//             theta_storage.col(n_stored) = theta;
//             n_stored++;
//           }
//         }
//       }
//     } else acceptance_prob = 0.0;
//     
//     if ((i > n_burnin) && ((i - 1) % n_thin == 0)) {
//       if (n_stored == 0) {
//         posterior_storage(0) = logprior + approx_loglik;
//         theta_storage.col(0) = theta;
//         n_stored++;
//       }
//       count_storage(n_stored - 1)++;
//     }
//     
//     if (!end_ram || i <= n_burnin) {
//       ramcmc::adapt_S(S, u, acceptance_prob, target_acceptance, i, gamma);
//     }
//   }
//   
//   trim_storage();
//   acceptance_rate /= (n_iter - n_burnin);
// }
// 
// 
// // template void mcmc::pm_mcmc_bsf(ung_ssm model, const bool end_ram, 
// //   unsigned int nsim_states);
// // template void mcmc::pm_mcmc_bsf(ung_bsm model, const bool end_ram, 
// //   unsigned int nsim_states);
// // 
// // template<class T>
// // void mcmc::pm_mcmc_bsf(T model, const bool end_ram, unsigned int nsim_states) {
// //   
// //   unsigned int m = model.m;
// //   unsigned n = model.n;
// //   
// //   // get the current values of theta
// //   arma::vec theta = model.get_theta();
// //   // compute the log[p(theta)]
// //   double logprior = log_prior_pdf(theta);
// //   
// //   arma::cube alpha(m, n, nsim_states);
// //   arma::mat weights(nsim_states, n);
// //   arma::umat indices(nsim_states, n - 1);
// //   double loglik = model.bsf_filter(nsim_states, alpha, weights, indices);
// // 
// //   double acceptance_prob = 0.0;
// //   
// //   filter_smoother(alpha, indices);
// //   arma::vec w0 = weights.col(n - 1);
// //   std::discrete_distribution<> sample0(w0.begin(), w0.end());
// //   update_storage(0, alpha.slice(sample0(model.engine)), 
// //     theta, logprior + loglik);
// //   counts(0)++;
// //   
// //   std::normal_distribution<> normal(0.0, 1.0);
// //   std::uniform_real_distribution<> unif(0.0, 1.0);
// //   
// //   for (unsigned int i = 0; i < n_iter; i++) {
// //     
// //     if (i % 16 == 0) {
// //       Rcpp::checkUserInterrupt();
// //     }
// //     
// //     // sample from standard normal distribution
// //     arma::vec u(n_par);
// //     for(unsigned int j = 0; j < n_par; j++) {
// //       u(j) = normal(model.engine);
// //     }
// //     
// //     // propose new theta
// //     arma::vec theta_prop = theta + S * u;
// //     // compute prior
// //     double logprior_prop = log_prior_pdf(theta_prop);
// //     
// //     if (logprior_prop > -arma::datum::inf) {
// //       // update parameters
// //       model.set_theta(theta_prop);
// //       // construct the approximate Gaussian model
// //       mode_estimate = initial_mode;
// //       ugg_ssm approx_model = model.approximate(mode_estimate, max_iter, conv_tol);
// //       
// //       // compute the log-likelihood of the approximate model
// //       double gaussian_loglik = approx_model.log_likelihood();
// //       // compute unnormalized mode-based correction terms 
// //       // log[g(y_t | ^alpha_t) / ~g(y_t | ^alpha_t)]
// //       arma::vec scales = model.scaling_factors(approx_model, mode_estimate);
// //       // compute the constant term
// //       double const_term = compute_const_term(model, approx_model); 
// //       double approx_loglik = gaussian_loglik + const_term + arma::accu(scales);
// //       
// //       arma::cube alpha_prop(m, n, nsim_states);
// //       arma::mat weights_prop(nsim_states, n);
// //       arma::umat indices_prop(nsim_states, n - 1);
// //       double loglik_prop = model.psi_filter(approx_model, approx_loglik, scales, 
// //         nsim_states, alpha_prop, weights_prop, indices_prop);
// //       
// //       //compute the acceptance probability
// //       // use explicit min(...) as we need this value later
// //       double q = proposal(theta, theta_prop);
// //       acceptance_prob = std::min(1.0, exp(loglik_prop - loglik + 
// //         logprior_prop - logprior + q));
// //       //accept
// //       if (unif(model.engine) < acceptance_prob) {
// //         loglik = loglik_prop;
// //         logprior = logprior_prop;
// //         theta = theta_prop;
// //         alpha = alpha_prop;
// //         indices = indices_prop;
// //         weights = weights_prop;
// //         filter_smoother(alpha, indices);
// //         arma::vec w = weights.col(n - 1);
// //         std::discrete_distribution<> sample(w.begin(), w.end());
// //         
// //         if (i >= n_burnin) {
// //           acceptance_rate++;
// //           n_stored++;
// //           update_storage(n_stored, alpha.slice(sample(model.engine)), 
// //             theta, logprior + loglik);
// //         } else {
// //           update_storage(0, alpha.slice(sample(model.engine)), 
// //             theta, logprior + loglik);
// //           counts(0) = 0;
// //         }
// //       }
// //     } acceptance_prob = 0.0;
// //     
// //     counts(n_stored)++;
// //     
// //     if (!end_ram || i < n_burnin) {
// //       ramcmc::adapt_S(S, u, acceptance_prob, target_acceptance, i + 1, gamma);
// //     }
// //     
// //   }
// //   
// //   trim_storage();
// //   
// //   acceptance_rate /= (n_iter - n_burnin);
// // }
// // 
// // // template<class T>
// // // double mcmc::pm_mcmc_bsf(T model, const bool end_ram, unsigned int nsim_states) {
// // //   
// // //   double acceptance_rate = 0.0;
// // //   
// // //   arma::vec theta = model.get_theta();
// // //   double logprior = model.log_prior_pdf(theta, prior_distributions, prior_parameters);
// // //   
// // //   double loglik = model.bsf_filter(0, nsim_states, alpha, weights, indices);
// // //   model.backtrack(alpha, indices);
// // //   unsigned int n_stored = 0;
// // //   double acceptance_prob = 0.0;
// // //   
// // //   std::normal_distribution<> normal(0.0, 1.0);
// // //   std::uniform_real_distribution<> unif(0.0, 1.0);
// // //   
// // //   for (unsigned int i = 0; i < n_iter; i++) {
// // //     
// // //     if (i % 16 == 0) {
// // //       Rcpp::checkUserInterrupt();
// // //     }
// // //     
// // //     // sample from standard normal distribution
// // //     arma::vec u(n_par);
// // //     for(unsigned int j = 0; j < n_par; j++) {
// // //       u(j) = normal(model.engine);
// // //     }
// // //     
// // //     // propose new theta
// // //     arma::vec theta_prop = theta + S * u;
// // //     // compute prior
// // //     double logprior_prop = 
// // //       model.log_prior_pdf(theta_prop, prior_distributions, prior_parameters);
// // //     
// // //     if (logprior_prop > -arma::datum::inf) {
// // //       // update parameters
// // //       model.set_theta(theta_prop);
// // //       // compute log-likelihood with proposed theta
// // //       double loglik_prop = model.bsf_filter(0, nsim_states, alpha_prop, weights, indices);
// // //       //compute the acceptance probability
// // //       // use explicit min(...) as we need this value later
// // //       double q = model.proposal(theta, theta_prop);
// // //       acceptance_prob = std::min(1.0, exp(loglik_prop - loglik + logprior_prop - logprior + q));
// // //       //accept
// // //       if (unif(model.engine) < acceptance_prob) {
// // //         if (i >= n_burnin) {
// // //           acceptance_rate++;
// // //         }
// // //         loglik = loglik_prop;
// // //         logprior = logprior_prop;
// // //         theta = theta_prop;
// // //       }
// // //     } else acceptance_prob = 0.0;
// // //     
// // //     //store
// // //     if ((i >= n_burnin) && ((i - 1) % n_thin == 0) && n_stored < n_samples) {
// // //       update_storage(n_stored, theta, logprior + loglik);
// // //       n_stored++;
// // //     }
// // //     
// // //     if (!end_ram || i < n_burnin) {
// // //       ramcmc::adapt_S(S, u, acceptance_prob, target_acceptance, i + 1, gamma);
// // //     }
// // //     
// // //   }
// // //   
// // //   return acceptance_rate / (n_iter - n_burnin);
// // //   
// // // }
// // // 
