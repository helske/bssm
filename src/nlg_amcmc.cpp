#ifdef _OPENMP
#include <omp.h>
#endif
#include <ramcmc.h>
#include "nlg_amcmc.h"
#include "nlg_ssm.h"

#include "filter_smoother.h"

nlg_amcmc::nlg_amcmc(const arma::uvec& prior_distributions, 
  const arma::mat& prior_parameters, const unsigned int n_iter, 
  const unsigned int n_burnin, const unsigned int n_thin, const unsigned int n, 
  const unsigned int m, const double target_acceptance, const double gamma, 
  const arma::mat& S) :
  mcmc(prior_distributions, prior_parameters, n_iter, n_burnin, n_thin, n, m,
    target_acceptance, gamma, S, true),
    weight_storage(arma::vec(n_samples, arma::fill::zeros)),
    approx_loglik_storage(arma::vec(n_samples)),
    prior_storage(arma::vec(n_samples)) {
}

void nlg_amcmc::trim_storage() {
  theta_storage.resize(n_par, n_stored);
  posterior_storage.resize(n_stored);
  count_storage.resize(n_stored);
  alpha_storage.resize(alpha_storage.n_rows, alpha_storage.n_cols, n_stored);
  weight_storage.resize(n_stored);
  approx_loglik_storage.resize(n_stored);
  prior_storage.resize(n_stored);
}

// run approximate MCMC for
// non-linear Gaussian state space model

void nlg_amcmc::approx_mcmc(nlg_ssm model, const bool end_ram) {
  
  unsigned int m = model.m;
  unsigned n = model.n;
  
  double logprior = model.log_prior_pdf.eval(model.theta);
  
  double loglik = model.ekf_loglik();
  
  double acceptance_prob = 0.0;
  unsigned int counts = 0;
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
      
      arma::mat alphahat_prop(m, n);
      double loglik_prop = model.ekf_loglik();
      
      if(arma::is_finite(loglik_prop)) {
        acceptance_prob = std::min(1.0, exp(loglik_prop - loglik +
          logprior_prop - logprior));
      } else {
        acceptance_prob = 0.0; 
      }
      
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
          approx_loglik_storage(n_stored) = loglik;
          prior_storage(n_stored) = logprior;
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

void nlg_amcmc::is_correction_bsf(nlg_ssm model, const unsigned int nsim_states, 
  const bool const_sim, const unsigned int n_threads) {
  
  if(n_threads > 1) {
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads) default(none) firstprivate(model)
{
  model.engine = std::mt19937(omp_get_thread_num() + 1);
  unsigned thread_size = floor(n_stored / n_threads);
  unsigned int start = omp_get_thread_num() * thread_size;
  unsigned int end = (omp_get_thread_num() + 1) * thread_size - 1;
  if(omp_get_thread_num() == (n_threads - 1)) {
    end = n_stored - 1;
  }
  
  arma::mat theta_piece = theta_storage(arma::span::all, arma::span(start, end));
  arma::cube alpha_piece(model.n, model.m, thread_size);
  arma::vec weights_piece(thread_size);
  arma::vec approx_loglik_piece = approx_loglik_storage.subvec(start, end);
  if (const_sim) {
    state_sampler_bsf_is2(model, nsim_states, theta_piece, approx_loglik_piece, 
      alpha_piece, weights_piece);
  } else {
    arma::uvec count_piece = count_storage(arma::span(start, end));
    state_sampler_bsf_is1(model, nsim_states, theta_piece, approx_loglik_piece, 
      alpha_piece, weights_piece, count_piece);
  }
  alpha_storage.slices(start, end) = alpha_piece;
  weight_storage.subvec(start, end) = weights_piece;
}
#else
    if (const_sim) {
      state_sampler_bsf_is2(model, nsim_states, approx_loglik_storage, theta_storage, 
        alpha_storage, weight_storage);
    } else {
      state_sampler_bsf_is1(model, nsim_states, approx_loglik_storage, theta_storage, 
        alpha_storage, weight_storage, count_storage);
    }
#endif
  } else {
    if (const_sim) {
      state_sampler_bsf_is2(model, nsim_states, approx_loglik_storage, theta_storage, 
        alpha_storage, weight_storage);
    } else {
      state_sampler_bsf_is1(model, nsim_states, approx_loglik_storage, theta_storage, 
        alpha_storage, weight_storage, count_storage);
    }
  }
  posterior_storage = prior_storage + log(weight_storage);
}

void nlg_amcmc::state_sampler_bsf_is2(nlg_ssm model, const unsigned int nsim_states, 
  const arma::vec& approx_loglik_storage, const arma::mat& theta,
  arma::cube& alpha, arma::vec& weights) {
  
  for (unsigned int i = 0; i < theta.n_cols; i++) {
    
    model.theta = theta.col(i);
    
    arma::cube alpha_i(model.m, model.n, nsim_states);
    arma::mat weights_i(nsim_states, model.n);
    arma::umat indices(nsim_states, model.n - 1);
    double loglik = model.bsf_filter(nsim_states, alpha_i, weights_i, indices);
    if(arma::is_finite(loglik)) {
      weights(i) = exp(loglik - approx_loglik_storage(i));
      filter_smoother(alpha_i, indices);
      arma::vec w = weights_i.col(model.n - 1);
      std::discrete_distribution<> sample(w.begin(), w.end());
      alpha.slice(i) = alpha_i.slice(sample(model.engine)).t();
    } else {
      weights(i) = 0.0;
      alpha.slice(i).zeros();
    }
  }
}


void nlg_amcmc::state_sampler_bsf_is1(nlg_ssm model, const unsigned int nsim_states, 
  const arma::vec& approx_loglik_storage, const arma::mat& theta,
  arma::cube& alpha, arma::vec& weights, const arma::uvec& counts) {
  
  for (unsigned int i = 0; i < theta.n_cols; i++) {
    
    model.theta = theta.col(i);
    
    unsigned int m = nsim_states * counts(i);
    arma::cube alpha_i(model.m, model.n, m);
    arma::mat weights_i(m, model.n);
    arma::umat indices(m, model.n - 1);
    double loglik = model.bsf_filter(m, alpha_i, weights_i, indices);
    if(arma::is_finite(loglik)) {
      weights(i) = exp(loglik - approx_loglik_storage(i));
      filter_smoother(alpha_i, indices);
      arma::vec w = weights_i.col(model.n - 1);
      std::discrete_distribution<> sample(w.begin(), w.end());
      alpha.slice(i) = alpha_i.slice(sample(model.engine)).t();
    } else {
      weights(i) = 0.0;
      alpha.slice(i).zeros();
    }
  }
}
