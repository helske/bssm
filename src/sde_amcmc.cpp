#ifdef _OPENMP
#include <omp.h>
#endif
#include <sitmo.h>
#include <ramcmc.h>
#include "sde_amcmc.h"
#include "sde_ssm.h"

#include "filter_smoother.h"

sde_amcmc::sde_amcmc(const unsigned int n_iter, 
  const unsigned int n_burnin, const unsigned int n_thin, const unsigned int n, 
  const double target_acceptance, const double gamma, 
  const arma::mat& S) :
  mcmc(arma::uvec(S.n_cols), arma::mat(1,1), n_iter, n_burnin, n_thin, n, 1,
    target_acceptance, gamma, S, true),
    weight_storage(arma::vec(n_samples, arma::fill::zeros)),
    approx_loglik_storage(arma::vec(n_samples)),
    prior_storage(arma::vec(n_samples)),
    iter_storage(arma::uvec(n_samples)){
}

void sde_amcmc::trim_storage() {
  theta_storage.resize(n_par, n_stored);
  posterior_storage.resize(n_stored);
  count_storage.resize(n_stored);
  alpha_storage.resize(alpha_storage.n_rows, alpha_storage.n_cols, n_stored);
  weight_storage.resize(n_stored);
  approx_loglik_storage.resize(n_stored);
  prior_storage.resize(n_stored);
  iter_storage.resize(n_stored);
}

// run approximate MCMC for
// non-linear Gaussian state space model

void sde_amcmc::approx_mcmc(sde_ssm model, const bool end_ram, 
  const unsigned int nsim_states, const unsigned int L) {
  
  unsigned int m = 1;
  unsigned n = model.n;
  // compute the log[p(theta)]
  double logprior = model.log_prior_pdf(model.theta);
  
  
  arma::cube alpha(m, n, nsim_states);
  arma::mat weights(nsim_states, n);
  arma::umat indices(nsim_states, n - 1);
  double loglik = model.bsf_filter(nsim_states, L, alpha, weights, indices);
  
  double acceptance_prob = 0.0;
  bool new_value = true;
  unsigned int n_values = 0;
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  arma::vec theta = model.theta;
  
  for (unsigned int i = 1; i <= n_iter; i++) {
    if (i % 4 == 0) {
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
    double logprior_prop = model.log_prior_pdf(theta_prop);
    
    if (arma::is_finite(logprior_prop)) {
      // update parameters
      model.theta = theta_prop;
      
      double loglik_prop = model.bsf_filter(nsim_states, L, alpha, weights, indices);
      
      //compute the acceptance probability
      // use explicit min(...) as we need this value later
      acceptance_prob = std::min(1.0, std::exp(loglik_prop - loglik +
        logprior_prop - logprior));
      
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
        approx_loglik_storage(n_stored) = loglik;
        prior_storage(n_stored) = logprior;
        theta_storage.col(n_stored) = theta;
        count_storage(n_stored) = 1;
        iter_storage(n_stored) = i; //count 0 as well
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

void sde_amcmc::is_correction_bsf(sde_ssm model, const unsigned int nsim_states, 
  const unsigned int L_c, const unsigned int L_f, const bool coupled,
  const bool const_sim, const unsigned int n_threads) {
  
  if(coupled) {
    model.coarse_engine = sitmo::prng_engine(model.seed);
  }
  
  if(n_threads > 1) {
    
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads) default(none) firstprivate(model)
{
  model.engine = sitmo::prng_engine(omp_get_thread_num() + 1);
  
  unsigned thread_size = std::floor(static_cast <double> (n_stored) / n_threads);
  unsigned int start = omp_get_thread_num() * thread_size;
  unsigned int end = (omp_get_thread_num() + 1) * thread_size - 1;
  if(omp_get_thread_num() == static_cast<int>(n_threads - 1)) {
    end = n_stored - 1;
  }
  if(coupled) {
    // fast forward the RNG
    model.coarse_engine.discard(iter_storage(start) * model.n * nsim_states * std::pow(2, L_c));
  } else {
    model.coarse_engine = sitmo::prng_engine(omp_get_thread_num() + n_threads + 1);
  }
  arma::mat theta_piece = theta_storage(arma::span::all, arma::span(start, end));
  arma::cube alpha_piece(model.n, 1, end - start + 1);
  arma::vec weights_piece(end - start + 1);
  arma::vec approx_loglik_piece = approx_loglik_storage.subvec(start, end);
  if (const_sim) {
    if(coupled) {
      arma::uvec iter_piece = iter_storage(arma::span(start, end));
      state_sampler_cbsf_is2(model, nsim_states, L_c, L_f, approx_loglik_piece, 
        theta_piece, alpha_piece, weights_piece, iter_piece);  
    } else {
      state_sampler_bsf_is2(model, nsim_states, L_f, approx_loglik_piece, 
        theta_piece, alpha_piece, weights_piece);
    }
  } else {
    arma::uvec count_piece = count_storage(arma::span(start, end));
    if(coupled) {
      arma::uvec iter_piece = iter_storage(arma::span(start, end));
      state_sampler_cbsf_is1(model, nsim_states, L_c, L_f, approx_loglik_piece, 
        theta_piece, alpha_piece, weights_piece, count_piece, iter_piece);  
    } else {
      state_sampler_bsf_is1(model, nsim_states, L_f, approx_loglik_piece, 
        theta_piece, alpha_piece, weights_piece, count_piece);
    }
  }
  alpha_storage.slices(start, end) = alpha_piece;
  weight_storage.subvec(start, end) = weights_piece;
}
#else
if (const_sim) {
  if(coupled) {
    state_sampler_cbsf_is2(model, nsim_states, L_c, L_f, 
      approx_loglik_storage, theta_storage, alpha_storage, weight_storage, iter_storage);
  } else {
    state_sampler_bsf_is2(model, nsim_states, L_f, approx_loglik_storage, 
      theta_storage, alpha_storage, weight_storage);
  }
} else {
  if(coupled) {
    state_sampler_cbsf_is1(model, nsim_states, L_c, L_f, approx_loglik_storage, 
      theta_storage, alpha_storage, weight_storage, count_storage, iter_storage);
  } else {
    state_sampler_bsf_is1(model, nsim_states, L_f, approx_loglik_storage, 
      theta_storage, alpha_storage, weight_storage, count_storage);
  }
}

#endif
  } else {
    if (const_sim) {
      if(coupled) {
        state_sampler_cbsf_is2(model, nsim_states, L_c, L_f, 
          approx_loglik_storage, theta_storage, alpha_storage, weight_storage, iter_storage);
      } else {
        state_sampler_bsf_is2(model, nsim_states, L_f, approx_loglik_storage, 
          theta_storage, alpha_storage, weight_storage);
      }
    } else {
      if(coupled) {
        state_sampler_cbsf_is1(model, nsim_states, L_c, L_f, approx_loglik_storage, 
          theta_storage, alpha_storage, weight_storage, count_storage, iter_storage);
      } else {
        state_sampler_bsf_is1(model, nsim_states, L_f, approx_loglik_storage, 
          theta_storage, alpha_storage, weight_storage, count_storage);
      }
    }
  }
  posterior_storage = prior_storage + approx_loglik_storage + arma::log(weight_storage);
}

void sde_amcmc::state_sampler_bsf_is2(sde_ssm& model, 
  const unsigned int nsim_states, const unsigned int L_f,
  const arma::vec& approx_loglik_storage, const arma::mat& theta,
  arma::cube& alpha, arma::vec& weights) {
  
  for (unsigned int i = 0; i < theta.n_cols; i++) {
    
    model.theta = theta.col(i);
    
    arma::cube alpha_i(1, model.n, nsim_states);
    arma::mat weights_i(nsim_states, model.n);
    arma::umat indices(nsim_states, model.n - 1);
    double loglik = model.bsf_filter(nsim_states, L_f, alpha_i, weights_i, indices);
    if(arma::is_finite(loglik)) {
      weights(i) = std::exp(loglik - approx_loglik_storage(i));
      
      filter_smoother(alpha_i, indices);
      arma::vec w = weights_i.col(model.n - 1);
      std::discrete_distribution<unsigned int> sample(w.begin(), w.end());
      alpha.slice(i) = alpha_i.slice(sample(model.engine)).t();
    } else {
      weights(i) = 0.0;
      alpha.slice(i).zeros();
    }
  }
}


void sde_amcmc::state_sampler_bsf_is1(sde_ssm& model, 
  const unsigned int nsim_states, const unsigned int L_f,
  const arma::vec& approx_loglik_storage, const arma::mat& theta,
  arma::cube& alpha, arma::vec& weights, const arma::uvec& counts) {
  
  for (unsigned int i = 0; i < theta.n_cols; i++) {
    
    model.theta = theta.col(i);
    
    unsigned int m = nsim_states * counts(i);
    arma::cube alpha_i(1, model.n, m);
    arma::mat weights_i(m, model.n);
    arma::umat indices(m, model.n - 1);
    double loglik = model.bsf_filter(m, L_f, alpha_i, weights_i, indices);
    if(arma::is_finite(loglik)) {
      weights(i) = std::exp(loglik - approx_loglik_storage(i));
      filter_smoother(alpha_i, indices);
      arma::vec w = weights_i.col(model.n - 1);
      std::discrete_distribution<unsigned int> sample(w.begin(), w.end());
      alpha.slice(i) = alpha_i.slice(sample(model.engine)).t();
    } else {
      weights(i) = 0.0;
      alpha.slice(i).zeros();
    }
  }
}

void sde_amcmc::state_sampler_cbsf_is2(sde_ssm& model, 
  const unsigned int nsim_states, const unsigned int L_c, const unsigned int L_f,
  const arma::vec& approx_loglik_storage, const arma::mat& theta,
  arma::cube& alpha, arma::vec& weights, const arma::uvec& iter) {
  
  
  arma::uvec iter_diff(theta.n_cols);
  iter_diff(0) = iter(0) - 1;
  iter_diff.subvec(1, iter_diff.n_elem - 1) = arma::diff(iter) - 1;
  int d = model.n * nsim_states * std::pow(2, L_c);
  for (unsigned int i = 0; i < theta.n_cols; i++) {
    model.theta = theta.col(i);
    arma::cube alpha_i(1, model.n, nsim_states);
    
    arma::mat weights_i(nsim_states, model.n);
    arma::umat indices(nsim_states, model.n - 1);
    model.coarse_engine.discard(iter_diff(i) * d);
    double loglik = model.coupled_bsf_filter(nsim_states, L_c, L_f, alpha_i, weights_i, indices);
    
    if(arma::is_finite(loglik)) {
      weights(i) = std::exp(loglik - approx_loglik_storage(i));
      filter_smoother(alpha_i, indices);
      arma::vec w = weights_i.col(model.n - 1);
      std::discrete_distribution<unsigned int> sample(w.begin(), w.end());
      alpha.slice(i) = alpha_i.slice(sample(model.engine)).t();
    } else {
      weights(i) = 0.0;
      alpha.slice(i).zeros();
    }
  }
}


void sde_amcmc::state_sampler_cbsf_is1(sde_ssm& model, 
  const unsigned int nsim_states, const unsigned int L_c, const unsigned int L_f,
  const arma::vec& approx_loglik_storage, const arma::mat& theta,
  arma::cube& alpha, arma::vec& weights, const arma::uvec& counts, const arma::uvec& iter) {
  
  arma::uvec iter_diff(theta.n_cols);
  iter_diff(0) = iter(0) - 1;
  iter_diff.subvec(1, iter_diff.n_elem - 1) = arma::diff(iter) - 1;
  
  int d = model.n * std::pow(2, L_c);
  
  for (unsigned int i = 0; i < theta.n_cols; i++) {
    
    model.theta = theta.col(i);
    
    unsigned int m = nsim_states * counts(i);
    arma::cube alpha_i(1, model.n, m);
    arma::mat weights_i(m, model.n);
    arma::umat indices(m, model.n - 1);
    model.coarse_engine.discard(iter_diff(i) * d * m);
    double loglik = model.coupled_bsf_filter(nsim_states, L_c, L_f, alpha_i, weights_i, indices);
    if(arma::is_finite(loglik)) {
      weights(i) = std::exp(loglik - approx_loglik_storage(i));
      filter_smoother(alpha_i, indices);
      arma::vec w = weights_i.col(model.n - 1);
      std::discrete_distribution<unsigned int> sample(w.begin(), w.end());
      alpha.slice(i) = alpha_i.slice(sample(model.engine)).t();
    } else {
      weights(i) = 0.0;
      alpha.slice(i).zeros();
    }
  }
}
