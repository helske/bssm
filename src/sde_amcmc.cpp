#ifdef _OPENMP
#include <omp.h>
#endif
#include <sitmo.h>
#include <ramcmc.h>
#include "sde_amcmc.h"
#include "sde_ssm.h"
#include "rep_mat.h"

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
  // test if already trimmed
  if(posterior_storage.n_elem != n_stored) {
    theta_storage.resize(n_par, n_stored);
    posterior_storage.resize(n_stored);
    count_storage.resize(n_stored);
    alpha_storage.resize(alpha_storage.n_rows, alpha_storage.n_cols, n_stored);
    weight_storage.resize(n_stored);
    approx_loglik_storage.resize(n_stored);
    prior_storage.resize(n_stored);
    iter_storage.resize(n_stored);
  }
}

void sde_amcmc::expand() {
  //trim extras first just in case
  trim_storage();
  n_stored = arma::accu(count_storage);
  
  arma::mat expanded_theta = rep_mat(theta_storage, count_storage);
  theta_storage.set_size(n_par, n_stored);
  theta_storage = expanded_theta;
  
  arma::vec expanded_posterior = rep_vec(posterior_storage, count_storage);
  posterior_storage.set_size(n_stored);
  posterior_storage = expanded_posterior;
  
  arma::cube expanded_alpha = rep_cube(alpha_storage, count_storage);
  alpha_storage.set_size(alpha_storage.n_rows, alpha_storage.n_cols, n_stored);
  alpha_storage = expanded_alpha;
  
  arma::vec expanded_weight = rep_vec(weight_storage, count_storage);
  weight_storage.set_size(n_stored);
  weight_storage = expanded_weight;
  
  arma::vec expanded_approx_loglik = rep_vec(approx_loglik_storage, count_storage);
  approx_loglik_storage.set_size(n_stored);
  approx_loglik_storage = expanded_approx_loglik;
  
  arma::vec expanded_prior = rep_vec(prior_storage, count_storage);
  prior_storage.set_size(n_stored);
  prior_storage = expanded_prior;
  
  arma::uvec expanded_iter = rep_uvec(iter_storage, count_storage);
  iter_storage.set_size(n_stored);
  iter_storage = expanded_iter;
  
  count_storage.resize(n_stored);
  count_storage.ones();
  
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
  const unsigned int is_type, const unsigned int n_threads) {
  
  // no coupling at the moment!
  if(coupled) Rcpp::stop("Coupling not supported yet!.");
  
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads) default(none) firstprivate(model) 
  {
  
  model.engine = sitmo::prng_engine(omp_get_thread_num() + 1);
  model.coarse_engine = sitmo::prng_engine(omp_get_thread_num() + n_threads + 1);
  
#pragma omp for schedule(dynamic)
  for (unsigned int i = 0; i < n_stored; i++) {
    model.theta = theta_storage.col(i);
    unsigned int nsim = nsim_states;
    if (is_type == 1) {
      nsim *= count_storage(i);
    }
    arma::cube alpha_i(1, model.n, nsim);
    arma::mat weights_i(nsim, model.n);
    arma::umat indices(nsim, model.n - 1);
    double loglik = model.bsf_filter(nsim, L_f, alpha_i, weights_i, indices);
    if(arma::is_finite(loglik)) {
      weight_storage(i) = std::exp(loglik - approx_loglik_storage(i));
      
      filter_smoother(alpha_i, indices);
      arma::vec w = weights_i.col(model.n - 1);
      std::discrete_distribution<unsigned int> sample(w.begin(), w.end());
      alpha_storage.slice(i) = alpha_i.slice(sample(model.engine)).t();
    } else {
      weight_storage(i) = 0.0;
      alpha_storage.slice(i).zeros();
    }
  }
}
#else
for (unsigned int i = 0; i < n_stored; i++) {
  model.theta = theta_storage.col(i);
  unsigned int nsim = nsim_states;
  if (is_type == 1) {
    nsim *= count_storage(i);
  }
  arma::cube alpha_i(1, model.n, nsim);
  arma::mat weights_i(nsim, model.n);
  arma::umat indices(nsim, model.n - 1);
  double loglik = model.bsf_filter(nsim, L_f, alpha_i, weights_i, indices);
  if(arma::is_finite(loglik)) {
    weight_storage(i) = std::exp(loglik - approx_loglik_storage(i));
    
    filter_smoother(alpha_i, indices);
    arma::vec w = weights_i.col(model.n - 1);
    std::discrete_distribution<unsigned int> sample(w.begin(), w.end());
    alpha_storage.slice(i) = alpha_i.slice(sample(model.engine)).t();
  } else {
    weight_storage(i) = 0.0;
    alpha_storage.slice(i).zeros();
  }
}
#endif
posterior_storage = prior_storage + approx_loglik_storage + arma::log(weight_storage);
}
