#ifdef _OPENMP
#include <omp.h>
#endif
#include <ramcmc.h>
#include "ung_amcmc.h"
#include "ugg_ssm.h"
#include "ung_ssm.h"
#include "ugg_bsm.h"
#include "ung_bsm.h"
#include "ung_svm.h"
#include "ung_ar1.h"

#include "rep_mat.h"
#include "distr_consts.h"
#include "filter_smoother.h"
#include "summary.h"

ung_amcmc::ung_amcmc(const unsigned int n_iter, 
  const unsigned int n_burnin, const unsigned int n_thin, const unsigned int n, 
  const unsigned int m, const double target_acceptance, const double gamma, 
  const arma::mat& S, const unsigned int output_type, const bool store_modes) :
  mcmc(n_iter, n_burnin, n_thin, n, m,
    target_acceptance, gamma, S, output_type),
    weight_storage(arma::vec(n_samples, arma::fill::zeros)),
    y_storage(arma::mat(n, n_samples * store_modes)), 
    H_storage(arma::mat(n, n_samples * store_modes)),
    scales_storage(arma::mat(n, n_samples * store_modes)),
    approx_loglik_storage(arma::vec(n_samples)), 
    prior_storage(arma::vec(n_samples)), store_modes(store_modes) {
}

void ung_amcmc::trim_storage() {
  theta_storage.resize(n_par, n_stored);
  posterior_storage.resize(n_stored);
  count_storage.resize(n_stored);
  if (output_type == 1) {
    alpha_storage.resize(alpha_storage.n_rows, alpha_storage.n_cols, n_stored);
  }
  approx_loglik_storage.resize(n_stored);
  weight_storage.resize(n_stored);
  prior_storage.resize(n_stored);
  if (store_modes) {
    scales_storage.resize(y_storage.n_rows, n_stored);
    y_storage.resize(y_storage.n_rows, n_stored);
    H_storage.resize(H_storage.n_rows, n_stored);
  }
}

void ung_amcmc::expand() {
  //trim extras first just in case
  trim_storage();
  n_stored = arma::accu(count_storage);
  
  arma::mat expanded_theta = rep_mat(theta_storage, count_storage);
  theta_storage.set_size(n_par, n_stored);
  theta_storage = expanded_theta;
  
  arma::vec expanded_posterior = rep_vec(posterior_storage, count_storage);
  posterior_storage.set_size(n_stored);
  posterior_storage = expanded_posterior;
  
  arma::vec expanded_weight = rep_vec(weight_storage, count_storage);
  weight_storage.set_size(n_stored);
  weight_storage = expanded_weight;
  
  arma::vec expanded_prior = rep_vec(prior_storage, count_storage);
  prior_storage.set_size(n_stored);
  prior_storage = expanded_prior;
  
  count_storage.resize(n_stored);
  count_storage.ones();
  
  arma::vec expanded_approx_loglik = rep_vec(approx_loglik_storage, count_storage);
  approx_loglik_storage.set_size(n_stored);
  approx_loglik_storage = expanded_approx_loglik;
  
  if (output_type == 1) {
    arma::cube expanded_alpha = rep_cube(alpha_storage, count_storage);
    alpha_storage.set_size(alpha_storage.n_rows, alpha_storage.n_cols, n_stored);
    alpha_storage = expanded_alpha;
  }
  
  if (store_modes) {
  arma::mat expanded_scales = rep_mat(scales_storage, count_storage);
  scales_storage.set_size(scales_storage.n_rows, n_stored);
  scales_storage = expanded_scales;
  
  arma::mat expanded_y = rep_mat(y_storage, count_storage);
  y_storage.set_size(y_storage.n_rows, n_stored);
  y_storage = expanded_y;
  
  arma::mat expanded_H = rep_mat(H_storage, count_storage);
  H_storage.set_size(H_storage.n_rows, n_stored);
  H_storage = expanded_H;
  
  }
}

// run approximate MCMC for
// non-linear and/or non-Gaussian state space model with linear-Gaussian states
template void ung_amcmc::approx_mcmc(ung_ssm model, const bool end_ram,
  const bool local_approx, const arma::vec& initial_mode,
  const unsigned int max_iter, const double conv_tol);
template void ung_amcmc::approx_mcmc(ung_bsm model, const bool end_ram,
  const bool local_approx, const arma::vec& initial_mode,
  const unsigned int max_iter, const double conv_tol);
template void ung_amcmc::approx_mcmc(ung_svm model, const bool end_ram,
  const bool local_approx, const arma::vec& initial_mode,
  const unsigned int max_iter, const double conv_tol);
template void ung_amcmc::approx_mcmc(ung_ar1 model, const bool end_ram,
  const bool local_approx, const arma::vec& initial_mode,
  const unsigned int max_iter, const double conv_tol);

template<class T>
void ung_amcmc::approx_mcmc(T model, const bool end_ram, const bool local_approx,
  const arma::vec& initial_mode, const unsigned int max_iter, const double conv_tol) {
  
  // get the current values of theta
  arma::vec theta = model.theta;
  // compute the log[p(theta)]
  double logprior = model.log_prior_pdf(theta);
  if (!arma::is_finite(logprior)) {
    Rcpp::stop("Initial prior probability is not finite.");
  }
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
  if (!std::isfinite(approx_loglik))
    Rcpp::stop("Initial log-likelihood is not finite.");
  double acceptance_prob = 0.0;
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  arma::vec scales_prop = scales;
  arma::vec approx_y = approx_model.y;
  arma::vec approx_H = approx_model.H;
  
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
    double logprior_prop = model.log_prior_pdf(theta_prop);
    
    if (logprior_prop > -std::numeric_limits<double>::infinity() && !std::isnan(logprior_prop)) {
      // update parameters
      model.update_model(theta_prop);
      
      if (local_approx) {
        // construct the approximate Gaussian model
        mode_estimate = initial_mode;
        model.approximate(approx_model, mode_estimate, max_iter, conv_tol);
        
      } else {
        model.approximate(approx_model, mode_estimate, 0, conv_tol);
        
      }
      // compute unnormalized mode-based correction terms
      // log[g(y_t | ^alpha_t) / ~g(y_t | ^alpha_t)]
      scales_prop = model.scaling_factors(approx_model, mode_estimate);
      sum_scales = arma::accu(scales_prop);
      // compute the constant term (not really a constant in all cases, bad name!)
      const_term = compute_const_term(model, approx_model);
      // compute the log-likelihood of the approximate model
      // we could (should) extract this from fast_smoother used in approximation
      gaussian_loglik = approx_model.log_likelihood();
      double approx_loglik_prop = gaussian_loglik + const_term + sum_scales;
      
      acceptance_prob = std::min(1.0, std::exp(approx_loglik_prop - approx_loglik +
        logprior_prop - logprior + model.log_proposal_ratio(theta_prop, theta)));
      
      if (unif(model.engine) < acceptance_prob) {
        if (i > n_burnin) {
          acceptance_rate++;
          n_values++;
        }
        approx_loglik = approx_loglik_prop;
        logprior = logprior_prop;
        theta = theta_prop;
        scales = scales_prop;
        approx_y = approx_model.y;
        approx_H = approx_model.H;
        new_value = true;
      }
    } else acceptance_prob = 0.0;
    
    if (i > n_burnin && n_values % n_thin == 0) {
      //new block
      if (new_value) {
        approx_loglik_storage(n_stored) = approx_loglik;
        theta_storage.col(n_stored) = theta;
        if (store_modes) {
          y_storage.col(n_stored) = approx_y;
          H_storage.col(n_stored) = approx_H;
          scales_storage.col(n_stored) = scales;
        }
        prior_storage(n_stored) = logprior;
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

// approximate MCMC

template void ung_amcmc::is_correction_psi(ung_ssm model, const unsigned int nsim_states, 
  const unsigned int is_type, const unsigned int n_threads);
template void ung_amcmc::is_correction_psi(ung_bsm model, const unsigned int nsim_states, 
  const unsigned int is_type, const unsigned int n_threads);
template void ung_amcmc::is_correction_psi(ung_svm model, const unsigned int nsim_states, 
  const unsigned int is_type, const unsigned int n_threads);
template void ung_amcmc::is_correction_psi(ung_ar1 model, const unsigned int nsim_states, 
  const unsigned int is_type, const unsigned int n_threads);

template <class T>
void ung_amcmc::is_correction_psi(T model, const unsigned int nsim_states, 
  const unsigned int is_type, const unsigned int n_threads) {
  
  arma::cube Valpha(model.m, model.m, model.n + 1, arma::fill::zeros);
  double sum_w = 0.0;
  
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads) default(shared) firstprivate(model) 
{
  
  model.engine = sitmo::prng_engine(omp_get_thread_num() + 1);
  
  arma::vec tmp(1);
  ugg_ssm approx_model = model.approximate(tmp, 0, 0);
  
#pragma omp for schedule(dynamic)
  for (unsigned int i = 0; i < theta_storage.n_cols; i++) {
    
    model.update_model(theta_storage.col(i));
    approx_model.Z = model.Z;
    approx_model.T = model.T;
    approx_model.R = model.R;
    approx_model.a1 = model.a1;
    approx_model.P1 = model.P1;
    approx_model.beta = model.beta;
    approx_model.D = model.D;
    approx_model.C = model.C;
    approx_model.RR = model.RR;
    approx_model.xbeta = model.xbeta;
    approx_model.y = y_storage.col(i);
    approx_model.H = H_storage.col(i);
    approx_model.compute_HH();
    
    unsigned int nsim = nsim_states;
    if (is_type == 1) {
      nsim *= count_storage(i);
    }
    
    arma::cube alpha_i(model.m, model.n + 1, nsim);
    arma::mat weights_i(nsim, model.n + 1);
    arma::umat indices(nsim, model.n);
    
    double loglik = model.psi_filter(approx_model, 0, scales_storage.col(i),
      nsim, alpha_i, weights_i, indices);
    
    weight_storage(i) = std::exp(loglik);
    if (output_type != 3) {
      filter_smoother(alpha_i, indices);
      arma::vec w = weights_i.col(model.n);
      if (output_type == 1) {
        std::discrete_distribution<unsigned int> sample(w.begin(), w.end());
        alpha_storage.slice(i) = alpha_i.slice(sample(model.engine)).t();
      } else {
        arma::mat alphahat_i(model.m, model.n + 1);
        arma::cube Vt_i(model.m, model.m, model.n + 1);
        weighted_summary(alpha_i, alphahat_i, Vt_i, w);
#pragma omp critical
{
  arma::mat diff = alphahat_i - alphahat;
  double tmp = count_storage(i) + sum_w;
  alphahat = (alphahat * sum_w + alphahat_i * count_storage(i)) / tmp;
  for (unsigned int t = 0; t < model.n + 1; t++) {
    Valpha.slice(t) += diff.col(t) * (alphahat_i.col(t) - alphahat.col(t)).t();
  }
  Vt = (Vt * sum_w + Vt_i * count_storage(i)) / tmp;
  sum_w = tmp;
}
      }
    }
  }
}
#else
arma::vec tmp(1);
ugg_ssm approx_model = model.approximate(tmp, 0, 0);

for (unsigned int i = 0; i < theta_storage.n_cols; i++) {
  
  model.update_model(theta_storage.col(i));
  approx_model.Z = model.Z;
  approx_model.T = model.T;
  approx_model.R = model.R;
  approx_model.a1 = model.a1;
  approx_model.P1 = model.P1;
  approx_model.beta = model.beta;
  approx_model.D = model.D;
  approx_model.C = model.C;
  approx_model.RR = model.RR;
  approx_model.xbeta = model.xbeta;
  approx_model.y = y_storage.col(i);
  approx_model.H = H_storage.col(i);
  approx_model.compute_HH();
  
  unsigned int nsim = nsim_states;
  if (is_type == 1) {
    nsim *= count_storage(i);
  }
  
  arma::cube alpha_i(model.m, model.n + 1, nsim);
  arma::mat weights_i(nsim, model.n + 1);
  arma::umat indices(nsim, model.n);
  double loglik = model.psi_filter(approx_model, 0, scales_storage.col(i),
    nsim, alpha_i, weights_i, indices);
  
  weight_storage(i) = std::exp(loglik);
  if (output_type != 3) {
    filter_smoother(alpha_i, indices);
    arma::vec w = weights_i.col(model.n);
    if (output_type == 1) {
      std::discrete_distribution<unsigned int> sample(w.begin(), w.end());
      alpha_storage.slice(i) = alpha_i.slice(sample(model.engine)).t();
    } else {
      arma::mat alphahat_i(model.m, model.n + 1);
      arma::cube Vt_i(model.m, model.m, model.n + 1);
      weighted_summary(alpha_i, alphahat_i, Vt_i, w);
      
      arma::mat diff = alphahat_i - alphahat;
      double tmp = count_storage(i) + sum_w;
      alphahat = (alphahat * sum_w + alphahat_i * count_storage(i)) / tmp;
      for (unsigned int t = 0; t < model.n + 1; t++) {
        Valpha.slice(t) += diff.col(t) * (alphahat_i.col(t) - alphahat.col(t)).t();
      }
      Vt = (Vt * sum_w + Vt_i * count_storage(i)) / tmp;
      sum_w = tmp;
    }
  }
}

#endif
if (output_type == 2) {
  Vt += Valpha / theta_storage.n_cols; // Var[E(alpha)] + E[Var(alpha)]
}
posterior_storage = prior_storage + approx_loglik_storage + 
  arma::log(weight_storage);
}

template void ung_amcmc::is_correction_bsf(ung_ssm model, 
  const unsigned int nsim_states, const unsigned int is_type, 
  const unsigned int n_threads);
template void ung_amcmc::is_correction_bsf(ung_bsm model, 
  const unsigned int nsim_states, const unsigned int is_type, 
  const unsigned int n_threads);
template void ung_amcmc::is_correction_bsf(ung_svm model, 
  const unsigned int nsim_states, const unsigned int is_type, 
  const unsigned int n_threads);
template void ung_amcmc::is_correction_bsf(ung_ar1 model, 
  const unsigned int nsim_states, const unsigned int is_type, 
  const unsigned int n_threads);

template <class T>
void ung_amcmc::is_correction_bsf(T model, const unsigned int nsim_states, 
  const unsigned int is_type, const unsigned int n_threads) {
  
  arma::cube Valpha(model.m, model.m, model.n + 1, arma::fill::zeros);
  double sum_w = 0.0;
  
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads) default(shared) firstprivate(model) 
{
  
  model.engine = sitmo::prng_engine(omp_get_thread_num() + 1);
  
#pragma omp for schedule(dynamic)
  for (unsigned int i = 0; i < theta_storage.n_cols; i++) {
    
    model.update_model(theta_storage.col(i));
    
    unsigned int nsim = nsim_states;
    if (is_type == 1) {
      nsim *= count_storage(i);
    }
    
    arma::cube alpha_i(model.m, model.n + 1, nsim);
    arma::mat weights_i(nsim, model.n + 1);
    arma::umat indices(nsim, model.n);
    
    double loglik = model.bsf_filter(nsim, alpha_i, weights_i, indices);
    weight_storage(i) = std::exp(loglik - approx_loglik_storage(i));
    if (output_type != 3) {
      filter_smoother(alpha_i, indices);
      arma::vec w = weights_i.col(model.n);
      if (output_type == 1) {
        std::discrete_distribution<unsigned int> sample(w.begin(), w.end());
        alpha_storage.slice(i) = alpha_i.slice(sample(model.engine)).t();
      } else {
        arma::mat alphahat_i(model.m, model.n + 1);
        arma::cube Vt_i(model.m, model.m, model.n + 1);
        weighted_summary(alpha_i, alphahat_i, Vt_i, w);
#pragma omp critical
{
  arma::mat diff = alphahat_i - alphahat;
  double tmp = count_storage(i) + sum_w;
  alphahat = (alphahat * sum_w + alphahat_i * count_storage(i)) / tmp;
  for (unsigned int t = 0; t < model.n + 1; t++) {
    Valpha.slice(t) += diff.col(t) * (alphahat_i.col(t) - alphahat.col(t)).t();
  }
  Vt = (Vt * sum_w + Vt_i * count_storage(i)) / tmp;
  sum_w = tmp;
}
      }
    }
    
  }
}
#else
for (unsigned int i = 0; i < theta_storage.n_cols; i++) {
  
  model.update_model(theta_storage.col(i));
  
  unsigned int nsim = nsim_states;
  if (is_type == 1) {
    nsim *= count_storage(i);
  }
  
  arma::cube alpha_i(model.m, model.n + 1, nsim);
  arma::mat weights_i(nsim, model.n + 1);
  arma::umat indices(nsim, model.n);
  
  double loglik = model.bsf_filter(nsim, alpha_i, weights_i, indices);
  weight_storage(i) = std::exp(loglik - approx_loglik_storage(i));
  
  if (output_type != 3) {
    filter_smoother(alpha_i, indices);
    arma::vec w = weights_i.col(model.n);
    if (output_type == 1) {
      std::discrete_distribution<unsigned int> sample(w.begin(), w.end());
      alpha_storage.slice(i) = alpha_i.slice(sample(model.engine)).t();
    } else {
      arma::mat alphahat_i(model.m, model.n + 1);
      arma::cube Vt_i(model.m, model.m, model.n + 1);
      weighted_summary(alpha_i, alphahat_i, Vt_i, w);
      arma::mat diff = alphahat_i - alphahat;
      double tmp = count_storage(i) + sum_w;
      alphahat = (alphahat * sum_w + alphahat_i * count_storage(i)) / tmp;
      for (unsigned int t = 0; t < model.n + 1; t++) {
        Valpha.slice(t) += diff.col(t) * (alphahat_i.col(t) - alphahat.col(t)).t();
      }
      Vt = (Vt * sum_w + Vt_i * count_storage(i)) / tmp;
      sum_w = tmp;
      
    }
  }
}
#endif
if (output_type == 2) {
  Vt += Valpha / theta_storage.n_cols; // Var[E(alpha)] + E[Var(alpha)]
}
posterior_storage = prior_storage + arma::log(weight_storage);
}

template void ung_amcmc::is_correction_spdk(ung_ssm model, unsigned int nsim_states, 
  unsigned int is_type, const unsigned int n_threads);
template void ung_amcmc::is_correction_spdk(ung_bsm model, unsigned int nsim_states, 
  unsigned int is_type, const unsigned int n_threads);
template void ung_amcmc::is_correction_spdk(ung_svm model, unsigned int nsim_states, 
  unsigned int is_type, const unsigned int n_threads);
template void ung_amcmc::is_correction_spdk(ung_ar1 model, unsigned int nsim_states, 
  unsigned int is_type, const unsigned int n_threads);

template <class T>
void ung_amcmc::is_correction_spdk(T model, const unsigned int nsim_states, 
  const unsigned int is_type, const unsigned int n_threads) {
  
  arma::cube Valpha(model.m, model.m, model.n + 1, arma::fill::zeros);
  double sum_w = 0.0;
  
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads) default(shared) firstprivate(model) 
{
  
  model.engine = sitmo::prng_engine(omp_get_thread_num() + 1);
  
  arma::vec tmp(1);
  ugg_ssm approx_model = model.approximate(tmp, 0, 0);
  
#pragma omp for schedule(dynamic)
  for (unsigned int i = 0; i < theta_storage.n_cols; i++) {
    
    model.update_model(theta_storage.col(i));
    approx_model.Z = model.Z;
    approx_model.T = model.T;
    approx_model.R = model.R;
    approx_model.a1 = model.a1;
    approx_model.P1 = model.P1;
    approx_model.beta = model.beta;
    approx_model.D = model.D;
    approx_model.C = model.C;
    approx_model.RR = model.RR;
    approx_model.xbeta = model.xbeta;
    approx_model.y = y_storage.col(i);
    approx_model.H = H_storage.col(i);
    approx_model.compute_HH();
    
    unsigned int nsim = nsim_states;
    if (is_type == 1) {
      nsim *= count_storage(i);
    }
    
    arma::cube alpha_i = approx_model.simulate_states(nsim, true);
    arma::vec weights_i = model.importance_weights(approx_model, alpha_i);
    weights_i = arma::exp(weights_i - arma::accu(scales_storage.col(i)));
    weight_storage(i) = arma::mean(weights_i);
    if (output_type != 3) {
      if (output_type == 1) {
        std::discrete_distribution<unsigned int> sample(weights_i.begin(), weights_i.end());
        alpha_storage.slice(i) = alpha_i.slice(sample(model.engine)).t();
      } else {
        arma::mat alphahat_i(model.m, model.n + 1);
        arma::cube Vt_i(model.m, model.m, model.n + 1);
        weighted_summary(alpha_i, alphahat_i, Vt_i, weights_i);
#pragma omp critical
{
  arma::mat diff = alphahat_i - alphahat;
  double tmp = count_storage(i) + sum_w;
  alphahat = (alphahat * sum_w + alphahat_i * count_storage(i)) / tmp;
  for (unsigned int t = 0; t < model.n + 1; t++) {
    Valpha.slice(t) += diff.col(t) * (alphahat_i.col(t) - alphahat.col(t)).t();
  }
  Vt = (Vt * sum_w + Vt_i * count_storage(i)) / tmp;
  sum_w = tmp;
}
      }
    }
  }
  
}
#else
arma::vec tmp(1);
ugg_ssm approx_model = model.approximate(tmp, 0, 0);

for (unsigned int i = 0; i < theta_storage.n_cols; i++) {
  
  model.update_model(theta_storage.col(i));
  approx_model.Z = model.Z;
  approx_model.T = model.T;
  approx_model.R = model.R;
  approx_model.a1 = model.a1;
  approx_model.P1 = model.P1;
  approx_model.beta = model.beta;
  approx_model.D = model.D;
  approx_model.C = model.C;
  approx_model.RR = model.RR;
  approx_model.xbeta = model.xbeta;
  approx_model.y = y_storage.col(i);
  approx_model.H = H_storage.col(i);
  approx_model.compute_HH();
  
  unsigned int nsim = nsim_states;
  if (is_type == 1) {
    nsim *= count_storage(i);
  }
  
  arma::cube alpha_i = approx_model.simulate_states(nsim, true);
  arma::vec weights_i = model.importance_weights(approx_model, alpha_i);
  weights_i = arma::exp(weights_i - arma::accu(scales_storage.col(i)));
  if (output_type != 3) {
    if (output_type == 1) {
      std::discrete_distribution<unsigned int> sample(weights_i.begin(), weights_i.end());
      alpha_storage.slice(i) = alpha_i.slice(sample(model.engine)).t();
    } else {
      arma::mat alphahat_i(model.m, model.n + 1);
      arma::cube Vt_i(model.m, model.m, model.n + 1);
      weighted_summary(alpha_i, alphahat_i, Vt_i, weights_i);
      arma::mat diff = alphahat_i - alphahat;
      double tmp = count_storage(i) + sum_w;
      alphahat = (alphahat * sum_w + alphahat_i * count_storage(i)) / tmp;
      for (unsigned int t = 0; t < model.n + 1; t++) {
        Valpha.slice(t) += diff.col(t) * (alphahat_i.col(t) - alphahat.col(t)).t();
      }
      Vt = (Vt * sum_w + Vt_i * count_storage(i)) / tmp;
      sum_w = tmp;
    }
  }
}

#endif
if (output_type == 2) {
  Vt += Valpha / theta_storage.n_cols; // Var[E(alpha)] + E[Var(alpha)]
}
posterior_storage = prior_storage + approx_loglik_storage + 
  arma::log(weight_storage);
}

template void ung_amcmc::approx_state_posterior(ung_ssm model, const unsigned int n_threads);
template void ung_amcmc::approx_state_posterior(ung_bsm model, const unsigned int n_threads);
template void ung_amcmc::approx_state_posterior(ung_svm model, const unsigned int n_threads);
template void ung_amcmc::approx_state_posterior(ung_ar1 model, const unsigned int n_threads);

template <class T>
void ung_amcmc::approx_state_posterior(T model, const unsigned int n_threads) {
  
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads) default(shared) firstprivate(model) 
{
  
  model.engine = sitmo::prng_engine(omp_get_thread_num() + 1);
  
  arma::vec tmp(1);
  ugg_ssm approx_model = model.approximate(tmp, 0, 0);
  
#pragma omp for schedule(dynamic)
  for (unsigned int i = 0; i < theta_storage.n_cols; i++) {
    
    model.update_model(theta_storage.col(i));
    approx_model.Z = model.Z;
    approx_model.T = model.T;
    approx_model.R = model.R;
    approx_model.a1 = model.a1;
    approx_model.P1 = model.P1;
    approx_model.beta = model.beta;
    approx_model.D = model.D;
    approx_model.C = model.C;
    approx_model.RR = model.RR;
    approx_model.xbeta = model.xbeta;
    approx_model.y = y_storage.col(i);
    approx_model.H = H_storage.col(i);
    approx_model.compute_HH();
    alpha_storage.slice(i) = approx_model.simulate_states(1).slice(0).t();
  }
}
#else
arma::vec tmp(1);
ugg_ssm approx_model = model.approximate(tmp, 0, 0);

for (unsigned int i = 0; i < theta_storage.n_cols; i++) {
  
  model.update_model(theta_storage.col(i));
  approx_model.Z = model.Z;
  approx_model.T = model.T;
  approx_model.R = model.R;
  approx_model.a1 = model.a1;
  approx_model.P1 = model.P1;
  approx_model.beta = model.beta;
  approx_model.D = model.D;
  approx_model.C = model.C;
  approx_model.RR = model.RR;
  approx_model.xbeta = model.xbeta;
  approx_model.y = y_storage.col(i);
  approx_model.H = H_storage.col(i);
  approx_model.compute_HH();
  alpha_storage.slice(i) = approx_model.simulate_states(1).slice(0).t();
}
#endif

}

