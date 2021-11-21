#ifdef _OPENMP
#include <omp.h>
#endif
#include <ramcmc.h>
#include "approx_mcmc.h"
#include "model_ssm_mng.h"
#include "model_ssm_ung.h"
#include "model_bsm_ng.h"
#include "model_svm.h"
#include "model_ar1_ng.h"
#include "model_ssm_nlg.h"
#include "model_ssm_sde.h"

#include "rep_mat.h"
#include "distr_consts.h"
#include "filter_smoother.h"
#include "summary.h"

#include "parset_ng.h"

approx_mcmc::approx_mcmc(const unsigned int iter,
  const unsigned int burnin, const unsigned int thin, const unsigned int n,
  const unsigned int m, const unsigned int k, const double target_acceptance, 
  const double gamma, const arma::mat& S, const unsigned int output_type, 
  const bool store_modes, const bool verbose) :
  mcmc(iter, burnin, thin, n, m,
    target_acceptance, gamma, S, output_type, verbose),
    weight_storage(arma::vec(n_samples, arma::fill::zeros)),
    mode_storage(arma::cube(k, n, n_samples * store_modes)),
    approx_loglik_storage(arma::vec(n_samples)),
    prior_storage(arma::vec(n_samples)), store_modes(store_modes) {
}

void approx_mcmc::trim_storage() {
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
    mode_storage.resize(mode_storage.n_rows, mode_storage.n_cols, n_stored);
  }
}

void approx_mcmc::expand() {
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
  
  arma::vec expanded_approx_loglik = rep_vec(approx_loglik_storage, count_storage);
  approx_loglik_storage.set_size(n_stored);
  approx_loglik_storage = expanded_approx_loglik;
  
  if (output_type == 1) {
    arma::cube expanded_alpha = rep_cube(alpha_storage, count_storage);
    alpha_storage.set_size(alpha_storage.n_rows, alpha_storage.n_cols, n_stored);
    alpha_storage = expanded_alpha;
  }
  
  if (store_modes) {
    arma::cube expanded_mode = rep_cube(mode_storage, count_storage);
    mode_storage.set_size(mode_storage.n_rows, mode_storage.n_cols, n_stored);
    mode_storage = expanded_mode;
  }
  
  count_storage.resize(n_stored);
  count_storage.ones();
}

// run approximate MCMC for
// non-linear and/or non-Gaussian state space model with linear-Gaussian states
template void approx_mcmc::amcmc(ssm_ung model, const unsigned int method, const bool end_ram, 
  const Rcpp::Function update_fn, const Rcpp::Function prior_fn);

template void approx_mcmc::amcmc(bsm_ng model, const unsigned int method, const bool end_ram, 
  const Rcpp::Function update_fn, const Rcpp::Function prior_fn);

template void approx_mcmc::amcmc(svm model, const unsigned int method, const bool end_ram, 
  const Rcpp::Function update_fn, const Rcpp::Function prior_fn);

template void approx_mcmc::amcmc(ar1_ng model, const unsigned int method, const bool end_ram, 
  const Rcpp::Function update_fn, const Rcpp::Function prior_fn);

template void approx_mcmc::amcmc(ssm_nlg model, const unsigned int method, const bool end_ram, 
  const Rcpp::Function update_fn, const Rcpp::Function prior_fn);

template void approx_mcmc::amcmc(ssm_mng model, const unsigned int method, const bool end_ram, 
  const Rcpp::Function update_fn, const Rcpp::Function prior_fn);

template<class T>
void approx_mcmc::amcmc(T model, const unsigned int method, const bool end_ram, 
  const Rcpp::Function update_fn, const Rcpp::Function prior_fn) {
  
  // get the current values of theta
  arma::vec theta = model.theta;
  model.update_model(theta, update_fn); // just in case
  
  // compute the log[p(theta)]
  double logprior = model.log_prior_pdf(theta, prior_fn);
  if (!arma::is_finite(logprior)) {
    Rcpp::stop("Initial prior probability is not finite.");
  }
  // placeholders
  arma::cube alpha(1, 1, 1);
  arma::mat weights(1, 1);
  arma::umat indices(1, 1);
  
  // compute the approximate log-likelihood
  arma::vec ll = model.log_likelihood(method, 0, alpha, weights, indices);
  double approx_loglik = ll(0);
  if (!std::isfinite(approx_loglik))
    Rcpp::stop("Initial log-likelihood is not finite.");
  
  
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  arma::mat mode(mode_storage.n_rows, mode_storage.n_cols);
  mode = model.mode_estimate;
  
  bool new_value = true;
  unsigned int n_values = 0;
  double acceptance_prob = 0.0;
  
  // don't update progress at each iteration
  unsigned int mod = std::max(1U, iter / 50);
  unsigned int ticks = 1;
  if (verbose) {
    Rcpp::Rcout<<"Starting MCMC. Progress:\n";
    Rcpp::Rcout<<"0%   10   20   30   40   50   60   70   80   90   100%\n";
    Rcpp::Rcout<<"|";
  }
  
  for (unsigned int i = 1; i <= iter; i++) {
    
    // sample from standard normal distribution
    arma::vec u(n_par);
    for(unsigned int j = 0; j < n_par; j++) {
      u(j) = normal(model.engine);
    }
    
    // propose new theta
    arma::vec theta_prop = theta + S * u;
    // compute prior
    double logprior_prop = model.log_prior_pdf(theta_prop, prior_fn);
    if (logprior_prop > -std::numeric_limits<double>::infinity() && !std::isnan(logprior_prop)) {
      // update parameters
      model.update_model(theta_prop, update_fn);
      
      arma::vec ll = model.log_likelihood(method, 0, alpha, weights, indices);
      double approx_loglik_prop = ll(0);
      acceptance_prob = std::min(1.0, std::exp(approx_loglik_prop - approx_loglik +
        logprior_prop - logprior));
      
      if (unif(model.engine) < acceptance_prob) {
        if (i > burnin) {
          acceptance_rate++;
          n_values++;
        }
        approx_loglik = approx_loglik_prop;
        logprior = logprior_prop;
        theta = theta_prop;
        mode = model.mode_estimate;
        new_value = true;
      }
    } else acceptance_prob = 0.0;
    
    if (i > burnin && n_values % thin == 0) {
      //new block
      if (new_value) {
        approx_loglik_storage(n_stored) = approx_loglik;
        theta_storage.col(n_stored) = theta;
        if (store_modes) {
          mode_storage.slice(n_stored) = mode;
        }
        prior_storage(n_stored) = logprior;
        count_storage(n_stored) = 1;
        n_stored++;
        new_value = false;
      } else {
        count_storage(n_stored - 1)++;
      }
    }
    
    if (!end_ram || i <= burnin) {
      ramcmc::adapt_S(S, u, acceptance_prob, target_acceptance, i, gamma);
    }
    if (i % mod == 0) {
      Rcpp::checkUserInterrupt();
      if (verbose) {
        if (ticks % 5 == 0) {
          Rcpp::Rcout<<"|";
        } else {
          Rcpp::Rcout<<"-";
        }
        ticks++;
      } 
    }
  }
  if (verbose) Rcpp::Rcout<<"\n";
  if (n_stored == 0) Rcpp::stop("No proposals were accepted in MCMC. Check your model.");
  
  trim_storage();
  acceptance_rate /= (iter - burnin);
  // approx posterior
  posterior_storage = approx_loglik_storage + prior_storage;
}


// run approximate MCMC for SDE model
void approx_mcmc::amcmc(ssm_sde model, const unsigned int nsim, const bool end_ram) {
  
  unsigned int m = 1;
  unsigned n = model.n;
  // compute the log[p(theta)]
  double logprior = model.log_prior_pdf(model.theta);
  if (!arma::is_finite(logprior)) {
    Rcpp::stop("Initial prior probability is not finite.");
  }
  
  arma::cube alpha(m, n + 1, nsim);
  arma::mat weights(nsim, n + 1);
  arma::umat indices(nsim, n);
  double loglik = model.bsf_filter(nsim, model.L_c, alpha, weights, indices);
  if (!std::isfinite(loglik))
    Rcpp::stop("Initial log-likelihood is not finite.");
  
  double acceptance_prob = 0.0;
  bool new_value = true;
  unsigned int n_values = 0;
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  arma::vec theta = model.theta;
  
  // don't update progress at each iteration
  unsigned int mod = std::max(1U, iter / 50);
  unsigned int ticks = 1;
  if (verbose) {
    Rcpp::Rcout<<"Starting MCMC. Progress:\n";
    Rcpp::Rcout<<"0%   10   20   30   40   50   60   70   80   90   100%\n";
    Rcpp::Rcout<<"|";
  }
  
  for (unsigned int i = 1; i <= iter; i++) {
    
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
      
      double loglik_prop = model.bsf_filter(nsim, model.L_c, alpha, weights, indices);
      
      //compute the acceptance probability
      // use explicit min(...) as we need this value later
      acceptance_prob = std::min(1.0, std::exp(loglik_prop - loglik +
        logprior_prop - logprior));
      
      //accept
      if (unif(model.engine) < acceptance_prob) {
        if (i > burnin) {
          acceptance_rate++;
          n_values++;
        }
        loglik = loglik_prop;
        logprior = logprior_prop;
        theta = theta_prop;
        new_value = true;
      }
    } else acceptance_prob = 0.0;
    
    if (i > burnin && n_values % thin == 0) {
      //new block
      if (new_value) {
        approx_loglik_storage(n_stored) = loglik;
        prior_storage(n_stored) = logprior;
        theta_storage.col(n_stored) = theta;
        count_storage(n_stored) = 1;
        n_stored++;
        new_value = false;
      } else {
        count_storage(n_stored - 1)++;
      }
    }
    
    if (!end_ram || i <= burnin) {
      ramcmc::adapt_S(S, u, acceptance_prob, target_acceptance, i, gamma);
    }
    if (i % mod == 0) {
      Rcpp::checkUserInterrupt();
      if (verbose) {
        if (ticks % 5 == 0) {
          Rcpp::Rcout<<"|";
        } else {
          Rcpp::Rcout<<"-";
        }
        ticks++;
      } 
    }
  }
  if (verbose) Rcpp::Rcout<<"\n";
  if (n_stored == 0) Rcpp::stop("No proposals were accepted in MCMC. Check your model.");
  
  trim_storage();
  acceptance_rate /= (iter - burnin);
}


// IS-correction with psi-APF. 

// Note that update_fn is not actually used these models
template void approx_mcmc::is_correction_psi(bsm_ng model, const unsigned int nsim,
  const unsigned int is_type, const unsigned int n_threads, 
  const Rcpp::Function update_fn);
template void approx_mcmc::is_correction_psi(svm model, const unsigned int nsim,
  const unsigned int is_type, const unsigned int n_threads, 
  const Rcpp::Function update_fn);
template void approx_mcmc::is_correction_psi(ar1_ng model, const unsigned int nsim,
  const unsigned int is_type, const unsigned int n_threads, 
  const Rcpp::Function update_fn);
template void approx_mcmc::is_correction_psi(ssm_nlg model, const unsigned int nsim,
  const unsigned int is_type, const unsigned int n_threads, 
  const Rcpp::Function update_fn);

template <class T>
void approx_mcmc::is_correction_psi(T model, const unsigned int nsim,
  const unsigned int is_type, const unsigned int n_threads, 
  const Rcpp::Function update_fn) {
  if (verbose) {
    Rcpp::Rcout<<"\nStarting IS-correction phase with "<<n_threads<<" thread(s).\n";
  }
  
  arma::cube Valpha(model.m, model.m, model.n + 1, arma::fill::zeros);
  double sum_w = 0.0;
  
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads) default(shared) firstprivate(model)
{
  model.engine = sitmo::prng_engine(omp_get_thread_num() + 1);
#pragma omp for schedule(static)
#endif
  for (unsigned int i = 0; i < n_stored; i++) {
    
    model.update_model(theta_storage.col(i));
    model.approximate_for_is(mode_storage.slice(i));
    
    unsigned int nsimc = nsim;
    if (is_type == 1) {
      nsimc *= count_storage(i);
    }
    arma::cube alpha_i(model.m, model.n + 1, nsimc, arma::fill::zeros);
    arma::mat weights_i(nsimc, model.n + 1, arma::fill::zeros);
    arma::umat indices(nsimc, model.n, arma::fill::zeros);
    
    double loglik = model.psi_filter(nsimc, alpha_i, weights_i, indices);
    
    weight_storage(i) = std::exp(loglik);
    
    if (output_type != 3) {
      filter_smoother(alpha_i, indices);
      
      if (output_type == 1) {
        std::uniform_int_distribution<unsigned int> sample(0, nsimc - 1);
        alpha_storage.slice(i) = alpha_i.slice(sample(model.engine)).t();
      } else {
        arma::mat alphahat_i(model.m, model.n + 1);
        arma::cube Vt_i(model.m, model.m, model.n + 1);
        summary(alpha_i, alphahat_i, Vt_i);
#ifdef _OPENMP
#pragma omp critical
{
#endif
  double wnew = weight_storage(i) * count_storage(i);
  sum_w += wnew;
  arma::mat diff = alphahat_i - alphahat;
  alphahat += wnew / sum_w * diff; // update E(alpha)
  arma::mat diff2 = (alphahat_i - alphahat).t();
  for (unsigned int t = 0; t < model.n + 1; t++) {
    // update Var(alpha)
    Valpha.slice(t) += wnew * diff.col(t) * diff2.row(t);
  }
  // update Var(E(alpha))
  Vt += wnew / sum_w * (Vt_i - Vt);
#ifdef _OPENMP
}
#endif
      }
    }
  }
#ifdef _OPENMP
}
#endif

if (output_type == 2) {
  Vt += Valpha / sum_w; // Var[E(alpha)] + E[Var(alpha)]
}
posterior_storage = prior_storage + approx_loglik_storage +
  arma::log(weight_storage);
}

// IS-correction with bootstrap filter
template void approx_mcmc::is_correction_bsf(bsm_ng model,
  const unsigned int nsim, const unsigned int is_type,
  const unsigned int n_threads, 
  const Rcpp::Function update_fn);
template void approx_mcmc::is_correction_bsf(svm model,
  const unsigned int nsim, const unsigned int is_type,
  const unsigned int n_threads, 
  const Rcpp::Function update_fn);
template void approx_mcmc::is_correction_bsf(ar1_ng model,
  const unsigned int nsim, const unsigned int is_type,
  const unsigned int n_threads, 
  const Rcpp::Function update_fn);
template void approx_mcmc::is_correction_bsf(ssm_nlg model,
  const unsigned int nsim, const unsigned int is_type,
  const unsigned int n_threads, 
  const Rcpp::Function update_fn);

template <class T>
void approx_mcmc::is_correction_bsf(T model, const unsigned int nsim,
  const unsigned int is_type, const unsigned int n_threads, 
  const Rcpp::Function update_fn) {
  
  if (verbose) {
    Rcpp::Rcout<<"\nStarting IS-correction phase with "<<n_threads<<" thread(s).\n";
  }
  
  arma::cube Valpha(model.m, model.m, model.n + 1, arma::fill::zeros);
  double sum_w = 0.0;
  
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads) default(shared) firstprivate(model)
{
  model.engine = sitmo::prng_engine(omp_get_thread_num() + 1);
#pragma omp for schedule(static)
#endif
  for (unsigned int i = 0; i < n_stored; i++) {
    
    model.update_model(theta_storage.col(i));
    
    unsigned int nsimc = nsim;
    if (is_type == 1) {
      nsimc *= count_storage(i);
    }
    
    arma::cube alpha_i(model.m, model.n + 1, nsimc, arma::fill::zeros);
    arma::mat weights_i(nsimc, model.n + 1, arma::fill::zeros);
    arma::umat indices(nsimc, model.n, arma::fill::zeros);
    
    double loglik = model.bsf_filter(nsimc, alpha_i, weights_i, indices);
    weight_storage(i) = std::exp(loglik - approx_loglik_storage(i));
    if (output_type != 3) {
      filter_smoother(alpha_i, indices);
      if (output_type == 1) {
        std::uniform_int_distribution<unsigned int> sample(0, nsimc - 1);
        alpha_storage.slice(i) = alpha_i.slice(sample(model.engine)).t();
      } else {
        arma::mat alphahat_i(model.m, model.n + 1);
        arma::cube Vt_i(model.m, model.m, model.n + 1);
        summary(alpha_i, alphahat_i, Vt_i);
#ifdef _OPENMP      
#pragma omp critical
{
#endif
  double wnew = weight_storage(i) * count_storage(i);
  sum_w += wnew;
  arma::mat diff = alphahat_i - alphahat;
  alphahat += wnew / sum_w * diff; // update E(alpha)
  arma::mat diff2 = (alphahat_i - alphahat).t();
  for (unsigned int t = 0; t < model.n + 1; t++) {
    // update Var(alpha)
    Valpha.slice(t) += wnew * diff.col(t) * diff2.row(t);
  }
  // update Var(E(alpha))
  Vt += wnew / sum_w * (Vt_i - Vt);
#ifdef _OPENMP
}
#endif
      }
    }
  }
#ifdef _OPENMP
}
#endif

if (output_type == 2) {
  Vt += Valpha / sum_w; // Var[E(alpha)] + E[Var(alpha)]
}
posterior_storage = prior_storage + arma::log(weight_storage);
}


// SDE specialization for is_correction_bsf
template<>
void approx_mcmc::is_correction_bsf(ssm_sde model, const unsigned int nsim,
  const unsigned int is_type, const unsigned int n_threads, 
  const Rcpp::Function update_fn) {
  
  if (verbose) {
    Rcpp::Rcout<<"\nStarting IS-correction phase with "<<n_threads<<" thread(s).\n";
  }
  
  arma::cube Valpha(1, 1, model.n + 1, arma::fill::zeros);
  double sum_w = 0.0;
  
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads) default(shared) firstprivate(model)
{
  model.engine = sitmo::prng_engine(omp_get_thread_num() + 1);
  model.coarse_engine = sitmo::prng_engine(omp_get_thread_num() + n_threads + 1);
#pragma omp for schedule(static)
#endif
  for (unsigned int i = 0; i < n_stored; i++) {
    model.theta = theta_storage.col(i);
    unsigned int nsimc = nsim;
    if (is_type == 1) {
      nsimc *= count_storage(i);
    }
    arma::cube alpha_i(1, model.n + 1, nsimc, arma::fill::zeros);
    arma::mat weights_i(nsimc, model.n + 1, arma::fill::zeros);
    arma::umat indices(nsimc, model.n, arma::fill::zeros);
    double loglik = model.bsf_filter(nsimc, model.L_f, alpha_i, weights_i, indices);
    weight_storage(i) = std::exp(loglik - approx_loglik_storage(i));
    
    if (output_type != 3) {
      filter_smoother(alpha_i, indices);
      
      if (output_type == 1) {
        std::uniform_int_distribution<unsigned int> sample(0, nsimc - 1);
        alpha_storage.slice(i) = alpha_i.slice(sample(model.engine)).t();
      } else {
        arma::mat alphahat_i(1, model.n + 1);
        arma::cube Vt_i(1, 1, model.n + 1);
        summary(alpha_i, alphahat_i, Vt_i);
#ifdef _OPENMP
#pragma omp critical
{
#endif
  double wnew = weight_storage(i) * count_storage(i);
  sum_w += wnew;
  arma::mat diff = alphahat_i - alphahat;
  alphahat += wnew / sum_w * diff; // update E(alpha)
  arma::mat diff2 = (alphahat_i - alphahat).t();
  for (unsigned int t = 0; t < model.n + 1; t++) {
    // update Var(alpha)
    Valpha.slice(t) += wnew * diff.col(t) * diff2.row(t);
  }
  // update Var(E(alpha))
  Vt += wnew / sum_w * (Vt_i - Vt);
#ifdef _OPENMP
}
#endif
      }
    }
  }
#ifdef _OPENMP
}
#endif
if (output_type == 2) {
  Vt += Valpha / sum_w; // Var[E(alpha)] + E[Var(alpha)]
}
posterior_storage = prior_storage + approx_loglik_storage + arma::log(weight_storage);
}


// IS-correction using SPDK
// Not implemented for nonlinear models (we could though)
template void approx_mcmc::is_correction_spdk(bsm_ng model, const unsigned int nsim,
  unsigned int is_type, const unsigned int n_threads, 
  const Rcpp::Function update_fn);
template void approx_mcmc::is_correction_spdk(svm model, const unsigned int nsim,
  unsigned int is_type, const unsigned int n_threads, 
  const Rcpp::Function update_fn);
template void approx_mcmc::is_correction_spdk(ar1_ng model, const unsigned int nsim,
  unsigned int is_type, const unsigned int n_threads, 
  const Rcpp::Function update_fn);

template <class T>
void approx_mcmc::is_correction_spdk(T model, const unsigned int nsim,
  const unsigned int is_type, const unsigned int n_threads, 
  const Rcpp::Function update_fn) {
  
  if (verbose) {
    Rcpp::Rcout<<"\nStarting IS-correction phase with "<<n_threads<<" thread(s).\n";
  }
  
  arma::cube Valpha(model.m, model.m, model.n + 1, arma::fill::zeros);
  double sum_w = 0.0;
  
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads) default(shared) firstprivate(model)
{
  model.engine = sitmo::prng_engine(omp_get_thread_num() + 1);
  model.approx_model.engine = model.engine;
#pragma omp for schedule(static)
#endif
  for (unsigned int i = 0; i < n_stored; i++) {
    
    model.update_model(theta_storage.col(i));
    
    model.approximate_for_is(mode_storage.slice(i));
    
    unsigned int nsimc = nsim;
    if (is_type == 1) {
      nsimc *= count_storage(i);
    }
    
    arma::cube alpha_i = model.approx_model.simulate_states(nsimc);
    arma::vec weights_i = model.importance_weights(alpha_i);
    weights_i = arma::exp(weights_i - arma::accu(model.scales));
    weight_storage(i) = arma::mean(weights_i);
    if (output_type != 3) {
      if (output_type == 1) {
        std::discrete_distribution<unsigned int> sample(weights_i.begin(), weights_i.end());
        alpha_storage.slice(i) = alpha_i.slice(sample(model.engine)).t();
      } else {
        arma::mat alphahat_i(model.m, model.n + 1);
        arma::cube Vt_i(model.m, model.m, model.n + 1);
        weighted_summary(alpha_i, alphahat_i, Vt_i, weights_i);
#ifdef _OPENMP
#pragma omp critical
{
#endif
  double wnew = weight_storage(i) * count_storage(i);
  sum_w += wnew;
  arma::mat diff = alphahat_i - alphahat;
  alphahat += wnew / sum_w * diff; // update E(alpha)
  arma::mat diff2 = (alphahat_i - alphahat).t();
  for (unsigned int t = 0; t < model.n + 1; t++) {
    // update Var(alpha)
    Valpha.slice(t) += wnew * diff.col(t) * diff2.row(t);
  }
  // update Var(E(alpha))
  Vt += wnew / sum_w * (Vt_i - Vt);
#ifdef _OPENMP
}
#endif
      }
    }
  }
#ifdef _OPENMP
}
#endif

if (output_type == 2) {
  Vt += Valpha / sum_w; // Var[E(alpha)] + E[Var(alpha)]
}
posterior_storage = prior_storage + approx_loglik_storage +
  arma::log(weight_storage);
}


// is-correction functions for mng and ung to be used with parallelization
// avoids calling or sharing R function withing parallel region
// downside is additional memory requirements due to 
// storing of all n_stored * estimated model components
template<>
void approx_mcmc::is_correction_psi(ssm_ung model, const unsigned int nsim,
  const unsigned int is_type, const unsigned int n_threads, 
  const Rcpp::Function update_fn) {
  
  if (verbose) {
    Rcpp::Rcout<<"\nStarting IS-correction phase with "<<n_threads<<" thread(s).\n";
  }
  
  arma::cube Valpha(model.m, model.m, model.n + 1, arma::fill::zeros);
  double sum_w = 0.0;
  
#ifdef _OPENMP
  
  parset_ung pars(model, theta_storage, update_fn);
  
#pragma omp parallel num_threads(n_threads) default(shared) firstprivate(model)
{
  
  model.engine = sitmo::prng_engine(omp_get_thread_num() + 1);
  
#pragma omp for schedule(static)
  for (unsigned int i = 0; i < n_stored; i++) {
    
    pars.update(model, i);
    
    model.approximate_for_is(mode_storage.slice(i));
    
    unsigned int nsimc = nsim;
    if (is_type == 1) {
      nsimc *= count_storage(i);
    }
    arma::cube alpha_i(model.m, model.n + 1, nsimc, arma::fill::zeros);
    arma::mat weights_i(nsimc, model.n + 1, arma::fill::zeros);
    arma::umat indices(nsimc, model.n, arma::fill::zeros);
    
    double loglik = model.psi_filter(nsimc, alpha_i, weights_i, indices);
    
    weight_storage(i) = std::exp(loglik);
    
    if (output_type != 3) {
      filter_smoother(alpha_i, indices);
      
      if (output_type == 1) {
        std::uniform_int_distribution<unsigned int> sample(0, nsimc - 1);
        alpha_storage.slice(i) = alpha_i.slice(sample(model.engine)).t();
      } else {
        arma::mat alphahat_i(model.m, model.n + 1);
        arma::cube Vt_i(model.m, model.m, model.n + 1);
        summary(alpha_i, alphahat_i, Vt_i);
        
#pragma omp critical
{
  double wnew = weight_storage(i) * count_storage(i);
  sum_w += wnew;
  arma::mat diff = alphahat_i - alphahat;
  alphahat += wnew / sum_w * diff; // update E(alpha)
  arma::mat diff2 = (alphahat_i - alphahat).t();
  for (unsigned int t = 0; t < model.n + 1; t++) {
    // update Var(alpha)
    Valpha.slice(t) += wnew * diff.col(t) * diff2.row(t);
  }
  // update Var(E(alpha))
  Vt += wnew / sum_w * (Vt_i - Vt);
}
      }
    }
  }
}
#else

for (unsigned int i = 0; i < n_stored; i++) {
  
  model.update_model(theta_storage.col(i), update_fn);
  model.approximate_for_is(mode_storage.slice(i));
  
  unsigned int nsimc = nsim;
  if (is_type == 1) {
    nsimc *= count_storage(i);
  }
  arma::cube alpha_i(model.m, model.n + 1, nsimc, arma::fill::zeros);
  arma::mat weights_i(nsimc, model.n + 1, arma::fill::zeros);
  arma::umat indices(nsimc, model.n, arma::fill::zeros);
  
  double loglik = model.psi_filter(nsimc, alpha_i, weights_i, indices);
  weight_storage(i) = std::exp(loglik);
  
  if (output_type != 3) {
    filter_smoother(alpha_i, indices);
    
    if (output_type == 1) {
      std::uniform_int_distribution<unsigned int> sample(0, nsimc - 1);
      alpha_storage.slice(i) = alpha_i.slice(sample(model.engine)).t();
    } else {
      arma::mat alphahat_i(model.m, model.n + 1);
      arma::cube Vt_i(model.m, model.m, model.n + 1);
      summary(alpha_i, alphahat_i, Vt_i);
      
      double wnew = weight_storage(i) * count_storage(i);
      sum_w += wnew;
      arma::mat diff = alphahat_i - alphahat;
      alphahat += wnew / sum_w * diff; // update E(alpha)
      arma::mat diff2 = (alphahat_i - alphahat).t();
      for (unsigned int t = 0; t < model.n + 1; t++) {
        // update Var(alpha)
        Valpha.slice(t) += wnew * diff.col(t) * diff2.row(t);
      }
      // update Var(E(alpha))
      Vt += wnew / sum_w * (Vt_i - Vt);
    }
  }
}

#endif
if (output_type == 2) {
  Vt += Valpha / sum_w; // Var[E(alpha)] + E[Var(alpha)]
}
posterior_storage = prior_storage + approx_loglik_storage +
  arma::log(weight_storage);
}

template<>
void approx_mcmc::is_correction_bsf(ssm_ung model, const unsigned int nsim,
  const unsigned int is_type, const unsigned int n_threads, 
  const Rcpp::Function update_fn) {
  
  if (verbose) {
    Rcpp::Rcout<<"\nStarting IS-correction phase with "<<n_threads<<" thread(s).\n";
  }
  
  arma::cube Valpha(model.m, model.m, model.n + 1, arma::fill::zeros);
  double sum_w = 0.0;
  
#ifdef _OPENMP
  
  parset_ung pars(model, theta_storage, update_fn);
  
#pragma omp parallel num_threads(n_threads) default(shared) firstprivate(model)
{
  
  model.engine = sitmo::prng_engine(omp_get_thread_num() + 1);
  
#pragma omp for schedule(static)
  for (unsigned int i = 0; i < n_stored; i++) {
    
    pars.update(model, i);
    
    unsigned int nsimc = nsim;
    if (is_type == 1) {
      nsimc *= count_storage(i);
    }
    
    arma::cube alpha_i(model.m, model.n + 1, nsimc, arma::fill::zeros);
    arma::mat weights_i(nsimc, model.n + 1, arma::fill::zeros);
    arma::umat indices(nsimc, model.n, arma::fill::zeros);
    
    double loglik = model.bsf_filter(nsimc, alpha_i, weights_i, indices);
    weight_storage(i) = std::exp(loglik - approx_loglik_storage(i));
    if (output_type != 3) {
      filter_smoother(alpha_i, indices);
      
      if (output_type == 1) {
        std::uniform_int_distribution<unsigned int> sample(0, nsimc - 1);
        alpha_storage.slice(i) = alpha_i.slice(sample(model.engine)).t();
      } else {
        arma::mat alphahat_i(model.m, model.n + 1);
        arma::cube Vt_i(model.m, model.m, model.n + 1);
        summary(alpha_i, alphahat_i, Vt_i);
        
#pragma omp critical
{
  double wnew = weight_storage(i) * count_storage(i);
  sum_w += wnew;
  arma::mat diff = alphahat_i - alphahat;
  alphahat += wnew / sum_w * diff; // update E(alpha)
  arma::mat diff2 = (alphahat_i - alphahat).t();
  for (unsigned int t = 0; t < model.n + 1; t++) {
    // update Var(alpha)
    Valpha.slice(t) += wnew * diff.col(t) * diff2.row(t);
  }
  // update Var(E(alpha))
  Vt += wnew / sum_w * (Vt_i - Vt);
}
      }
    }
  }
}
#else
for (unsigned int i = 0; i < n_stored; i++) {
  
  model.update_model(theta_storage.col(i), update_fn);
  
  
  unsigned int nsimc = nsim;
  if (is_type == 1) {
    nsimc *= count_storage(i);
  }
  
  arma::cube alpha_i(model.m, model.n + 1, nsimc, arma::fill::zeros);
  arma::mat weights_i(nsimc, model.n + 1, arma::fill::zeros);
  arma::umat indices(nsimc, model.n, arma::fill::zeros);
  
  double loglik = model.bsf_filter(nsimc, alpha_i, weights_i, indices);
  weight_storage(i) = std::exp(loglik - approx_loglik_storage(i));
  
  if (output_type != 3) {
    filter_smoother(alpha_i, indices);
    
    if (output_type == 1) {
      std::uniform_int_distribution<unsigned int> sample(0, nsimc - 1);
      alpha_storage.slice(i) = alpha_i.slice(sample(model.engine)).t();
    } else {
      arma::mat alphahat_i(model.m, model.n + 1);
      arma::cube Vt_i(model.m, model.m, model.n + 1);
      summary(alpha_i, alphahat_i, Vt_i);
      
      double wnew = weight_storage(i) * count_storage(i);
      sum_w += wnew;
      arma::mat diff = alphahat_i - alphahat;
      alphahat += wnew / sum_w * diff; // update E(alpha)
      arma::mat diff2 = (alphahat_i - alphahat).t();
      for (unsigned int t = 0; t < model.n + 1; t++) {
        // update Var(alpha)
        Valpha.slice(t) += wnew * diff.col(t) * diff2.row(t);
      }
      // update Var(E(alpha))
      Vt += wnew / sum_w * (Vt_i - Vt);
      
    }
  }
}
#endif
if (output_type == 2) {
  Vt += Valpha / sum_w; // Var[E(alpha)] + E[Var(alpha)]
}
posterior_storage = prior_storage + arma::log(weight_storage);
}

template<>
void approx_mcmc::is_correction_spdk(ssm_ung model, const unsigned int nsim,
  const unsigned int is_type, const unsigned int n_threads, 
  const Rcpp::Function update_fn) {
  
  if (verbose) {
    Rcpp::Rcout<<"\nStarting IS-correction phase with "<<n_threads<<" thread(s).\n";
  }
  
  arma::cube Valpha(model.m, model.m, model.n + 1, arma::fill::zeros);
  double sum_w = 0.0;
  
#ifdef _OPENMP
  
  parset_ung pars(model, theta_storage, update_fn);
  
#pragma omp parallel num_threads(n_threads) default(shared) firstprivate(model)
{
  
  model.engine = sitmo::prng_engine(omp_get_thread_num() + 1);
  
  model.approx_model.engine = model.engine;
  
#pragma omp for schedule(static)
  for (unsigned int i = 0; i < n_stored; i++) {
    
    pars.update(model, i);
    
    model.approximate_for_is(mode_storage.slice(i));
    
    unsigned int nsimc = nsim;
    if (is_type == 1) {
      nsimc *= count_storage(i);
    }
    
    arma::cube alpha_i = model.approx_model.simulate_states(nsimc);
    arma::vec weights_i = model.importance_weights(alpha_i);
    weights_i = arma::exp(weights_i - arma::accu(model.scales));
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
  double wnew = weight_storage(i) * count_storage(i);
  sum_w += wnew;
  arma::mat diff = alphahat_i - alphahat;
  alphahat += wnew / sum_w * diff; // update E(alpha)
  arma::mat diff2 = (alphahat_i - alphahat).t();
  for (unsigned int t = 0; t < model.n + 1; t++) {
    // update Var(alpha)
    Valpha.slice(t) += wnew * diff.col(t) * diff2.row(t);
  }
  // update Var(E(alpha))
  Vt += wnew / sum_w * (Vt_i - Vt);
}
      }
    }
  }
  
}
#else

for (unsigned int i = 0; i < n_stored; i++) {
  
  model.update_model(theta_storage.col(i), update_fn);
  model.approximate_for_is(mode_storage.slice(i));
  
  unsigned int nsimc = nsim;
  if (is_type == 1) {
    nsimc *= count_storage(i);
  }
  
  arma::cube alpha_i = model.approx_model.simulate_states(nsimc);
  arma::vec weights_i = model.importance_weights(alpha_i);
  weights_i = arma::exp(weights_i - arma::accu(model.scales));
  weight_storage(i) = arma::mean(weights_i);
  if (output_type != 3) {
    if (output_type == 1) {
      std::discrete_distribution<unsigned int> sample(weights_i.begin(), weights_i.end());
      alpha_storage.slice(i) = alpha_i.slice(sample(model.engine)).t();
    } else {
      arma::mat alphahat_i(model.m, model.n + 1);
      arma::cube Vt_i(model.m, model.m, model.n + 1);
      weighted_summary(alpha_i, alphahat_i, Vt_i, weights_i);
      
      double wnew = weight_storage(i) * count_storage(i);
      sum_w += wnew;
      arma::mat diff = alphahat_i - alphahat;
      alphahat += wnew / sum_w * diff; // update E(alpha)
      arma::mat diff2 = (alphahat_i - alphahat).t();
      for (unsigned int t = 0; t < model.n + 1; t++) {
        // update Var(alpha)
        Valpha.slice(t) += wnew * diff.col(t) * diff2.row(t);
      }
      // update Var(E(alpha))
      Vt += wnew / sum_w * (Vt_i - Vt);
    }
  }
}

#endif
if (output_type == 2) {
  Vt += Valpha / sum_w; // Var[E(alpha)] + E[Var(alpha)]
}
posterior_storage = prior_storage + approx_loglik_storage +
  arma::log(weight_storage);
}

template<>
void approx_mcmc::is_correction_psi(ssm_mng model, const unsigned int nsim,
  const unsigned int is_type, const unsigned int n_threads, 
  const Rcpp::Function update_fn) {
  
  if (verbose) {
    Rcpp::Rcout<<"\nStarting IS-correction phase with "<<n_threads<<" thread(s).\n";
  }
  
  arma::cube Valpha(model.m, model.m, model.n + 1, arma::fill::zeros);
  double sum_w = 0.0;
  
#ifdef _OPENMP
  
  parset_mng pars(model, theta_storage, update_fn);
  
#pragma omp parallel num_threads(n_threads) default(shared) firstprivate(model)
{
  
  model.engine = sitmo::prng_engine(omp_get_thread_num() + 1);
  
#pragma omp for schedule(static)
  for (unsigned int i = 0; i < n_stored; i++) {
    
    pars.update(model, i);
    
    model.approximate_for_is(mode_storage.slice(i));
    
    unsigned int nsimc = nsim;
    if (is_type == 1) {
      nsimc *= count_storage(i);
    }
    arma::cube alpha_i(model.m, model.n + 1, nsimc, arma::fill::zeros);
    arma::mat weights_i(nsimc, model.n + 1, arma::fill::zeros);
    arma::umat indices(nsimc, model.n, arma::fill::zeros);
    
    double loglik = model.psi_filter(nsimc, alpha_i, weights_i, indices);
    
    weight_storage(i) = std::exp(loglik);
    
    if (output_type != 3) {
      filter_smoother(alpha_i, indices);
      
      if (output_type == 1) {
        std::uniform_int_distribution<unsigned int> sample(0, nsimc - 1);
        alpha_storage.slice(i) = alpha_i.slice(sample(model.engine)).t();
      } else {
        arma::mat alphahat_i(model.m, model.n + 1);
        arma::cube Vt_i(model.m, model.m, model.n + 1);
        summary(alpha_i, alphahat_i, Vt_i);
        
#pragma omp critical
{
  double wnew = weight_storage(i) * count_storage(i);
  sum_w += wnew;
  arma::mat diff = alphahat_i - alphahat;
  alphahat += wnew / sum_w * diff; // update E(alpha)
  arma::mat diff2 = (alphahat_i - alphahat).t();
  for (unsigned int t = 0; t < model.n + 1; t++) {
    // update Var(alpha)
    Valpha.slice(t) += wnew * diff.col(t) * diff2.row(t);
  }
  // update Var(E(alpha))
  Vt += wnew / sum_w * (Vt_i - Vt);
}
      }
    }
  }
}
#else

for (unsigned int i = 0; i < n_stored; i++) {
  
  model.update_model(theta_storage.col(i), update_fn);
  model.approximate_for_is(mode_storage.slice(i));
  
  unsigned int nsimc = nsim;
  if (is_type == 1) {
    nsimc *= count_storage(i);
  }
  arma::cube alpha_i(model.m, model.n + 1, nsimc, arma::fill::zeros);
  arma::mat weights_i(nsimc, model.n + 1, arma::fill::zeros);
  arma::umat indices(nsimc, model.n, arma::fill::zeros);
  
  double loglik = model.psi_filter(nsimc, alpha_i, weights_i, indices);
  weight_storage(i) = std::exp(loglik);
  
  if (output_type != 3) {
    filter_smoother(alpha_i, indices);
    
    if (output_type == 1) {
      std::uniform_int_distribution<unsigned int> sample(0, nsimc - 1);
      alpha_storage.slice(i) = alpha_i.slice(sample(model.engine)).t();
    } else {
      arma::mat alphahat_i(model.m, model.n + 1);
      arma::cube Vt_i(model.m, model.m, model.n + 1);
      summary(alpha_i, alphahat_i, Vt_i);
      
      double wnew = weight_storage(i) * count_storage(i);
      sum_w += wnew;
      arma::mat diff = alphahat_i - alphahat;
      alphahat += wnew / sum_w * diff; // update E(alpha)
      arma::mat diff2 = (alphahat_i - alphahat).t();
      for (unsigned int t = 0; t < model.n + 1; t++) {
        // update Var(alpha)
        Valpha.slice(t) += wnew * diff.col(t) * diff2.row(t);
      }
      // update Var(E(alpha))
      Vt += wnew / sum_w * (Vt_i - Vt);
    }
  }
}

#endif
if (output_type == 2) {
  Vt += Valpha / sum_w; // Var[E(alpha)] + E[Var(alpha)]
}
posterior_storage = prior_storage + approx_loglik_storage +
  arma::log(weight_storage);
}

template<>
void approx_mcmc::is_correction_bsf(ssm_mng model, const unsigned int nsim,
  const unsigned int is_type, const unsigned int n_threads, 
  const Rcpp::Function update_fn) {
  
  if (verbose) {
    Rcpp::Rcout<<"\nStarting IS-correction phase with "<<n_threads<<" thread(s).\n";
  }
  
  arma::cube Valpha(model.m, model.m, model.n + 1, arma::fill::zeros);
  double sum_w = 0.0;
  
#ifdef _OPENMP
  
  parset_mng pars(model, theta_storage, update_fn);
  
#pragma omp parallel num_threads(n_threads) default(shared) firstprivate(model)
{
  
  model.engine = sitmo::prng_engine(omp_get_thread_num() + 1);
  
#pragma omp for schedule(static)
  for (unsigned int i = 0; i < n_stored; i++) {
    
    pars.update(model, i);
    
    unsigned int nsimc = nsim;
    if (is_type == 1) {
      nsimc *= count_storage(i);
    }
    
    arma::cube alpha_i(model.m, model.n + 1, nsimc, arma::fill::zeros);
    arma::mat weights_i(nsimc, model.n + 1, arma::fill::zeros);
    arma::umat indices(nsimc, model.n, arma::fill::zeros);
    
    double loglik = model.bsf_filter(nsimc, alpha_i, weights_i, indices);
    weight_storage(i) = std::exp(loglik - approx_loglik_storage(i));
    if (output_type != 3) {
      filter_smoother(alpha_i, indices);
      
      if (output_type == 1) {
        std::uniform_int_distribution<unsigned int> sample(0, nsimc - 1);
        alpha_storage.slice(i) = alpha_i.slice(sample(model.engine)).t();
      } else {
        arma::mat alphahat_i(model.m, model.n + 1);
        arma::cube Vt_i(model.m, model.m, model.n + 1);
        summary(alpha_i, alphahat_i, Vt_i);
        
#pragma omp critical
{
  double wnew = weight_storage(i) * count_storage(i);
  sum_w += wnew;
  arma::mat diff = alphahat_i - alphahat;
  alphahat += wnew / sum_w * diff; // update E(alpha)
  arma::mat diff2 = (alphahat_i - alphahat).t();
  for (unsigned int t = 0; t < model.n + 1; t++) {
    // update Var(alpha)
    Valpha.slice(t) += wnew * diff.col(t) * diff2.row(t);
  }
  // update Var(E(alpha))
  Vt += wnew / sum_w * (Vt_i - Vt);
}
      }
    }
  }
}
#else
for (unsigned int i = 0; i < n_stored; i++) {
  
  model.update_model(theta_storage.col(i), update_fn);
  
  unsigned int nsimc = nsim;
  if (is_type == 1) {
    nsimc *= count_storage(i);
  }
  
  arma::cube alpha_i(model.m, model.n + 1, nsimc, arma::fill::zeros);
  arma::mat weights_i(nsimc, model.n + 1, arma::fill::zeros);
  arma::umat indices(nsimc, model.n, arma::fill::zeros);
  
  double loglik = model.bsf_filter(nsimc, alpha_i, weights_i, indices);
  weight_storage(i) = std::exp(loglik - approx_loglik_storage(i));
  
  if (output_type != 3) {
    filter_smoother(alpha_i, indices);
    
    if (output_type == 1) {
      std::uniform_int_distribution<unsigned int> sample(0, nsimc - 1);
      alpha_storage.slice(i) = alpha_i.slice(sample(model.engine)).t();
    } else {
      arma::mat alphahat_i(model.m, model.n + 1);
      arma::cube Vt_i(model.m, model.m, model.n + 1);
      summary(alpha_i, alphahat_i, Vt_i);
      
      double wnew = weight_storage(i) * count_storage(i);
      sum_w += wnew;
      arma::mat diff = alphahat_i - alphahat;
      alphahat += wnew / sum_w * diff; // update E(alpha)
      arma::mat diff2 = (alphahat_i - alphahat).t();
      for (unsigned int t = 0; t < model.n + 1; t++) {
        // update Var(alpha)
        Valpha.slice(t) += wnew * diff.col(t) * diff2.row(t);
      }
      // update Var(E(alpha))
      Vt += wnew / sum_w * (Vt_i - Vt);
      
    }
  }
}
#endif
if (output_type == 2) {
  Vt += Valpha / sum_w; // Var[E(alpha)] + E[Var(alpha)]
}
posterior_storage = prior_storage + arma::log(weight_storage);
}

template<>
void approx_mcmc::is_correction_spdk(ssm_mng model, const unsigned int nsim,
  const unsigned int is_type, const unsigned int n_threads, 
  const Rcpp::Function update_fn) {
  
  if (verbose) {
    Rcpp::Rcout<<"\nStarting IS-correction phase with "<<n_threads<<" thread(s).\n";
  }
  
  arma::cube Valpha(model.m, model.m, model.n + 1, arma::fill::zeros);
  double sum_w = 0.0;
  
#ifdef _OPENMP
  
  parset_mng pars(model, theta_storage, update_fn);
  
#pragma omp parallel num_threads(n_threads) default(shared) firstprivate(model)
{
  
  model.engine = sitmo::prng_engine(omp_get_thread_num() + 1);
  
  model.approx_model.engine = model.engine;
  
#pragma omp for schedule(static)
  for (unsigned int i = 0; i < n_stored; i++) {
    
    pars.update(model, i);
    
    model.approximate_for_is(mode_storage.slice(i));
    
    unsigned int nsimc = nsim;
    if (is_type == 1) {
      nsimc *= count_storage(i);
    }
    
    arma::cube alpha_i = model.approx_model.simulate_states(nsimc);
    arma::vec weights_i = model.importance_weights(alpha_i);
    weights_i = arma::exp(weights_i - arma::accu(model.scales));
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
  double wnew = weight_storage(i) * count_storage(i);
  sum_w += wnew;
  arma::mat diff = alphahat_i - alphahat;
  alphahat += wnew / sum_w * diff; // update E(alpha)
  arma::mat diff2 = (alphahat_i - alphahat).t();
  for (unsigned int t = 0; t < model.n + 1; t++) {
    // update Var(alpha)
    Valpha.slice(t) += wnew * diff.col(t) * diff2.row(t);
  }
  // update Var(E(alpha))
  Vt += wnew / sum_w * (Vt_i - Vt);
}
      }
    }
  }
  
}
#else

for (unsigned int i = 0; i < n_stored; i++) {
  model.update_model(theta_storage.col(i), update_fn);
  model.approximate_for_is(mode_storage.slice(i));
  
  unsigned int nsimc = nsim;
  if (is_type == 1) {
    nsimc *= count_storage(i);
  }
  
  arma::cube alpha_i = model.approx_model.simulate_states(nsimc);
  arma::vec weights_i = model.importance_weights(alpha_i);
  weights_i = arma::exp(weights_i - arma::accu(model.scales));
  weight_storage(i) = arma::mean(weights_i);
  if (output_type != 3) {
    if (output_type == 1) {
      std::discrete_distribution<unsigned int> sample(weights_i.begin(), weights_i.end());
      alpha_storage.slice(i) = alpha_i.slice(sample(model.engine)).t();
    } else {
      arma::mat alphahat_i(model.m, model.n + 1);
      arma::cube Vt_i(model.m, model.m, model.n + 1);
      weighted_summary(alpha_i, alphahat_i, Vt_i, weights_i);
      
      double wnew = weight_storage(i) * count_storage(i);
      sum_w += wnew;
      arma::mat diff = alphahat_i - alphahat;
      alphahat += wnew / sum_w * diff; // update E(alpha)
      arma::mat diff2 = (alphahat_i - alphahat).t();
      for (unsigned int t = 0; t < model.n + 1; t++) {
        // update Var(alpha)
        Valpha.slice(t) += wnew * diff.col(t) * diff2.row(t);
      }
      // update Var(E(alpha))
      Vt += wnew / sum_w * (Vt_i - Vt);
    }
  }
}

#endif
if (output_type == 2) {
  Vt += Valpha / sum_w; // Var[E(alpha)] + E[Var(alpha)]
}
posterior_storage = prior_storage + approx_loglik_storage +
  arma::log(weight_storage);
}


// Sampling states from approximate posterior using simulation smoother
template void approx_mcmc::approx_state_posterior(ssm_nlg model, const unsigned int n_threads, 
  const Rcpp::Function update_fn);
template void approx_mcmc::approx_state_posterior(bsm_ng model, const unsigned int n_threads, 
  const Rcpp::Function update_fn);
template void approx_mcmc::approx_state_posterior(svm model, const unsigned int n_threads, 
  const Rcpp::Function update_fn);
template void approx_mcmc::approx_state_posterior(ar1_ng model, const unsigned int n_threads, 
  const Rcpp::Function update_fn);

template <class T>
void approx_mcmc::approx_state_posterior(T model, const unsigned int n_threads, 
  const Rcpp::Function update_fn) {
  
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads) default(shared) firstprivate(model)
{
  model.engine = sitmo::prng_engine(omp_get_thread_num() + 1);
#pragma omp for schedule(static)  
#endif
  for (unsigned int i = 0; i < n_stored; i++) {
    model.update_model(theta_storage.col(i));
    model.approximate_for_is(mode_storage.slice(i));
    alpha_storage.slice(i) = model.approx_model.simulate_states(1).slice(0).t();
  }
#ifdef _OPENMP
}
#endif

}

// for avoiding R function call within parallel region (critical pragma is not enough)
template<>
void approx_mcmc::approx_state_posterior(ssm_ung model, const unsigned int n_threads, 
  const Rcpp::Function update_fn) {
  
#ifdef _OPENMP
  
  parset_ung pars(model, theta_storage, update_fn);
  
#pragma omp parallel num_threads(n_threads) default(shared) firstprivate(model)
{
  
  model.engine = sitmo::prng_engine(omp_get_thread_num() + 1);
  
#pragma omp for schedule(static)
  for (unsigned int i = 0; i < n_stored; i++) {
    pars.update(model, i);
    model.approximate_for_is(mode_storage.slice(i));
    alpha_storage.slice(i) = model.approx_model.simulate_states(1).slice(0).t();
  }
}
#else
for (unsigned int i = 0; i < n_stored; i++) {
  model.update_model(theta_storage.col(i), update_fn);
  model.approximate_for_is(mode_storage.slice(i));
  alpha_storage.slice(i) = model.approx_model.simulate_states(1).slice(0).t();
}
#endif
}

template<>
void approx_mcmc::approx_state_posterior(ssm_mng model, const unsigned int n_threads, 
  const Rcpp::Function update_fn) {
  
#ifdef _OPENMP
  
  parset_mng pars(model, theta_storage, update_fn);
  
#pragma omp parallel num_threads(n_threads) default(shared) firstprivate(model)
{
  
  model.engine = sitmo::prng_engine(omp_get_thread_num() + 1);
  
#pragma omp for schedule(static)
  for (unsigned int i = 0; i < n_stored; i++) {
    pars.update(model, i);
    model.approximate_for_is(mode_storage.slice(i));
    alpha_storage.slice(i) = model.approx_model.simulate_states(1).slice(0).t();
  }
}
#else
for (unsigned int i = 0; i < n_stored; i++) {
  model.update_model(theta_storage.col(i), update_fn);
  model.approximate_for_is(mode_storage.slice(i));
  alpha_storage.slice(i) = model.approx_model.simulate_states(1).slice(0).t();
}
#endif
}

// should parallelize this as well
template void approx_mcmc::approx_state_summary(ssm_ung model, 
  const Rcpp::Function update_fn);
template void approx_mcmc::approx_state_summary(bsm_ng model, 
  const Rcpp::Function update_fn);
template void approx_mcmc::approx_state_summary(svm model, 
  const Rcpp::Function update_fn);
template void approx_mcmc::approx_state_summary(ar1_ng model, 
  const Rcpp::Function update_fn);
template void approx_mcmc::approx_state_summary(ssm_nlg model, 
  const Rcpp::Function update_fn);
template void approx_mcmc::approx_state_summary(ssm_mng model, 
  const Rcpp::Function update_fn);

template <class T>
void approx_mcmc::approx_state_summary(T model, 
  const Rcpp::Function update_fn) {
  
  arma::cube Valpha(model.m, model.m, model.n + 1, arma::fill::zeros);
  
  double sum_w = 0;
  arma::mat alphahat_i(model.m, model.n + 1);
  arma::cube Vt_i(model.m, model.m, model.n + 1);
  
  for (unsigned int i = 0; i < n_stored; i++) {
    model.update_model(theta_storage.col(i), update_fn);
    model.approximate_for_is(mode_storage.slice(i));
    model.approx_model.smoother(alphahat_i, Vt_i);
    
    sum_w += count_storage(i);
    arma::mat diff = alphahat_i - alphahat;
    alphahat += count_storage(i) / sum_w * diff; // update E(alpha)
    arma::mat diff2 = (alphahat_i - alphahat).t();
    for (unsigned int t = 0; t < model.n + 1; t++) {
      // update Var(alpha)
      Valpha.slice(t) += count_storage(i) * diff.col(t) * diff2.row(t);
    }
    // update Var(E(alpha))
    Vt += count_storage(i) / sum_w * (Vt_i - Vt);
  }
  Vt += Valpha / sum_w; // Var[E(alpha)] + E[Var(alpha)]
}

// EKF based inference for nonlinear models
void approx_mcmc::ekf_mcmc(ssm_nlg model, const bool end_ram) {
  
  arma::vec theta = model.theta;
  double logprior = model.log_prior_pdf_(theta);
  
  // compute the log-likelihood
  double loglik = model.ekf_loglik();
  if (!arma::is_finite(loglik)) {
    Rcpp::stop("Initial approximate likelihood is not finite.");
  }
  double acceptance_prob = 0.0;
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  
  bool new_value = true;
  unsigned int n_values = 0;
  
  // don't update progress at each iteration
  unsigned int mod = std::max(1U, iter / 50);
  unsigned int ticks = 1;
  if (verbose) {
    Rcpp::Rcout<<"Starting MCMC. Progress:\n";
    Rcpp::Rcout<<"0%   10   20   30   40   50   60   70   80   90   100%\n";
    Rcpp::Rcout<<"|";
  }
  
  for (unsigned int i = 1; i <= iter; i++) {
    
    // sample from standard normal distribution
    arma::vec u(n_par);
    for(unsigned int j = 0; j < n_par; j++) {
      u(j) = normal(model.engine);
    }
    
    // propose new theta
    arma::vec theta_prop = theta + S * u;
    // compute prior
    double logprior_prop = model.log_prior_pdf_(theta_prop);
    
    if (logprior_prop > -std::numeric_limits<double>::infinity() && !std::isnan(logprior_prop)) {
      // update parameters
      model.theta = theta_prop;
      double loglik_prop = model.ekf_loglik();
      
      if (loglik_prop > -std::numeric_limits<double>::infinity() && !std::isnan(loglik_prop)) {
        
        acceptance_prob = std::min(1.0,
          std::exp(loglik_prop - loglik + logprior_prop - logprior));
        
      } else {
        acceptance_prob = 0.0;
      }
      
      if (unif(model.engine) < acceptance_prob) {
        if (i > burnin) {
          acceptance_rate++;
          n_values++;
        }
        loglik = loglik_prop;
        logprior = logprior_prop;
        theta = theta_prop;
        new_value = true;
      }
    } else acceptance_prob = 0.0;
    
    if (i > burnin && n_values % thin == 0) {
      //new block
      if (new_value) {
        approx_loglik_storage(n_stored) = loglik;
        prior_storage(n_stored) = logprior;
        theta_storage.col(n_stored) = theta;
        count_storage(n_stored) = 1;
        n_stored++;
        new_value = false;
      } else {
        count_storage(n_stored - 1)++;
      }
    }
    
    if (!end_ram || i <= burnin) {
      ramcmc::adapt_S(S, u, acceptance_prob, target_acceptance, i, gamma);
    }
    if (i % mod == 0) {
      Rcpp::checkUserInterrupt();
      if (verbose) {
        if (ticks % 5 == 0) {
          Rcpp::Rcout<<"|";
        } else {
          Rcpp::Rcout<<"-";
        }
        ticks++;
      } 
    }
  }
  if (verbose) Rcpp::Rcout<<"\n";
  if (n_stored == 0) Rcpp::stop("No proposals were accepted in MCMC. Check your model.");
  
  trim_storage();
  acceptance_rate /= (iter - burnin);
}

void approx_mcmc::ekf_state_sample(ssm_nlg model, const unsigned int n_threads) {
  
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads) default(shared) firstprivate(model)
{
  model.engine = sitmo::prng_engine(omp_get_thread_num() + 1);
#pragma omp for schedule(static)
#endif
  for (unsigned int i = 0; i < n_stored; i++) {
    model.update_model(theta_storage.col(i));
    model.approximate_by_ekf();
    alpha_storage.slice(i) = model.approx_model.simulate_states(1).slice(0).t();
    
  }
#ifdef _OPENMP
}
#endif

posterior_storage = prior_storage + approx_loglik_storage;
}

void approx_mcmc::ekf_state_summary(ssm_nlg model) {
  
  arma::cube Valpha(model.m, model.m, model.n + 1, arma::fill::zeros);
  
  double sum_w = 0;
  arma::mat alphahat_i(model.m, model.n + 1);
  arma::cube Vt_i(model.m, model.m, model.n + 1);
  
  for (unsigned int i = 0; i < n_stored; i++) {
    
    model.update_model(theta_storage.col(i));
    model.ekf_smoother(alphahat_i, Vt_i);
    
    sum_w += count_storage(i);
    arma::mat diff = alphahat_i - alphahat;
    alphahat += count_storage(i) / sum_w * diff; // update E(alpha)
    arma::mat diff2 = (alphahat_i - alphahat).t();
    for (unsigned int t = 0; t < model.n + 1; t++) {
      // update Var(alpha)
      Valpha.slice(t) += count_storage(i) * diff.col(t) * diff2.row(t);
    }
    // update Var(E(alpha))
    Vt += count_storage(i) / sum_w * (Vt_i - Vt);
  }
  Vt += Valpha / sum_w; // Var[E(alpha)] + E[Var(alpha)]
}


