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

approx_mcmc::approx_mcmc(const unsigned int iter,
  const unsigned int burnin, const unsigned int thin, const unsigned int n,
  const unsigned int m, const unsigned int p, const double target_acceptance, 
  const double gamma, const arma::mat& S, const unsigned int output_type, 
  const bool store_modes) :
  mcmc(iter, burnin, thin, n, m,
    target_acceptance, gamma, S, output_type),
    weight_storage(arma::vec(n_samples, arma::fill::zeros)),
    mode_storage(arma::cube(p, n, n_samples * store_modes)),
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
    arma::cube expanded_mode = rep_cube(mode_storage, count_storage);
    mode_storage.set_size(mode_storage.n_rows, mode_storage.n_cols, n_stored);
    mode_storage = expanded_mode;
  }
}

// run approximate MCMC for
// non-linear and/or non-Gaussian state space model with linear-Gaussian states
template void approx_mcmc::amcmc(ssm_ung model, const bool end_ram);
template void approx_mcmc::amcmc(bsm_ng model, const bool end_ram);
template void approx_mcmc::amcmc(svm model, const bool end_ram);
template void approx_mcmc::amcmc(ar1_ng model, const bool end_ram);
template void approx_mcmc::amcmc(ssm_nlg model, const bool end_ram);
template void approx_mcmc::amcmc(ssm_mng model, const bool end_ram);

template<class T>
void approx_mcmc::amcmc(T model, const bool end_ram) {
  
  // get the current values of theta
  arma::vec theta = model.theta;
  model.update_model(theta); // just in case
  // compute the log[p(theta)]
  double logprior = model.log_prior_pdf(theta);
  if (!arma::is_finite(logprior)) {
    Rcpp::stop("Initial prior probability is not finite.");
  }
  // placeholders
  arma::cube alpha(1, 1, 1);
  arma::mat weights(1, 1);
  arma::umat indices(1, 1);
  
  // compute the approximate log-likelihood
  arma::vec ll = model.log_likelihood(1, 0, alpha, weights, indices);
  double approx_loglik = ll(0);
  if (!std::isfinite(approx_loglik))
    Rcpp::stop("Initial log-likelihood is not finite.");
  
  
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  arma::mat mode(model.p, model.n);
  mode = model.mode_estimate;
  
  bool new_value = true;
  unsigned int n_values = 0;
  double acceptance_prob = 0.0;
  
  for (unsigned int i = 1; i <= iter; i++) {
    
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
      
      arma::vec ll = model.log_likelihood(1, 0, alpha, weights, indices);
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
  }
  
  trim_storage();
  acceptance_rate /= (iter - burnin);
}

// approximate MCMC

template void approx_mcmc::is_correction_psi(ssm_ung model, const unsigned int nsim,
  const unsigned int is_type, const unsigned int n_threads);
template void approx_mcmc::is_correction_psi(bsm_ng model, const unsigned int nsim,
  const unsigned int is_type, const unsigned int n_threads);
template void approx_mcmc::is_correction_psi(svm model, const unsigned int nsim,
  const unsigned int is_type, const unsigned int n_threads);
template void approx_mcmc::is_correction_psi(ar1_ng model, const unsigned int nsim,
  const unsigned int is_type, const unsigned int n_threads);
template void approx_mcmc::is_correction_psi(ssm_mng model, const unsigned int nsim,
  const unsigned int is_type, const unsigned int n_threads);
template void approx_mcmc::is_correction_psi(ssm_nlg model, const unsigned int nsim,
  const unsigned int is_type, const unsigned int n_threads);

template <class T>
void approx_mcmc::is_correction_psi(T model, const unsigned int nsim,
  const unsigned int is_type, const unsigned int n_threads) {
  
  arma::cube Valpha(model.m, model.m, model.n + 1, arma::fill::zeros);
  double sum_w = 0.0;
  
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads) default(shared) firstprivate(model)
{
  
  model.engine = sitmo::prng_engine(omp_get_thread_num() + 1);
  
  
#pragma omp for schedule(dynamic)
  for (unsigned int i = 0; i < theta_storage.n_cols; i++) {
    
    // needs critical as updating might call R function
#pragma omp critical
{
  model.update_model(theta_storage.col(i));
}
model.approximate_for_is(mode_storage.slice(i));

unsigned int nsimc = nsim;
if (is_type == 1) {
  nsimc *= count_storage(i);
}
arma::cube alpha_i(model.m, model.n + 1, nsimc);
arma::mat weights_i(nsimc, model.n + 1);
arma::umat indices(nsimc, model.n);

double loglik = model.psi_filter(nsim, alpha_i, weights_i, indices);
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

for (unsigned int i = 0; i < theta_storage.n_cols; i++) {
  
  model.update_model(theta_storage.col(i));
  model.approximate_for_is(mode_storage.slice(i));
  
  unsigned int nsimc = nsim;
  if (is_type == 1) {
    nsimc *= count_storage(i);
  }
  arma::cube alpha_i(model.m, model.n + 1, nsimc);
  arma::mat weights_i(nsimc, model.n + 1);
  arma::umat indices(nsimc, model.n);
  
  double loglik = model.psi_filter(nsim, alpha_i, weights_i, indices);
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

template void approx_mcmc::is_correction_bsf(ssm_ung model,
  const unsigned int nsim, const unsigned int is_type,
  const unsigned int n_threads);
template void approx_mcmc::is_correction_bsf(bsm_ng model,
  const unsigned int nsim, const unsigned int is_type,
  const unsigned int n_threads);
template void approx_mcmc::is_correction_bsf(svm model,
  const unsigned int nsim, const unsigned int is_type,
  const unsigned int n_threads);
template void approx_mcmc::is_correction_bsf(ar1_ng model,
  const unsigned int nsim, const unsigned int is_type,
  const unsigned int n_threads);
template void approx_mcmc::is_correction_bsf(ssm_mng model,
  const unsigned int nsim, const unsigned int is_type,
  const unsigned int n_threads);
template void approx_mcmc::is_correction_bsf(ssm_nlg model,
  const unsigned int nsim, const unsigned int is_type,
  const unsigned int n_threads);

template <class T>
void approx_mcmc::is_correction_bsf(T model, const unsigned int nsim,
  const unsigned int is_type, const unsigned int n_threads) {
  
  arma::cube Valpha(model.m, model.m, model.n + 1, arma::fill::zeros);
  double sum_w = 0.0;
  
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads) default(shared) firstprivate(model)
{
  
  model.engine = sitmo::prng_engine(omp_get_thread_num() + 1);
  
#pragma omp for schedule(dynamic)
  for (unsigned int i = 0; i < theta_storage.n_cols; i++) {
    
    // needs critical as updating might call R function
#pragma omp critical
{
  model.update_model(theta_storage.col(i));
}

unsigned int nsimc = nsim;
if (is_type == 1) {
  nsimc *= count_storage(i);
}

arma::cube alpha_i(model.m, model.n + 1, nsimc);
arma::mat weights_i(nsimc, model.n + 1);
arma::umat indices(nsimc, model.n);

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
  
  // needs critical as updating might call R function
#pragma omp critical
{
  model.update_model(theta_storage.col(i));
}

unsigned int nsimc = nsim;
if (is_type == 1) {
  nsimc *= count_storage(i);
}

arma::cube alpha_i(model.m, model.n + 1, nsimc);
arma::mat weights_i(nsimc, model.n + 1);
arma::umat indices(nsimc, model.n);

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

template void approx_mcmc::is_correction_spdk(ssm_ung model, const unsigned int nsim,
  unsigned int is_type, const unsigned int n_threads);
template void approx_mcmc::is_correction_spdk(bsm_ng model, const unsigned int nsim,
  unsigned int is_type, const unsigned int n_threads);
template void approx_mcmc::is_correction_spdk(svm model, const unsigned int nsim,
  unsigned int is_type, const unsigned int n_threads);
template void approx_mcmc::is_correction_spdk(ar1_ng model, const unsigned int nsim,
  unsigned int is_type, const unsigned int n_threads);


template <class T>
void approx_mcmc::is_correction_spdk(T model, const unsigned int nsim,
  const unsigned int is_type, const unsigned int n_threads) {
  
  arma::cube Valpha(model.m, model.m, model.n + 1, arma::fill::zeros);
  double sum_w = 0.0;
  
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads) default(shared) firstprivate(model)
{
  
  model.engine = sitmo::prng_engine(omp_get_thread_num() + 1);
  
  
#pragma omp for schedule(dynamic)
  for (unsigned int i = 0; i < theta_storage.n_cols; i++) {
    
    // needs critical as updating might call R function
#pragma omp critical
{
  model.update_model(theta_storage.col(i));
}
model.approximate_for_is(mode_storage.slice(i));

unsigned int nsimc = nsim;
if (is_type == 1) {
  nsimc *= count_storage(i);
}

arma::cube alpha_i = model.approx_model.simulate_states(nsimc, true);
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
  
  model.approximate_for_is(mode_storage.slice(i));
  
  unsigned int nsimc = nsim;
  if (is_type == 1) {
    nsimc *= count_storage(i);
  }
  
  arma::cube alpha_i = model.approx_model.simulate_states(nsimc, true);
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

template void approx_mcmc::approx_state_posterior(ssm_ung model, const unsigned int n_threads);
template void approx_mcmc::approx_state_posterior(bsm_ng model, const unsigned int n_threads);
template void approx_mcmc::approx_state_posterior(svm model, const unsigned int n_threads);
template void approx_mcmc::approx_state_posterior(ar1_ng model, const unsigned int n_threads);
template void approx_mcmc::approx_state_posterior(ssm_nlg model, const unsigned int n_threads);
template void approx_mcmc::approx_state_posterior(ssm_mng model, const unsigned int n_threads);

template <class T>
void approx_mcmc::approx_state_posterior(T model, const unsigned int n_threads) {
  
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads) default(shared) firstprivate(model)
{
  
  model.engine = sitmo::prng_engine(omp_get_thread_num() + 1);
  
  
#pragma omp for schedule(dynamic)
  for (unsigned int i = 0; i < theta_storage.n_cols; i++) {
#pragma omp critical
{
  model.update_model(theta_storage.col(i));
}
    model.approximate_for_is(mode_storage.slice(i));
    alpha_storage.slice(i) = model.approx_model.simulate_states(1).slice(0).t();
  }
}
#else
for (unsigned int i = 0; i < theta_storage.n_cols; i++) {
  model.update_model(theta_storage.col(i));
  model.approximate_for_is(mode_storage.slice(i));
  alpha_storage.slice(i) = approx_model.simulate_states(1).slice(0).t();
}
#endif

}

template void approx_mcmc::approx_state_summary(ssm_ung model);
template void approx_mcmc::approx_state_summary(bsm_ng model);
template void approx_mcmc::approx_state_summary(svm model);
template void approx_mcmc::approx_state_summary(ar1_ng model);
template void approx_mcmc::approx_state_summary(ssm_nlg model);
template void approx_mcmc::approx_state_summary(ssm_mng model);

template <class T>
void approx_mcmc::approx_state_summary(T model) {
  
  arma::cube Valpha(model.m, model.m, model.n + 1, arma::fill::zeros);
  
  model.update_model(theta_storage.col(0));
  model.approximate_for_is(mode_storage.slice(0));
  model.approx_model.smoother(alphahat, Vt);
  
  double sum_w = count_storage(0);
  arma::mat alphahat_i = alphahat;
  arma::cube Vt_i = Vt;
  
  for (unsigned int i = 1; i < n_stored; i++) {
    model.update_model(theta_storage.col(i));
    model.approximate_for_is(mode_storage.slice(i));
    model.approx_model.smoother(alphahat_i, Vt_i);
    
    arma::mat diff = alphahat_i - alphahat;
    double tmp = count_storage(i) + sum_w;
    alphahat = (alphahat * sum_w + alphahat_i * count_storage(i)) / tmp;
    
    for (unsigned int t = 0; t < model.n + 1; t++) {
      Valpha.slice(t) += diff.col(t) * (alphahat_i.col(t) - alphahat.col(t)).t();
    }
    Vt = (Vt * sum_w + Vt_i * count_storage(i)) / tmp;
    sum_w = tmp;
  }
  Vt += Valpha / sum_w; // Var[E(alpha)] + E[Var(alpha)]
}

void approx_mcmc::ekf_mcmc(ssm_nlg model, const bool end_ram) {
  
  
  arma::vec theta = model.theta;
  double logprior = model.log_prior_pdf(theta);
  
  model.update_model(theta); // just in case
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
  
  for (unsigned int i = 1; i <= iter; i++) {
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
  }
  
  trim_storage();
  acceptance_rate /= (iter - burnin);
}


void approx_mcmc::ekf_state_sample(ssm_nlg model, const unsigned int n_threads) {
  
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads) default(shared) firstprivate(model)
{
  model.engine = sitmo::prng_engine(omp_get_thread_num() + 1);
  
#pragma omp for schedule(dynamic)
  for (unsigned int i = 0; i < theta_storage.n_cols; i++) {
#pragma omp critical
{
  model.update_model(theta_storage.col(i));
}  
    model.approximate_for_is(mode_storage.slice(i));
    alpha_storage.slice(i) = model.approx_model.simulate_states(1).slice(0).t();
    
  }
}
#else

ssm_mlg approx_model(model.y, Z, H, T, R, a1, P1, arma::cube(0,0,0),
  arma::mat(0,0), D, C, model.seed);

#pragma omp for schedule(dynamic)
for (unsigned int i = 0; i < theta_storage.n_cols; i++) {
  
  model.update(theta_storage.col(i));
  model.approximate_for_is(mode_storage.slice(i));
  alpha_storage.slice(i) = model.approx_model.simulate_states(1).slice(0).t();
  
}
#endif

posterior_storage = prior_storage + approx_loglik_storage;
}

void approx_mcmc::ekf_state_summary(ssm_nlg model) {
  
  // first iteration
  model.update_model(theta_storage.col(0));
  model.ekf_smoother(alphahat, Vt);
  
  double sum_w = count_storage(0);
  arma::mat alphahat_i = alphahat;
  arma::cube Vt_i = Vt;
  
  arma::cube Valpha(model.m, model.m, model.n + 1, arma::fill::zeros);
  
  for (unsigned int i = 1; i < theta_storage.n_cols; i++) {
    
    model.update_model(theta_storage.col(i));
    model.ekf_smoother(alphahat_i, Vt_i);
    arma::mat diff = alphahat_i - alphahat;
    double tmp = count_storage(i) + sum_w;
    alphahat = (alphahat * sum_w + alphahat_i * count_storage(i)) / tmp;
    
    for (unsigned int t = 0; t < model.n + 1; t++) {
      Valpha.slice(t) += diff.col(t) * (alphahat_i.col(t) - alphahat.col(t)).t();
    }
    Vt = (Vt * sum_w + Vt_i * count_storage(i)) / tmp;
    sum_w = tmp;
  }
  Vt += Valpha / sum_w; // Var[E(alpha)] + E[Var(alpha)]
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
  
  for (unsigned int i = 1; i <= iter; i++) {
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
  }
  
  trim_storage();
  acceptance_rate /= (iter - burnin);
}

template<>
void approx_mcmc::is_correction_bsf<ssm_sde>(ssm_sde model, const unsigned int nsim,
  const unsigned int is_type, const unsigned int n_threads) {
  
  arma::cube Valpha(1, 1, model.n + 1, arma::fill::zeros);
  double sum_w = 0.0;
  
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads) default(shared) firstprivate(model)
{
  
  model.engine = sitmo::prng_engine(omp_get_thread_num() + 1);
  model.coarse_engine = sitmo::prng_engine(omp_get_thread_num() + n_threads + 1);
  
#pragma omp for schedule(dynamic)
  for (unsigned int i = 0; i < n_stored; i++) {
    model.theta = theta_storage.col(i);
    unsigned int nsimc = nsim;
    if (is_type == 1) {
      nsimc *= count_storage(i);
    }
    arma::cube alpha_i(1, model.n + 1, nsimc);
    arma::mat weights_i(nsimc, model.n + 1);
    arma::umat indices(nsimc, model.n);
    double loglik = model.bsf_filter(nsim, model.L_f, alpha_i, weights_i, indices);
    weight_storage(i) = std::exp(loglik - approx_loglik_storage(i));
    
    if (output_type != 3) {
      filter_smoother(alpha_i, indices);
      arma::vec w = weights_i.col(model.n);
      if (output_type == 1) {
        std::discrete_distribution<unsigned int> sample(w.begin(), w.end());
        alpha_storage.slice(i) = alpha_i.slice(sample(model.engine)).t();
      } else {
        arma::mat alphahat_i(1, model.n + 1);
        arma::cube Vt_i(1, 1, model.n + 1);
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
for (unsigned int i = 0; i < n_stored; i++) {
  model.theta = theta_storage.col(i);
  unsigned int nsimc = nsim;
  if (is_type == 1) {
    nsimc *= count_storage(i);
  }
  arma::cube alpha_i(1, model.n + 1, nsimc);
  arma::mat weights_i(nsimc, model.n + 1);
  arma::umat indices(nsimc, model.n);
  double loglik = model.bsf_filter(nsim, model.L_f, alpha_i, weights_i, indices);
  weight_storage(i) = std::exp(loglik - approx_loglik_storage(i));
  
  if (output_type != 3) {
    filter_smoother(alpha_i, indices);
    arma::vec w = weights_i.col(model.n);
    if (output_type == 1) {
      std::discrete_distribution<unsigned int> sample(w.begin(), w.end());
      alpha_storage.slice(i) = alpha_i.slice(sample(model.engine)).t();
    } else {
      arma::mat alphahat_i(1, model.n + 1);
      arma::cube Vt_i(1, 1, model.n + 1);
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
posterior_storage = prior_storage + approx_loglik_storage + arma::log(weight_storage);
}

