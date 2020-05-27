#ifdef _OPENMP
#include <omp.h>
#endif
#include <ramcmc.h>
#include "mcmc.h"

#include "distr_consts.h"
#include "filter_smoother.h"
#include "summary.h"

#include "model_ugg_ssm.h"
#include "model_ugg_bsm.h"
#include "model_ugg_ar1.h"

#include "model_ung_ssm.h"
#include "model_ung_bsm.h"
#include "model_ung_ar1.h"
#include "model_ung_svm.h"

#include "model_nlg_ssm.h"
#include "model_sde_ssm.h"
#include "model_mng_ssm.h"
#include "model_lgg_ssm.h"

mcmc::mcmc(
  const unsigned int n_iter, 
  const unsigned int n_burnin,
  const unsigned int n_thin, 
  const unsigned int n, 
  const unsigned int m,
  const double target_acceptance, 
  const double gamma, 
  const arma::mat& S,
  const unsigned int output_type) :
    n_iter(n_iter), n_burnin(n_burnin), n_thin(n_thin),
    n_samples(std::floor(double(n_iter - n_burnin) / double(n_thin))),
    n_par(S.n_rows),
    target_acceptance(target_acceptance), gamma(gamma), n_stored(0),
    posterior_storage(arma::vec(n_samples)),
    theta_storage(arma::mat(n_par, n_samples)),
    count_storage(arma::uvec(n_samples, arma::fill::zeros)),
    alpha_storage(arma::cube((output_type == 1) * n + 1, m, (output_type == 1) * n_samples)), 
    alphahat(arma::mat(m, (output_type == 2) * n + 1, arma::fill::zeros)), 
    Vt(arma::cube(m, m, (output_type == 2) * n + 1, arma::fill::zeros)), S(S),
    acceptance_rate(0.0), output_type(output_type) {
}


void mcmc::trim_storage() {
  theta_storage.resize(n_par, n_stored);
  posterior_storage.resize(n_stored);
  count_storage.resize(n_stored);
  if (output_type == 1)
    alpha_storage.resize(alpha_storage.n_rows, alpha_storage.n_cols, n_stored);
}

template void mcmc::state_posterior(ugg_ssm model, const unsigned int n_threads);
template void mcmc::state_posterior(ugg_bsm model, const unsigned int n_threads);
template void mcmc::state_posterior(ugg_ar1 model, const unsigned int n_threads);
template void mcmc::state_posterior(lgg_ssm model, const unsigned int n_threads);

template <class T>
void mcmc::state_posterior(T model, const unsigned int n_threads) {

  if(n_threads > 1) {
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads) default(shared) firstprivate(model)
{
  model.engine = sitmo::prng_engine(omp_get_thread_num() + 1);
  unsigned int thread_size = unsigned(std::floor(double(n_stored) / n_threads));
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
template void mcmc::state_summary(ugg_ssm& model, arma::mat& alphahat, arma::cube& Vt);
template void mcmc::state_summary(ugg_bsm& model, arma::mat& alphahat, arma::cube& Vt);
template void mcmc::state_summary(ugg_ar1& model, arma::mat& alphahat, arma::cube& Vt);
template void mcmc::state_summary(lgg_ssm& model, arma::mat& alphahat, arma::cube& Vt);

template <class T>
void mcmc::state_summary(T& model, arma::mat& alphahat, arma::cube& Vt) {

  arma::cube Valpha(model.m, model.m, model.n + 1, arma::fill::zeros);

  arma::vec theta = theta_storage.col(0);
  model.update_model(theta);
  model.smoother(alphahat, Vt);

  double sum_w = count_storage(0);
  arma::mat alphahat_i = alphahat;
  arma::cube Vt_i = Vt;

  for (unsigned int i = 1; i < n_stored; i++) {
    arma::vec theta = theta_storage.col(i);
    model.update_model(theta);
    model.smoother(alphahat_i, Vt_i);

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

template void mcmc::state_sampler(ugg_ssm& model, const arma::mat& theta, arma::cube& alpha);
template void mcmc::state_sampler(ugg_bsm& model, const arma::mat& theta, arma::cube& alpha);
template void mcmc::state_sampler(ugg_ar1& model, const arma::mat& theta, arma::cube& alpha);
template void mcmc::state_sampler(lgg_ssm& model, const arma::mat& theta, arma::cube& alpha);
template <class T>

void mcmc::state_sampler(T& model, const arma::mat& theta, arma::cube& alpha) {
  for (unsigned int i = 0; i < theta.n_cols; i++) {
    //arma::vec theta_i = theta.col(i);
    model.update_model(theta.col(i));
    alpha.slice(i) = model.simulate_states(1).slice(0).t();
  }
}



// run MCMC for linear-Gaussian state space model
// target the marginal p(theta | y)
// sample states separately given the posterior sample of theta
template void mcmc::mcmc_gaussian(ugg_ssm& model, const bool end_ram);
template void mcmc::mcmc_gaussian(ugg_bsm& model, const bool end_ram);
template void mcmc::mcmc_gaussian(ugg_ar1& model, const bool end_ram);
template void mcmc::mcmc_gaussian(lgg_ssm& model, const bool end_ram);

template<class T>
void mcmc::mcmc_gaussian(T& model, const bool end_ram) {

  arma::vec theta = model.theta;
  double logprior = model.log_prior_pdf(theta); 
  double loglik = model.log_likelihood();

  if (!std::isfinite(logprior))
    Rcpp::stop("Initial prior probability is not finite.");

  if (!std::isfinite(loglik))
    Rcpp::stop("Initial log-likelihood is not finite.");

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
    double logprior_prop;
    logprior_prop = model.log_prior_pdf(theta_prop); 
    
    if (logprior_prop > -std::numeric_limits<double>::infinity() && !std::isnan(logprior_prop)) {
      // update model based on the proposal
      model.update_model(theta_prop);
      
      // compute log-likelihood with proposed theta
      double loglik_prop = model.log_likelihood();
      //compute the acceptance probability
      // use explicit min(...) as we need this value later
      acceptance_prob =
        std::min(1.0, std::exp(loglik_prop - loglik + logprior_prop - logprior));
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

// run pseudo-marginal MCMC
template void mcmc::pm_mcmc(ung_ssm& model,
  const unsigned int method,
  const unsigned int nsim_states,
  const bool end_ram);
template void mcmc::pm_mcmc(ung_bsm& model,
  const unsigned int method,
  const unsigned int nsim_states,
  const bool end_ram);

template void mcmc::pm_mcmc(ung_ar1& model,
  const unsigned int method,
  const unsigned int nsim_states,
  const bool end_ram);

template void mcmc::pm_mcmc(ung_svm& model,
  const unsigned int method,
  const unsigned int nsim_states,
  const bool end_ram);

template void mcmc::pm_mcmc(nlg_ssm& model,
  const unsigned int method,
  const unsigned int nsim_states,
  const bool end_ram);

template void mcmc::pm_mcmc(mng_ssm& model,
  const unsigned int method,
  const unsigned int nsim_states,
  const bool end_ram);

template void mcmc::pm_mcmc(sde_ssm& model,
  const unsigned int method,
  const unsigned int nsim_states,
  const bool end_ram);

template<class T>
void mcmc::pm_mcmc(
    T& model,
    const unsigned int method,
    const unsigned int nsim_states,
    const bool end_ram) {

  unsigned int m = model.m;
  unsigned int n = model.n;

  // get the current values of theta
  arma::vec theta = model.theta;
  // compute the log[p(theta)]
  double logprior = model.log_prior_pdf(theta);
  if (!arma::is_finite(logprior)) {
    Rcpp::stop("Initial prior probability is not finite.");
  }
  arma::cube alpha(m, n + 1, nsim_states);
  arma::mat weights(nsim_states, n + 1);
  // reduce the space in case of SPDK (does not use indices)
  arma::umat indices(nsim_states * (method != 3) + 1, n * (method != 3) + 1);

  // compute the log-likelihood (unbiased and approximate)
  arma::vec ll = model.log_likelihood(method, nsim_states, alpha, weights, indices);

  if (!std::isfinite(ll(0)))
    Rcpp::stop("Initial log-likelihood is not finite.");

  arma::mat alphahat_i(m, (output_type != 3) * n + 1);
  arma::cube Vt_i(m, m, (output_type != 3) * n + 1);
  arma::cube Valphahat(m, m, (output_type != 3) * n + 1, arma::fill::zeros);
  arma::mat sampled_alpha(m, (output_type != 3) * n + 1);
  if (output_type != 3) {
    sample_or_summarise(
      output_type == 1, method, alpha, weights.col(n), indices,
      sampled_alpha, alphahat_i, Vt_i, model.engine);
  }
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
    double logprior_prop = model.log_prior_pdf(theta_prop);

    if (logprior_prop > -std::numeric_limits<double>::infinity() && !std::isnan(logprior_prop)) {

      // update parameters
      model.update_model(theta_prop);

      // compute the log-likelihood (unbiased and approximate)
      arma::vec ll_prop = model.log_likelihood(method, nsim_states, alpha, weights, indices);

      //compute the acceptance probability for RAM using the approximate ll
      acceptance_prob = std::min(1.0, std::exp(
        ll_prop(1) - ll(1) +
          logprior_prop - logprior));

      //accept
      double log_alpha = ll_prop(0) - ll(0) +
        logprior_prop - logprior;

      //accept
      if (log(unif(model.engine)) < log_alpha) {
        if (i > n_burnin) {
          acceptance_rate++;
          n_values++;
        }
        if (output_type != 3) {
          sample_or_summarise(
            output_type == 1, method, alpha, weights.col(n), indices,
            sampled_alpha, alphahat_i, Vt_i, model.engine);
        }
        ll = ll_prop;
        logprior = logprior_prop;
        theta = theta_prop;
        new_value = true;

      }
    } else acceptance_prob = 0.0;

    // note: thinning does not affect this
    if (i > n_burnin && output_type == 2) {
      arma::mat diff = alphahat_i - alphahat;
      alphahat = (alphahat * (i - n_burnin - 1) + alphahat_i) / (i - n_burnin);
      Vt = (Vt * (i - n_burnin - 1) + Vt_i) / (i - n_burnin);
      for (unsigned int t = 0; t < model.n + 1; t++) {
        Valphahat.slice(t) += diff.col(t) * (alphahat_i.col(t) - alphahat.col(t)).t();
      }
    }

    if (i > n_burnin && n_values % n_thin == 0) {
      //new block
      if (new_value) {
        posterior_storage(n_stored) = logprior + ll(0);
        theta_storage.col(n_stored) = theta;
        count_storage(n_stored) = 1;
        if (output_type == 1) {
          alpha_storage.slice(n_stored) = sampled_alpha.t();
        }
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
  if (output_type == 2) {
    Vt += Valphahat / (n_iter - n_burnin); // Var[E(alpha)] + E[Var(alpha)]
  }
  trim_storage();
  acceptance_rate /= (n_iter - n_burnin);
}

// delayed acceptance pseudo-marginal MCMC
template void mcmc::da_mcmc(ung_ssm& model,
  const unsigned int method,
  const unsigned int nsim_states,
  const bool end_ram);

template void mcmc::da_mcmc(ung_bsm& model,
  const unsigned int method,
  const unsigned int nsim_states,
  const bool end_ram);

template void mcmc::da_mcmc(ung_ar1& model,
  const unsigned int method,
  const unsigned int nsim_states,
  const bool end_ram);

template void mcmc::da_mcmc(ung_svm& model,
  const unsigned int method,
  const unsigned int nsim_states,
  const bool end_ram);

template void mcmc::da_mcmc(nlg_ssm& model,
  const unsigned int method,
  const unsigned int nsim_states,
  const bool end_ram);

template void mcmc::da_mcmc(mng_ssm& model,
  const unsigned int method,
  const unsigned int nsim_states,
  const bool end_ram);

template void mcmc::da_mcmc(sde_ssm& model,
  const unsigned int method,
  const unsigned int nsim_states,
  const bool end_ram);

template<class T>
void mcmc::da_mcmc(T& model,
  const unsigned int method,
  const unsigned int nsim_states,
  const bool end_ram) {

  unsigned int m = model.m;
  unsigned int n = model.n;

  // get the current values of theta
  arma::vec theta = model.theta;
  // compute the log[p(theta)]
  double logprior = model.log_prior_pdf(theta);
  if (!arma::is_finite(logprior)) {
    Rcpp::stop("Initial prior probability is not finite.");
  }
  arma::cube alpha(m, n + 1, nsim_states);
  arma::mat weights(nsim_states, n + 1);
  // reduce the space in case of SPDK (does not use indices)
  arma::umat indices(nsim_states * (method != 3) + 1, n * (method != 3) + 1);

  // compute the log-likelihood (unbiased and approximate)
  arma::vec ll =
    model.log_likelihood(method, nsim_states, alpha, weights, indices);

  if (!std::isfinite(ll(0)))
    Rcpp::stop("Initial log-likelihood is not finite.");

  arma::mat alphahat_i(m, (output_type != 3) * n + 1);
  arma::cube Vt_i(m, m, (output_type != 3) * n + 1);
  arma::cube Valphahat(m, m, (output_type != 3) * n + 1, arma::fill::zeros);
  arma::mat sampled_alpha(m, (output_type != 3) * n + 1);
  if (output_type != 3) {
    sample_or_summarise(
      output_type == 1, method, alpha, weights.col(n), indices,
      sampled_alpha, alphahat_i, Vt_i, model.engine);
  }
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
    double logprior_prop = model.log_prior_pdf(theta_prop);

    if (logprior_prop > -std::numeric_limits<double>::infinity() && !std::isnan(logprior_prop)) {

      // update parameters
      model.update_model(theta_prop);

      // compute the approximate log-likelihood (nsim_states = 0)
      arma::vec ll_prop = model.log_likelihood(method, 0, alpha, weights, indices);

      // initial acceptance probability, also used in RAM
      acceptance_prob = std::min(1.0, std::exp(ll_prop(1) - ll(1) +
        logprior_prop - logprior));

      // initial acceptance
      if (unif(model.engine) < acceptance_prob) {

        // compute the unbiased log-likelihood estimate
        ll_prop = model.log_likelihood(method, nsim_states, alpha, weights, indices);

        // second stage acceptance log-probability
        double log_alpha = ll_prop(0) + ll(1) - ll(0) - ll_prop(1);

        if (log(unif(model.engine)) < log_alpha) {
          if (i > n_burnin) {
            acceptance_rate++;
            n_values++;
          }
          if (output_type != 3) {
            sample_or_summarise(
              output_type == 1, method, alpha, weights.col(n), indices,
              sampled_alpha, alphahat_i, Vt_i, model.engine);
          }
          ll = ll_prop;
          logprior = logprior_prop;
          theta = theta_prop;
          new_value = true;
        }

      }
    } else acceptance_prob = 0.0;

    // note: thinning does not affect this
    if (i > n_burnin && output_type == 2) {
      arma::mat diff = alphahat_i - alphahat;
      alphahat = (alphahat * (i - n_burnin - 1) + alphahat_i) / (i - n_burnin);
      Vt = (Vt * (i - n_burnin - 1) + Vt_i) / (i - n_burnin);
      for (unsigned int t = 0; t < model.n + 1; t++) {
        Valphahat.slice(t) += diff.col(t) * (alphahat_i.col(t) - alphahat.col(t)).t();
      }
    }

    if (i > n_burnin && n_values % n_thin == 0) {
      //new block
      if (new_value) {
        posterior_storage(n_stored) = logprior + ll(0);
        theta_storage.col(n_stored) = theta;
        count_storage(n_stored) = 1;
        if (output_type == 1) {
          alpha_storage.slice(n_stored) = sampled_alpha.t();
        }
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
  if (output_type == 2) {
    Vt += Valphahat / (n_iter - n_burnin); // Var[E(alpha)] + E[Var(alpha)]
  }
  trim_storage();
  acceptance_rate /= (n_iter - n_burnin);
}
