#ifdef _OPENMP
#include <omp.h>
#endif
#include <ramcmc.h>
#include "mcmc.h"

#include "distr_consts.h"
#include "filter_smoother.h"
#include "summary.h"

#include "model_ssm_mlg.h"
#include "model_ssm_ulg.h"
#include "model_bsm_lg.h"
#include "model_ar1_lg.h"

#include "model_ssm_mng.h"
#include "model_ssm_ung.h"
#include "model_bsm_ng.h"
#include "model_ar1_ng.h"
#include "model_svm.h"

#include "model_ssm_nlg.h"
#include "model_ssm_sde.h"

#include "parset_lg.h"

// used as placeholder
Rcpp::Environment pkg = Rcpp::Environment::namespace_env("bssm");

Rcpp::Function default_update_fn = pkg["default_update_fn"];
Rcpp::Function default_prior_fn = pkg["default_prior_fn"];

mcmc::mcmc(
  const unsigned int iter, 
  const unsigned int burnin,
  const unsigned int thin, 
  const unsigned int n, 
  const unsigned int m,
  const double target_acceptance, 
  const double gamma, 
  const arma::mat& S,
  const unsigned int output_type, const bool verbose) :
  iter(iter), burnin(burnin), thin(thin),
  n_samples(std::floor(double(iter - burnin) / double(thin))),
  n_par(S.n_rows),
  target_acceptance(target_acceptance), gamma(gamma), n_stored(0),
  posterior_storage(arma::vec(n_samples, arma::fill::zeros)),
  theta_storage(arma::mat(n_par, n_samples, arma::fill::zeros)),
  count_storage(arma::uvec(n_samples, arma::fill::zeros)),
  alpha_storage(arma::cube((output_type == 1) * n + 1, m, (output_type == 1) * 
    n_samples, arma::fill::zeros)), 
    alphahat(arma::mat(m, (output_type == 2) * n + 1, arma::fill::zeros)), 
    Vt(arma::cube(m, m, (output_type == 2) * n + 1, arma::fill::zeros)), S(S),
    acceptance_rate(0.0), output_type(output_type), verbose(verbose) {
}


void mcmc::trim_storage() {
  theta_storage.resize(n_par, n_stored);
  posterior_storage.resize(n_stored);
  count_storage.resize(n_stored);
  if (output_type == 1)
    alpha_storage.resize(alpha_storage.n_rows, alpha_storage.n_cols, n_stored);
}

// for circumventing calls to R during parallel runs

template void mcmc::state_posterior(bsm_lg model, const unsigned int n_threads, 
  const Rcpp::Function update_fn);
template void mcmc::state_posterior(ar1_lg model, const unsigned int n_threads,
  const Rcpp::Function update_fn);

template <class T>
void mcmc::state_posterior(T model, const unsigned int n_threads, 
  const Rcpp::Function update_fn) {
  
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads) default(shared) firstprivate(model)
{
  model.engine = sitmo::prng_engine(omp_get_thread_num() + 1);
#pragma omp for schedule(static)
#endif
  for (unsigned int i = 0; i < n_stored; i++) {
    model.update_model(theta_storage.col(i));
    alpha_storage.slice(i) = model.simulate_states(1).slice(0).t();
  }
#ifdef _OPENMP
}
#endif

}

template<>
void mcmc::state_posterior(ssm_ulg model, const unsigned int n_threads, 
  const Rcpp::Function update_fn) {
  
  
#ifdef _OPENMP
  parset_ulg pars(model, theta_storage, update_fn);
#pragma omp parallel num_threads(n_threads) default(shared) firstprivate(model)
{
  model.engine = sitmo::prng_engine(omp_get_thread_num() + 1);
#pragma omp for schedule(static)
  for (unsigned int i = 0; i < n_stored; i++) {
    pars.update(model, i);
    alpha_storage.slice(i) = model.simulate_states(1).slice(0).t();
  }
}
#else
for (unsigned int i = 0; i < n_stored; i++) {
  model.update_model(theta_storage.col(i), update_fn);
  alpha_storage.slice(i) = model.simulate_states(1).slice(0).t();
}
#endif
}

template<>
void mcmc::state_posterior(ssm_mlg model, const unsigned int n_threads,
  const Rcpp::Function update_fn) {
  
  
#ifdef _OPENMP
  parset_mlg pars(model, theta_storage, update_fn);
#pragma omp parallel num_threads(n_threads) default(shared) firstprivate(model)
{
  model.engine = sitmo::prng_engine(omp_get_thread_num() + 1);
#pragma omp for schedule(static)
  for (unsigned int i = 0; i < n_stored; i++) {
    pars.update(model, i);
    alpha_storage.slice(i) = model.simulate_states(1).slice(0).t();
  }
}
#else
for (unsigned int i = 0; i < n_stored; i++) {
  model.update_model(theta_storage.col(i), update_fn);
  alpha_storage.slice(i) = model.simulate_states(1).slice(0).t();
}
#endif
}

template void mcmc::state_summary(ssm_ulg model, 
  const Rcpp::Function update_fn);
template void mcmc::state_summary(bsm_lg model, 
  const Rcpp::Function update_fn);
template void mcmc::state_summary(ar1_lg model, 
  const Rcpp::Function update_fn);
template void mcmc::state_summary(ssm_mlg model, 
  const Rcpp::Function update_fn);

template <class T>
void mcmc::state_summary(T model, 
  const Rcpp::Function update_fn) {
  
  arma::cube Valpha(model.m, model.m, model.n + 1, arma::fill::zeros);
  
  double sum_w = 0;
  arma::mat alphahat_i(model.m, model.n + 1);
  arma::cube Vt_i(model.m, model.m, model.n + 1);
  
  for (unsigned int i = 0; i < n_stored; i++) {
    
    model.update_model(theta_storage.col(i), update_fn);
    model.smoother(alphahat_i, Vt_i);
    
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


// run MCMC for linear-Gaussian state space model
// target the marginal p(theta | y)
// sample states separately given the posterior sample of theta
template void mcmc::mcmc_gaussian(ssm_ulg model, const bool end_ram, 
  const Rcpp::Function update_fn, const Rcpp::Function prior_fn);
template void mcmc::mcmc_gaussian(bsm_lg model, const bool end_ram, 
  const Rcpp::Function update_fn, const Rcpp::Function prior_fn);
template void mcmc::mcmc_gaussian(ar1_lg model, const bool end_ram, 
  const Rcpp::Function update_fn, const Rcpp::Function prior_fn);
template void mcmc::mcmc_gaussian(ssm_mlg model, const bool end_ram, 
  const Rcpp::Function update_fn, const Rcpp::Function prior_fn);

template<class T>
void mcmc::mcmc_gaussian(T model, const bool end_ram, 
  const Rcpp::Function update_fn, const Rcpp::Function prior_fn) {
  
  arma::vec theta = model.theta;
  model.update_model(theta, update_fn); // just in case
  double logprior = model.log_prior_pdf(theta, prior_fn); 
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
    double logprior_prop;
    logprior_prop = model.log_prior_pdf(theta_prop, prior_fn); 
    
    if (logprior_prop > -std::numeric_limits<double>::infinity() && 
      !std::isnan(logprior_prop)) {
      
      // update model based on the proposal
      model.update_model(theta_prop, update_fn);
      
      // compute log-likelihood with proposed theta
      double loglik_prop = model.log_likelihood();
      
      //compute the acceptance probability
      // use explicit min(...) as we need this value later
      acceptance_prob =
        std::min(1.0, std::exp(loglik_prop - loglik + logprior_prop - logprior));
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
        posterior_storage(n_stored) = logprior + loglik;
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
  if(n_stored == 0) Rcpp::stop("No proposals were accepted in MCMC. Check your model.");
  
  trim_storage();
  acceptance_rate /= (iter - burnin);
  
}

// run pseudo-marginal MCMC
template void mcmc::pm_mcmc(ssm_ung model,
  const unsigned int method,
  const unsigned int nsim,
  const bool end_ram, 
  const Rcpp::Function update_fn, const Rcpp::Function prior_fn);

template void mcmc::pm_mcmc(bsm_ng model,
  const unsigned int method,
  const unsigned int nsim,
  const bool end_ram, 
  const Rcpp::Function update_fn, const Rcpp::Function prior_fn);

template void mcmc::pm_mcmc(ar1_ng model,
  const unsigned int method,
  const unsigned int nsim,
  const bool end_ram, 
  const Rcpp::Function update_fn, const Rcpp::Function prior_fn);

template void mcmc::pm_mcmc(svm model,
  const unsigned int method,
  const unsigned int nsim,
  const bool end_ram, 
  const Rcpp::Function update_fn, const Rcpp::Function prior_fn);

template void mcmc::pm_mcmc(ssm_nlg model,
  const unsigned int method,
  const unsigned int nsim,
  const bool end_ram, 
  const Rcpp::Function update_fn, const Rcpp::Function prior_fn);

template void mcmc::pm_mcmc(ssm_mng model,
  const unsigned int method,
  const unsigned int nsim,
  const bool end_ram, 
  const Rcpp::Function update_fn, const Rcpp::Function prior_fn);


template<class T>
void mcmc::pm_mcmc(
    T model,
    const unsigned int method,
    const unsigned int nsim,
    const bool end_ram, 
    const Rcpp::Function update_fn, const Rcpp::Function prior_fn) {
  
  unsigned int m = model.m;
  unsigned int n = model.n;
  
  // get the current values of theta
  arma::vec theta = model.theta;
  model.update_model(theta, update_fn); // just in case
  // compute the log[p(theta)]
  double logprior = model.log_prior_pdf(theta, prior_fn);
  if (!arma::is_finite(logprior)) {
    Rcpp::stop("Initial prior probability is not finite.");
  }
  arma::cube alpha(m, n + 1, nsim);
  arma::mat weights(nsim, n + 1);
  // reduce the space in case of SPDK (does not use indices)
  arma::umat indices((nsim - 1) * (method != 3) + 1, n * (method != 3) + 1);
  
  // compute the log-likelihood (unbiased and approximate)
  arma::vec ll = model.log_likelihood(method, nsim, alpha, weights, indices);
  
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
      
      // compute the log-likelihood (unbiased and approximate)
      arma::vec ll_prop = model.log_likelihood(method, nsim, alpha, weights, indices);
      
      //compute the acceptance probability for RAM using the approximate ll
      acceptance_prob = std::min(1.0, std::exp(
        ll_prop(1) - ll(1) +
          logprior_prop - logprior));
      
      //accept
      double log_alpha = ll_prop(0) - ll(0) +
        logprior_prop - logprior;
      
      //accept
      if (log(unif(model.engine)) < log_alpha) {
        if (i > burnin) {
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
    if (i > burnin && output_type == 2) {
      arma::mat diff = alphahat_i - alphahat;
      alphahat = (alphahat * (i - burnin - 1) + alphahat_i) / (i - burnin);
      Vt = (Vt * (i - burnin - 1) + Vt_i) / (i - burnin);
      for (unsigned int t = 0; t < model.n + 1; t++) {
        Valphahat.slice(t) += diff.col(t) * (alphahat_i.col(t) - alphahat.col(t)).t();
      }
    }
    
    if (i > burnin && n_values % thin == 0) {
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
  if(n_stored == 0) Rcpp::stop("No proposals were accepted in MCMC. Check your model.");
  
  if (output_type == 2) {
    Vt += Valphahat / (iter - burnin); // Var[E(alpha)] + E[Var(alpha)]
  }
  
  trim_storage();
  acceptance_rate /= (iter - burnin);
}

// delayed acceptance pseudo-marginal MCMC
template void mcmc::da_mcmc(ssm_ung model,
  const unsigned int method,
  const unsigned int nsim,
  const bool end_ram, 
  const Rcpp::Function update_fn, const Rcpp::Function prior_fn);

template void mcmc::da_mcmc(bsm_ng model,
  const unsigned int method,
  const unsigned int nsim,
  const bool end_ram, 
  const Rcpp::Function update_fn, const Rcpp::Function prior_fn);

template void mcmc::da_mcmc(ar1_ng model,
  const unsigned int method,
  const unsigned int nsim,
  const bool end_ram, 
  const Rcpp::Function update_fn, const Rcpp::Function prior_fn);

template void mcmc::da_mcmc(svm model,
  const unsigned int method,
  const unsigned int nsim,
  const bool end_ram, 
  const Rcpp::Function update_fn, const Rcpp::Function prior_fn);

template void mcmc::da_mcmc(ssm_nlg model,
  const unsigned int method,
  const unsigned int nsim,
  const bool end_ram, 
  const Rcpp::Function update_fn, const Rcpp::Function prior_fn);

template void mcmc::da_mcmc(ssm_mng model,
  const unsigned int method,
  const unsigned int nsim,
  const bool end_ram, 
  const Rcpp::Function update_fn, const Rcpp::Function prior_fn);

template<class T>
void mcmc::da_mcmc(T model,
  const unsigned int method,
  const unsigned int nsim,
  const bool end_ram, 
  const Rcpp::Function update_fn, const Rcpp::Function prior_fn) {
  
  unsigned int m = model.m;
  unsigned int n = model.n;
  
  // get the current values of theta
  arma::vec theta = model.theta;
  model.update_model(theta, update_fn); // just in case
  // compute the log[p(theta)]
  double logprior = model.log_prior_pdf(theta, prior_fn);
  if (!arma::is_finite(logprior)) {
    Rcpp::stop("Initial prior probability is not finite.");
  }
  arma::cube alpha(m, n + 1, nsim);
  arma::mat weights(nsim, n + 1);
  // reduce the space in case of SPDK (does not use indices)
  arma::umat indices((nsim - 1) * (method != 3) + 1, n * (method != 3) + 1);
  
  // compute the log-likelihood (unbiased and approximate)
  arma::vec ll =
    model.log_likelihood(method, nsim, alpha, weights, indices);
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
      // compute the approximate log-likelihood (nsim = 0)
      arma::vec ll_prop = model.log_likelihood(method, 0, alpha, weights, indices);
      
      // initial acceptance probability, also used in RAM
      acceptance_prob = std::min(1.0, std::exp(ll_prop(1) - ll(1) +
        logprior_prop - logprior));
      // initial acceptance
      if (unif(model.engine) < acceptance_prob) {
        
        // compute the unbiased log-likelihood estimate
        ll_prop = model.log_likelihood(method, nsim, alpha, weights, indices);
        
        // second stage acceptance log-probability
        double log_alpha = ll_prop(0) + ll(1) - ll(0) - ll_prop(1);
        
        if (log(unif(model.engine)) < log_alpha) {
          if (i > burnin) {
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
    if (i > burnin && output_type == 2) {
      arma::mat diff = alphahat_i - alphahat;
      alphahat = (alphahat * (i - burnin - 1) + alphahat_i) / (i - burnin);
      Vt = (Vt * (i - burnin - 1) + Vt_i) / (i - burnin);
      for (unsigned int t = 0; t < model.n + 1; t++) {
        Valphahat.slice(t) += diff.col(t) * (alphahat_i.col(t) - alphahat.col(t)).t();
      }
    }
    
    if (i > burnin && n_values % thin == 0) {
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
  if(n_stored == 0) Rcpp::stop("No proposals were accepted in MCMC. Check your model.");
  
  if (output_type == 2) {
    Vt += Valphahat / (iter - burnin); // Var[E(alpha)] + E[Var(alpha)]
  }
  trim_storage();
  acceptance_rate /= (iter - burnin);
}


void mcmc::pm_mcmc(
    ssm_sde model,
    const unsigned int nsim,
    const bool end_ram) {
  
  unsigned int m = model.m;
  unsigned int n = model.n;
  
  // get the current values of theta
  arma::vec theta = model.theta;
  model.update_model(theta); // just in case
  // compute the log[p(theta)]
  double logprior = model.log_prior_pdf(theta);
  if (!arma::is_finite(logprior)) {
    Rcpp::stop("Initial prior probability is not finite.");
  }
  arma::cube alpha(m, n + 1, nsim);
  arma::mat weights(nsim, n + 1);
  // reduce the space in case of SPDK (does not use indices)
  arma::umat indices(nsim, n  + 1);
  
  // compute the log-likelihood
  double ll = model.bsf_filter(nsim, model.L_f, alpha, weights, indices);
  
  if (!std::isfinite(ll))
    Rcpp::stop("Initial log-likelihood is not finite.");
  
  arma::mat alphahat_i(m, n + 1);
  arma::cube Vt_i(m, m, n + 1);
  arma::cube Valphahat(m, m, n + 1, arma::fill::zeros);
  arma::mat sampled_alpha(m, n + 1);
  if (output_type != 3) {
    sample_or_summarise(
      output_type == 1, 1, alpha, weights.col(n), indices,
      sampled_alpha, alphahat_i, Vt_i, model.engine);
  }
  
  double acceptance_prob = 0.0;
  bool new_value = true;
  unsigned int n_values = 0;
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  
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
    
    if (logprior_prop > -std::numeric_limits<double>::infinity() && !std::isnan(logprior_prop)) {
      
      // update parameters
      model.update_model(theta_prop);
      
      // compute the log-likelihood (unbiased and approximate)
      double ll_prop =  model.bsf_filter(nsim, model.L_f, alpha, weights, indices);
      
      //compute the acceptance probability for RAM
      acceptance_prob = std::min(1.0, std::exp(
        ll_prop - ll + logprior_prop - logprior));
      
      //accept
      double log_alpha = ll_prop - ll + logprior_prop - logprior;
      
      //accept
      if (log(unif(model.engine)) < log_alpha) {
        if (i > burnin) {
          acceptance_rate++;
          n_values++;
        }
        if (output_type != 3) {
          sample_or_summarise(
            output_type == 1, 1, alpha, weights.col(n), indices,
            sampled_alpha, alphahat_i, Vt_i, model.engine);
        }
        ll = ll_prop;
        logprior = logprior_prop;
        theta = theta_prop;
        new_value = true;
        
      }
    } else acceptance_prob = 0.0;
    
    // note: thinning does not affect this
    if (i > burnin && output_type == 2) {
      arma::mat diff = alphahat_i - alphahat;
      alphahat = (alphahat * (i - burnin - 1) + alphahat_i) / (i - burnin);
      Vt = (Vt * (i - burnin - 1) + Vt_i) / (i - burnin);
      for (unsigned int t = 0; t < model.n + 1; t++) {
        Valphahat.slice(t) += diff.col(t) * (alphahat_i.col(t) - alphahat.col(t)).t();
      }
    }
    
    if (i > burnin && n_values % thin == 0) {
      //new block
      if (new_value) {
        posterior_storage(n_stored) = logprior + ll;
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
  if(n_stored == 0) Rcpp::stop("No proposals were accepted in MCMC. Check your model.");
  
  if (output_type == 2) {
    Vt += Valphahat / (iter - burnin); // Var[E(alpha)] + E[Var(alpha)]
  }
  
  trim_storage();
  acceptance_rate /= (iter - burnin);
}

// DA-MCMC for SDE models
void mcmc::da_mcmc(ssm_sde model, 
  const unsigned int nsim, const bool end_ram){
  
  // get the current values of theta
  arma::vec theta = model.theta;
  model.update_model(theta); // just in case
  // compute the log[p(theta)]
  double logprior = model.log_prior_pdf(theta);
  if (!arma::is_finite(logprior)) {
    Rcpp::stop("Initial prior probability is not finite.");
  }
  
  unsigned int m = model.m;
  unsigned int n = model.n;
  
  arma::cube alpha(m, n + 1, nsim);
  arma::mat weights(nsim, n + 1);
  arma::umat indices(nsim, n + 1);
  
  double ll_c = model.bsf_filter(nsim, model.L_c, alpha, weights, indices);
  double ll_f = model.bsf_filter(nsim, model.L_f, alpha, weights, indices);
  
  if (!std::isfinite(ll_f))
    Rcpp::stop("Initial log-likelihood is not finite.");
  
  arma::mat alphahat_i(m, (output_type != 3) * n + 1);
  arma::cube Vt_i(m, m, (output_type != 3) * n + 1);
  arma::cube Valphahat(m, m, (output_type != 3) * n + 1, arma::fill::zeros);
  arma::mat sampled_alpha(m, (output_type != 3) * n + 1);
  
  if (output_type != 3) {
    sample_or_summarise(
      output_type == 1, 1, alpha, weights.col(n), indices,
      sampled_alpha, alphahat_i, Vt_i, model.engine);
  }
  
  double acceptance_prob = 0.0;
  bool new_value = true;
  unsigned int n_values = 0;
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  
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
    
    if (logprior_prop > -std::numeric_limits<double>::infinity() && !std::isnan(logprior_prop)) {
      
      // update parameters
      model.update_model(theta_prop);
      double ll_c_prop = model.bsf_filter(nsim, model.L_c, alpha, weights, indices);
      
      // initial acceptance probability, also used in RAM
      acceptance_prob = std::min(1.0, std::exp(ll_c_prop - ll_c +
        logprior_prop - logprior));
      // initial acceptance
      if (unif(model.engine) < acceptance_prob) {
        
        // compute the log-likelihood estimate using finer mesh
        double ll_f_prop = model.bsf_filter(nsim, model.L_f, alpha, weights, indices);
        
        // second stage acceptance log-probability
        double log_alpha = ll_f_prop + ll_c - ll_f - ll_c_prop;
        
        if (log(unif(model.engine)) < log_alpha) {
          if (i > burnin) {
            acceptance_rate++;
            n_values++;
          }
          if (output_type != 3) {
            sample_or_summarise(
              output_type == 1, 1, alpha, weights.col(n), indices,
              sampled_alpha, alphahat_i, Vt_i, model.engine);
          }
          ll_f = ll_f_prop;
          ll_c = ll_c_prop;
          logprior = logprior_prop;
          theta = theta_prop;
          new_value = true;
        }
        
      }
    } else acceptance_prob = 0.0;
    
    // note: thinning does not affect this
    if (i > burnin && output_type == 2) {
      arma::mat diff = alphahat_i - alphahat;
      alphahat = (alphahat * (i - burnin - 1) + alphahat_i) / (i - burnin);
      Vt = (Vt * (i - burnin - 1) + Vt_i) / (i - burnin);
      for (unsigned int t = 0; t < model.n + 1; t++) {
        Valphahat.slice(t) += diff.col(t) * (alphahat_i.col(t) - alphahat.col(t)).t();
      }
    }
    
    if (i > burnin && n_values % thin == 0) {
      //new block
      if (new_value) {
        posterior_storage(n_stored) = logprior + ll_f;
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
  if(n_stored == 0) Rcpp::stop("No proposals were accepted in MCMC. Check your model.");
  
  if (output_type == 2) {
    Vt += Valphahat / (iter - burnin); // Var[E(alpha)] + E[Var(alpha)]
  }
  
  trim_storage();
  acceptance_rate /= (iter - burnin);
}
