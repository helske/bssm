#include <RcppArmadillo.h>
#include <ramcmc.h>
#include "mcmc.h"
#include "gssm.h"
#include "ngssm.h"


// store new theta and posterior
void mcmc::update_storage(unsigned int index, const arma::vec& current_theta, 
  double current_posterior) {
  theta_store.col(index) = current_theta;
  posterior_store(index) = current_posterior;
  n_stored++;
}

// store new alpha, theta, and posterior
void mcmc::update_storage(unsigned int index, const arma::mat& current_alpha, 
  const arma::vec& current_theta, double current_posterior) {
  alpha_store.slice(index) = current_alpha.t();
  theta_store.col(index) = current_theta;
  posterior_store(index) = current_posterior;
}

// store new alpha
void mcmc::update_storage(unsigned int index, const arma::mat& current_alpha) {
  alpha_store.slice(index) = current_alpha.t();
}

// run MCMC for linear-Gaussian state space model
// target the marginal p(theta | y)
// sample states separately given the posterior sample of theta

double mcmc::mcmc_gaussian(gssm& model, bool end_ram) {
  
  double acceptance_rate = 0.0;
  
  arma::vec theta = model.get_theta();
  double logprior = model.log_prior_pdf(theta, prior_distributions, prior_parameters);
  double loglik = model.log_likelihood();
  
  unsigned int n_stored = 0;
  double acceptance_prob = 0.0;
  
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  
  for (unsigned int i = 0; i < n_iter; i++) {
    
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
    double logprior_prop = 
      model.log_prior_pdf(theta_prop, prior_distributions, prior_parameters);
    
    if (logprior_prop > -arma::datum::inf) {
      // update parameters
      model.set_theta(theta_prop);
      // compute log-likelihood with proposed theta
      double loglik_prop = model.log_likelihood();
      //compute the acceptance probability
      // use explicit min(...) as we need this value later
      double q = model.proposal(theta, theta_prop);
      acceptance_prob = std::min(1.0, exp(loglik_prop - loglik + logprior_prop - logprior + q));
      //accept
      if (unif(model.engine) < acceptance_prob) {
        if (i >= n_burnin) {
          acceptance_rate++;
        }
        loglik = loglik_prop;
        logprior = logprior_prop;
        theta = theta_prop;
      }
    } else acceptance_prob = 0.0;
    
    //store
    if ((i >= n_burnin) && (i % n_thin == 0) && n_stored < n_samples) {
      update_storage(n_stored, theta, logprior + loglik);
      n_stored++;
    }
    
    if (!end_ram || i < n_burnin) {
      ramcmc::adapt_S(S, u, acceptance_prob, target_acceptance, i + 1, gamma);
    }
    
  }
  
  return acceptance_rate / (n_iter - n_burnin);
}

// run pseudo-marginal MCMC for non-linear and/or non-Gaussian state space model
// using psi-PF
double mcmc::pm_mcmc_psi(ngssm& model, bool end_ram, unsigned int nsim_states, 
  arma::vec& signal, unsigned int max_iter, double conv_tol) {
  
  double acceptance_rate = 0.0;
  
  arma::vec theta = model.get_theta();
  double logprior = model.log_prior_pdf(theta, prior_distributions, prior_parameters);
  gssm gmodel = gaussian_approximation(signal, max_iter, conv_tol) 
  double loglik = model.psi_filter(0, nsim_states, alpha, weights, indices);
  model.backtrack(alpha, indices);
  unsigned int n_stored = 0;
  double acceptance_prob = 0.0;
  
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  
  for (unsigned int i = 0; i < n_iter; i++) {
    
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
    double logprior_prop = 
      model.log_prior_pdf(theta_prop, prior_distributions, prior_parameters);
    
    if (logprior_prop > -arma::datum::inf) {
      // update parameters
      model.set_theta(theta_prop);
      // compute log-likelihood with proposed theta
      double loglik_prop = model.psi_filter(0, nsim_states, alpha_prop, weights, indices);
      //compute the acceptance probability
      // use explicit min(...) as we need this value later
      double q = model.proposal(theta, theta_prop);
      acceptance_prob = std::min(1.0, exp(loglik_prop - loglik + logprior_prop - logprior + q));
      //accept
      if (unif(model.engine) < acceptance_prob) {
        if (i >= n_burnin) {
          acceptance_rate++;
        }
        loglik = loglik_prop;
        logprior = logprior_prop;
        theta = theta_prop;
      }
    } else acceptance_prob = 0.0;
    
    //store
    if ((i >= n_burnin) && (i % n_thin == 0) && n_stored < n_samples) {
      update_storage(n_stored, theta, logprior + loglik);
      n_stored++;
    }
    
    if (!end_ram || i < n_burnin) {
      ramcmc::adapt_S(S, u, acceptance_prob, target_acceptance, i + 1, gamma);
    }
    
  }
  
  return acceptance_rate / (n_iter - n_burnin);
  
}

// run pseudo-marginal MCMC for non-linear and/or non-Gaussian state space model
// using bsf-PF
double mcmc::pm_mcmc_psi(ngssm& model, bool end_ram, unsigned int nsim_states) {
  
  double acceptance_rate = 0.0;
  
  arma::vec theta = model.get_theta();
  double logprior = model.log_prior_pdf(theta, prior_distributions, prior_parameters);
  
  double loglik = model.bsf_filter(0, nsim_states, alpha, weights, indices);
  model.backtrack(alpha, indices);
  unsigned int n_stored = 0;
  double acceptance_prob = 0.0;
  
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  
  for (unsigned int i = 0; i < n_iter; i++) {
    
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
    double logprior_prop = 
      model.log_prior_pdf(theta_prop, prior_distributions, prior_parameters);
    
    if (logprior_prop > -arma::datum::inf) {
      // update parameters
      model.set_theta(theta_prop);
      // compute log-likelihood with proposed theta
      double loglik_prop = model.bsf_filter(0, nsim_states, alpha_prop, weights, indices);
      //compute the acceptance probability
      // use explicit min(...) as we need this value later
      double q = model.proposal(theta, theta_prop);
      acceptance_prob = std::min(1.0, exp(loglik_prop - loglik + logprior_prop - logprior + q));
      //accept
      if (unif(model.engine) < acceptance_prob) {
        if (i >= n_burnin) {
          acceptance_rate++;
        }
        loglik = loglik_prop;
        logprior = logprior_prop;
        theta = theta_prop;
      }
    } else acceptance_prob = 0.0;
    
    //store
    if ((i >= n_burnin) && (i % n_thin == 0) && n_stored < n_samples) {
      update_storage(n_stored, theta, logprior + loglik);
      n_stored++;
    }
    
    if (!end_ram || i < n_burnin) {
      ramcmc::adapt_S(S, u, acceptance_prob, target_acceptance, i + 1, gamma);
    }
    
  }
  
  return acceptance_rate / (n_iter - n_burnin);
  
}

