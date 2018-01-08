#ifdef _OPENMP
#include <omp.h>
#endif
#include <sitmo.h>
#include <ramcmc.h>
#include "nlg_amcmc.h"
#include "nlg_ssm.h"

#include "rep_mat.h"
#include "filter_smoother.h"

nlg_amcmc::nlg_amcmc(const unsigned int n_iter, 
  const unsigned int n_burnin, const unsigned int n_thin, const unsigned int n, 
  const unsigned int m, const double target_acceptance, const double gamma, 
  const arma::mat& S, const bool store_modes) :
  mcmc(n_iter, n_burnin, n_thin, n, m,
    target_acceptance, gamma, S, true),
    weight_storage(arma::vec(n_samples, arma::fill::zeros)),
    approx_loglik_storage(arma::vec(n_samples)),
    scales_storage(arma::vec(n_samples)),
    prior_storage(arma::vec(n_samples)),
    store_modes(store_modes),
    mode_storage(arma::cube(m, n + 1, n_samples * store_modes)){
}

void nlg_amcmc::trim_storage() {
  theta_storage.resize(n_par, n_stored);
  posterior_storage.resize(n_stored);
  count_storage.resize(n_stored);
  alpha_storage.resize(alpha_storage.n_rows, alpha_storage.n_cols, n_stored);
  scales_storage.resize(n_stored);
  weight_storage.resize(n_stored);
  approx_loglik_storage.resize(n_stored);
  prior_storage.resize(n_stored);
  mode_storage.resize(mode_storage.n_rows, mode_storage.n_cols, n_stored);
}

void nlg_amcmc::expand() {
  
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
  
  arma::mat expanded_scales = rep_mat(scales_storage, count_storage);
  scales_storage.set_size(scales_storage.n_rows, n_stored);
  scales_storage = expanded_scales;
  
  arma::vec expanded_weight = rep_vec(weight_storage, count_storage);
  weight_storage.set_size(n_stored);
  weight_storage = expanded_weight;
  
  arma::vec expanded_approx_loglik = rep_vec(approx_loglik_storage, count_storage);
  approx_loglik_storage.set_size(n_stored);
  approx_loglik_storage = expanded_approx_loglik;
  
  arma::vec expanded_prior = rep_vec(prior_storage, count_storage);
  prior_storage.set_size(n_stored);
  prior_storage = expanded_prior;
  
  arma::cube expanded_mode = rep_cube(mode_storage, count_storage);
  mode_storage.set_size(mode_storage.n_rows, mode_storage.n_cols, n_stored);
  mode_storage = expanded_mode;
  count_storage.resize(n_stored);
  count_storage.ones();
  
}
// run approximate MCMC for
// non-linear Gaussian state space model

void nlg_amcmc::approx_mcmc(nlg_ssm model, const unsigned int max_iter, 
  const double conv_tol, const bool end_ram, const unsigned int iekf_iter) {
  
  unsigned int m = model.m;
  unsigned n = model.n;
  
  double logprior = model.log_prior_pdf(model.theta);
  if (!arma::is_finite(logprior)) {
    Rcpp::stop("Initial prior probability is not finite.");
  }
  arma::mat mode_estimate(m, n);
  mgg_ssm approx_model0 = model.approximate(mode_estimate, max_iter, conv_tol, iekf_iter);
  if (!arma::is_finite(mode_estimate)) {
    Rcpp::stop("Approximation based on initial theta failed.");
  }
  double sum_scales = arma::accu(model.scaling_factors(approx_model0, mode_estimate));
  // compute the log-likelihood of the approximate model
  double loglik = approx_model0.log_likelihood() + sum_scales;
  if (!arma::is_finite(loglik)) {
    Rcpp::stop("Initial approximate likelihood is not finite.");
  }
  double acceptance_prob = 0.0;
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  
  arma::vec theta = model.theta;
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
      model.theta = theta_prop;
      arma::mat mode_estimate_prop(m, n);
      mgg_ssm approx_model = model.approximate(mode_estimate_prop, max_iter, 
        conv_tol, iekf_iter);
      double loglik_prop;
      double sum_scales_prop = 0.0; // initialize in order to get rid of false warning
      if(!is_finite(mode_estimate_prop)) {
        loglik_prop = -std::numeric_limits<double>::infinity();
      } else {
        sum_scales_prop = 
          arma::accu(model.scaling_factors(approx_model, mode_estimate_prop));
        // compute the log-likelihood of the approximate model
        loglik_prop = approx_model.log_likelihood() + sum_scales_prop;
      }
      
      if (loglik_prop > -std::numeric_limits<double>::infinity() && !std::isnan(loglik_prop)) {
      
        acceptance_prob = std::min(1.0, 
          std::exp(loglik_prop - loglik + logprior_prop - logprior));
        
      } else {
        acceptance_prob = 0.0; 
      }
      
      if (unif(model.engine) < acceptance_prob) {
        if (i > n_burnin) {
          acceptance_rate++;
          n_values++;
        }
        loglik = loglik_prop;
        logprior = logprior_prop;
        theta = theta_prop;
        sum_scales = sum_scales_prop;
        mode_estimate = mode_estimate_prop;
        new_value = true;
      }
    } else acceptance_prob = 0.0;
    
    if (i > n_burnin && n_values % n_thin == 0) {
      //new block
      if (new_value) {
        approx_loglik_storage(n_stored) = loglik;
        prior_storage(n_stored) = logprior;
        theta_storage.col(n_stored) = theta;
        scales_storage(n_stored) = sum_scales;
        count_storage(n_stored) = 1;
        if(store_modes) {
          mode_storage.slice(n_stored) = mode_estimate;
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
  
  trim_storage();
  acceptance_rate /= (n_iter - n_burnin);
}

void nlg_amcmc::ekf_mcmc(nlg_ssm model, const bool end_ram, const unsigned int iekf_iter) {
  
  double logprior = model.log_prior_pdf(model.theta);
  
  // compute the log-likelihood
  double loglik = model.ekf_loglik(iekf_iter);
  if (!arma::is_finite(loglik)) {
    Rcpp::stop("Initial approximate likelihood is not finite.");
  }
  double acceptance_prob = 0.0;
  std::normal_distribution<> normal(0.0, 1.0);
  std::uniform_real_distribution<> unif(0.0, 1.0);
  
  arma::vec theta = model.theta;
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
      model.theta = theta_prop;
      double loglik_prop = model.ekf_loglik(iekf_iter);
      
      if (loglik_prop > -std::numeric_limits<double>::infinity() && !std::isnan(loglik_prop)) {
        
        acceptance_prob = std::min(1.0, 
          std::exp(loglik_prop - loglik + logprior_prop - logprior));
        
      } else {
        acceptance_prob = 0.0; 
      }
      
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

void nlg_amcmc::is_correction_bsf(nlg_ssm model, const unsigned int nsim_states, 
  const unsigned int is_type, const unsigned int n_threads) {
  
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads) default(none) firstprivate(model) 
{
  
  model.engine = sitmo::prng_engine(omp_get_thread_num() + 1);
  
#pragma omp for schedule(dynamic)
  for (unsigned int i = 0; i < theta_storage.n_cols; i++) {
    
    model.theta = theta_storage.col(i);
    
    unsigned int nsim = nsim_states;
    if (is_type == 1) {
      nsim *= count_storage(i);
    }
    
    arma::cube alpha_i(model.m, model.n + 1, nsim);
    arma::mat weights_i(nsim, model.n + 1);
    arma::umat indices(nsim, model.n);
    
    double loglik = model.bsf_filter(nsim, alpha_i, weights_i, indices);
    if(arma::is_finite(loglik)) {
      weight_storage(i) = std::exp(loglik - approx_loglik_storage(i));
      filter_smoother(alpha_i, indices);
      arma::vec w = weights_i.col(model.n);
      std::discrete_distribution<unsigned int> sample(w.begin(), w.end());
      alpha_storage.slice(i) = alpha_i.slice(sample(model.engine)).t();
    } else {
      weight_storage(i) = 0.0;
      alpha_storage.slice(i).zeros();
    }
  }
}
#else
for (unsigned int i = 0; i < theta_storage.n_cols; i++) {
  
  model.theta = theta_storage.col(i);
  
  unsigned int nsim = nsim_states;
  if (is_type == 1) {
    nsim *= count_storage(i);
  }
  
  arma::cube alpha_i(model.m, model.n + 1, nsim);
  arma::mat weights_i(nsim, model.n + 1);
  arma::umat indices(nsim, model.n);
  
  double loglik = model.bsf_filter(nsim, alpha_i, weights_i, indices);
  if(arma::is_finite(loglik)) {
    weight_storage(i) = std::exp(loglik - approx_loglik_storage(i));
    filter_smoother(alpha_i, indices);
    arma::vec w = weights_i.col(model.n);
    std::discrete_distribution<unsigned int> sample(w.begin(), w.end());
    alpha_storage.slice(i) = alpha_i.slice(sample(model.engine)).t();
  } else {
    weight_storage(i) = 0.0;
    alpha_storage.slice(i).zeros();
  }
}
#endif

posterior_storage = prior_storage + arma::log(weight_storage);
}


void nlg_amcmc::is_correction_psi(nlg_ssm model, const unsigned int nsim_states, 
  const unsigned int is_type, const unsigned int n_threads) {
  
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads) default(none) firstprivate(model)
{
  model.engine = sitmo::prng_engine(omp_get_thread_num() + 1);
  
  unsigned int p = model.p;
  unsigned int n = model.n;
  unsigned int m = model.m;
  unsigned int k = model.k;
  
  arma::vec a1(m);
  arma::mat P1(m, m);
  arma::cube Z(p, m, n);
  arma::cube H(p, p, (n - 1) * model.Htv + 1);
  arma::cube T(m, m, n);
  arma::cube R(m, k, (n - 1) * model.Rtv + 1);
  arma::mat D(p, n);
  arma::mat C(m, n);
  
  mgg_ssm approx_model(model.y, Z, H, T, R, a1, P1, arma::cube(0,0,0),
    arma::mat(0,0), D, C, model.seed);
  
#pragma omp for schedule(dynamic)
  for (unsigned int i = 0; i < theta_storage.n_cols; i++) {
    
    model.theta = theta_storage.col(i);
    
    approx_model.a1 = model.a1_fn(model.theta, model.known_params);
    approx_model.P1 = model.P1_fn(model.theta, model.known_params);
    for (unsigned int t = 0; t < Z.n_slices; t++) {
      approx_model.Z.slice(t) = model.Z_gn(t, mode_storage.slice(i).col(t), model.theta, model.known_params, model.known_tv_params);
      approx_model.T.slice(t) = model.T_gn(t, mode_storage.slice(i).col(t), model.theta, model.known_params, model.known_tv_params);
      approx_model.D.col(t) = model.Z_fn(t, mode_storage.slice(i).col(t), model.theta, model.known_params, model.known_tv_params) -
        approx_model.Z.slice(t) * mode_storage.slice(i).col(t);
      approx_model.C.col(t) =  model.T_fn(t, mode_storage.slice(i).col(t), model.theta, model.known_params, model.known_tv_params) -
        approx_model.T.slice(t) * mode_storage.slice(i).col(t);
    }
    for (unsigned int t = 0; t < H.n_slices; t++) {
      approx_model.H.slice(t) = model.H_fn(t, mode_storage.slice(i).col(t), model.theta, model.known_params, model.known_tv_params);
    }
    for (unsigned int t = 0; t < R.n_slices; t++) {
      approx_model.R.slice(t) = model.R_fn(t, mode_storage.slice(i).col(t), model.theta, model.known_params, model.known_tv_params);
    }
    approx_model.compute_HH();
    approx_model.compute_RR();
    
    unsigned int nsim = nsim_states;
    if (is_type == 1) {
      nsim *= count_storage(i);
    }
    
    arma::cube alpha_i(model.m, model.n + 1, nsim);
    arma::mat weights_i(nsim, model.n + 1);
    arma::umat indices(nsim, model.n);
    
    double loglik = model.psi_filter(approx_model, 0.0, nsim, alpha_i, weights_i, indices);
    if(arma::is_finite(loglik)) {
      weight_storage(i) = std::exp(loglik);
      
      filter_smoother(alpha_i, indices);
      arma::vec w = weights_i.col(model.n);
      std::discrete_distribution<unsigned int> sample(w.begin(), w.end());
      alpha_storage.slice(i) = alpha_i.slice(sample(model.engine)).t(); 
    } else {
      weight_storage(i) = 0.0;
      alpha_storage.slice(i).zeros();
    }
  }
}
#else
unsigned int p = model.p;
unsigned int n = model.n;
unsigned int m = model.m;
unsigned int k = model.k;

arma::vec a1(m);
arma::mat P1(m, m);
arma::cube Z(p, m, n);
arma::cube H(p, p, (n - 1) * model.Htv + 1);
arma::cube T(m, m, n);
arma::cube R(m, k, (n - 1) * model.Rtv + 1);
arma::mat D(p, n);
arma::mat C(m, n);

mgg_ssm approx_model(model.y, Z, H, T, R, a1, P1, arma::cube(0,0,0),
  arma::mat(0,0), D, C, model.seed);

for (unsigned int i = 0; i < theta_storage.n_cols; i++) {
  
  model.theta = theta_storage.col(i);
  
  approx_model.a1 = model.a1_fn(model.theta, model.known_params);
  approx_model.P1 = model.P1_fn(model.theta, model.known_params);
  for (unsigned int t = 0; t < Z.n_slices; t++) {
    approx_model.Z.slice(t) = model.Z_gn(t, mode_storage.slice(i).col(t), model.theta, model.known_params, model.known_tv_params);
    approx_model.T.slice(t) = model.T_gn(t, mode_storage.slice(i).col(t), model.theta, model.known_params, model.known_tv_params);
    approx_model.D.col(t) = model.Z_fn(t, mode_storage.slice(i).col(t), model.theta, model.known_params, model.known_tv_params) -
      approx_model.Z.slice(t) * mode_storage.slice(i).col(t);
    approx_model.C.col(t) =  model.T_fn(t, mode_storage.slice(i).col(t), model.theta, model.known_params, model.known_tv_params) -
      approx_model.T.slice(t) * mode_storage.slice(i).col(t);
  }
  for (unsigned int t = 0; t < H.n_slices; t++) {
    approx_model.H.slice(t) = model.H_fn(t, mode_storage.slice(i).col(t), model.theta, model.known_params, model.known_tv_params);
  }
  for (unsigned int t = 0; t < R.n_slices; t++) {
    approx_model.R.slice(t) = model.R_fn(t, mode_storage.slice(i).col(t), model.theta, model.known_params, model.known_tv_params);
  }
  approx_model.compute_HH();
  approx_model.compute_RR();
  
  unsigned int nsim = nsim_states;
  if (is_type == 1) {
    nsim *= count_storage(i);
  }
  
  arma::cube alpha_i(model.m, model.n + 1, nsim);
  arma::mat weights_i(nsim, model.n + 1);
  arma::umat indices(nsim, model.n);
  
  double loglik = model.psi_filter(approx_model, 0.0, nsim, alpha_i, weights_i, indices);
  if(arma::is_finite(loglik)) {
    weight_storage(i) = std::exp(loglik);
    
    filter_smoother(alpha_i, indices);
    arma::vec w = weights_i.col(model.n);
    std::discrete_distribution<unsigned int> sample(w.begin(), w.end());
    alpha_storage.slice(i) = alpha_i.slice(sample(model.engine)).t(); 
  } else {
    weight_storage(i) = 0.0;
    alpha_storage.slice(i).zeros();
  }
}
#endif
posterior_storage = prior_storage + approx_loglik_storage - scales_storage + 
  arma::log(weight_storage);
}

void nlg_amcmc::state_ekf_sample(nlg_ssm model, const unsigned int n_threads, const unsigned int iekf_iter) {
  
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads) default(none) firstprivate(model)
{
  model.engine = sitmo::prng_engine(omp_get_thread_num() + 1);
  
  unsigned int p = model.p;
  unsigned int n = model.n;
  unsigned int m = model.m;
  unsigned int k = model.k;
  
  arma::vec a1(m);
  arma::mat P1(m, m);
  arma::cube Z(p, m, n);
  arma::cube H(p, p, (n - 1) * model.Htv + 1);
  arma::cube T(m, m, n);
  arma::cube R(m, k, (n - 1) * model.Rtv + 1);
  arma::mat D(p, n);
  arma::mat C(m, n);
  
  mgg_ssm approx_model(model.y, Z, H, T, R, a1, P1, arma::cube(0,0,0),
    arma::mat(0,0), D, C, model.seed);
  
#pragma omp for schedule(dynamic)
  for (unsigned int i = 0; i < theta_storage.n_cols; i++) {
    
    model.theta = theta_storage.col(i);
    arma::mat at(m, n + 1);
    arma::mat att(m, n);
    arma::cube Pt(m, m, n + 1);
    arma::cube Ptt(m, m, n);
    model.ekf(at, att, Pt, Ptt, iekf_iter);
    
    approx_model.a1 = model.a1_fn(model.theta, model.known_params);
    approx_model.P1 = model.P1_fn(model.theta, model.known_params);
    for (unsigned int t = 0; t < Z.n_slices; t++) {
      approx_model.Z.slice(t) = model.Z_gn(t, at.col(t), model.theta, model.known_params, model.known_tv_params);
      approx_model.T.slice(t) = model.T_gn(t, att.col(t), model.theta, model.known_params, model.known_tv_params);
      approx_model.D.col(t) = model.Z_fn(t, at.col(t), model.theta, model.known_params, model.known_tv_params) -
        approx_model.Z.slice(t) * at.col(t);
      approx_model.C.col(t) =  model.T_fn(t, att.col(t), model.theta, model.known_params, model.known_tv_params) -
        approx_model.T.slice(t) * att.col(t);
    }
    for (unsigned int t = 0; t < H.n_slices; t++) {
      approx_model.H.slice(t) = model.H_fn(t, at.col(t), model.theta, model.known_params, model.known_tv_params);
    }
    for (unsigned int t = 0; t < R.n_slices; t++) {
      approx_model.R.slice(t) = model.R_fn(t, att.col(t), model.theta, model.known_params, model.known_tv_params);
    }
    approx_model.compute_HH();
    approx_model.compute_RR();
    alpha_storage.slice(i) = approx_model.simulate_states().slice(0).t();
    
  }
}
#else
unsigned int p = model.p;
unsigned int n = model.n;
unsigned int m = model.m;
unsigned int k = model.k;

arma::vec a1(m);
arma::mat P1(m, m);
arma::cube Z(p, m, n);
arma::cube H(p, p, (n - 1) * model.Htv + 1);
arma::cube T(m, m, n);
arma::cube R(m, k, (n - 1) * model.Rtv + 1);
arma::mat D(p, n);
arma::mat C(m, n);

mgg_ssm approx_model(model.y, Z, H, T, R, a1, P1, arma::cube(0,0,0),
  arma::mat(0,0), D, C, model.seed);

#pragma omp for schedule(dynamic)
for (unsigned int i = 0; i < theta_storage.n_cols; i++) {
  
  model.theta = theta_storage.col(i);
  
  arma::mat at(m, n + 1);
  arma::mat att(m, n);
  arma::cube Pt(m, m, n + 1);
  arma::cube Ptt(m, m, n);
  model.ekf(at, att, Pt, Ptt, iekf_iter);
  
  approx_model.a1 = model.a1_fn(model.theta, model.known_params);
  approx_model.P1 = model.P1_fn(model.theta, model.known_params);
  for (unsigned int t = 0; t < Z.n_slices; t++) {
    approx_model.Z.slice(t) = model.Z_gn(t, at.col(t), model.theta, model.known_params, model.known_tv_params);
    approx_model.T.slice(t) = model.T_gn(t, att.col(t), model.theta, model.known_params, model.known_tv_params);
    approx_model.D.col(t) = model.Z_fn(t, at.col(t), model.theta, model.known_params, model.known_tv_params) -
      approx_model.Z.slice(t) * at.col(t);
    approx_model.C.col(t) =  model.T_fn(t, att.col(t), model.theta, model.known_params, model.known_tv_params) -
      approx_model.T.slice(t) * att.col(t);
  }
  for (unsigned int t = 0; t < H.n_slices; t++) {
    approx_model.H.slice(t) = model.H_fn(t, at.col(t), model.theta, model.known_params, model.known_tv_params);
  }
  for (unsigned int t = 0; t < R.n_slices; t++) {
    approx_model.R.slice(t) = model.R_fn(t, att.col(t), model.theta, model.known_params, model.known_tv_params);
  }
  approx_model.compute_HH();
  approx_model.compute_RR();
  alpha_storage.slice(i) = approx_model.simulate_states().slice(0).t();
  
}
#endif

posterior_storage = prior_storage + approx_loglik_storage;
}

void nlg_amcmc::state_ekf_summary(nlg_ssm& model,
  arma::mat& alphahat, arma::cube& Vt, const unsigned int iekf_iter) {
  
  // first iteration
  model.theta = theta_storage.col(0);
  model.ekf_smoother(alphahat, Vt, iekf_iter);
  
  double sum_w = count_storage(0);
  arma::mat alphahat_i = alphahat;
  arma::cube Vt_i = Vt;
  
  arma::cube Valpha(model.m, model.m, model.n + 1, arma::fill::zeros);
  
  for (unsigned int i = 1; i < theta_storage.n_cols; i++) {
    
    model.theta = theta_storage.col(i);
    model.ekf_smoother(alphahat_i, Vt_i, iekf_iter);
    arma::mat diff = alphahat_i - alphahat;
    double tmp = count_storage(i) + sum_w;
    alphahat = (alphahat * sum_w + alphahat_i * count_storage(i)) / tmp;
    
    for (unsigned int t = 0; t < model.n + 1; t++) {
      Valpha.slice(t) += diff.col(t) * (alphahat_i.col(t) - alphahat.col(t)).t();
    }
    Vt = (Vt * sum_w + Vt_i * count_storage(i)) / tmp;
    sum_w = tmp;
  }
  Vt += Valpha / theta_storage.n_cols; // Var[E(alpha)] + E[Var(alpha)]
}


