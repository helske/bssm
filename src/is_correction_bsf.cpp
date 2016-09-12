#include "bssm.h"
#include "ngssm.h"
#include "ng_bsm.h"
#include "svm.h"

// [[Rcpp::plugins(openmp)]]
template <typename T>
void is_correction_bsf(T mod, const arma::mat& theta, const arma::vec& ll_store, const arma::uvec& counts,
  unsigned int nsim_states,
  unsigned int n_threads, arma::vec& weights_store, arma::cube& alpha_store, bool const_m,
  const arma::uvec& prior_types, const arma::mat& prior_pars) {
  
  unsigned n_iter = theta.n_cols;
  arma::uvec cum_counts = arma::cumsum(counts);
#pragma omp parallel num_threads(n_threads) default(none) \
  shared(n_threads, n_iter, nsim_states, theta, ll_store,            \
    weights_store, alpha_store, counts, cum_counts, const_m, prior_types, prior_pars) firstprivate(mod)
    {
#ifdef _OPENMP
      if (n_threads > 1) {
        mod.engine = std::mt19937(omp_get_thread_num() + 1);
      }
#endif
#pragma omp for schedule(static)
      for (unsigned int i = 0; i < n_iter; i++) {
        
        arma::vec theta_i = theta.col(i);
        mod.update_model(theta_i);
        
        unsigned int m = nsim_states;
        if (!const_m) {
          m *= counts(i);
        }
        arma::cube alpha(mod.m, mod.n, m);
        arma::mat V(m, mod.n);
        arma::umat ind(m, mod.n - 1);
        double logU = mod.particle_filter(m, alpha, V, ind);
        // still missing the proposal(theta) in case of log-scale...
        double q = mod.prior_pdf(theta_i, prior_types, prior_pars); //marginal costs due to recomputing
        weights_store(i) = exp(logU + q - ll_store(i));
        alpha_store.slice(i) = mod.backward_simulate(alpha, V, ind);
        
      }
    }
  
}

template void is_correction_bsf<ngssm>(ngssm mod, const arma::mat& theta, const arma::vec& ll_store, 
  const arma::uvec& counts,
  unsigned int nsim_states, unsigned int n_threads, arma::vec& weights_store,
  arma::cube& alpha_store, bool const_m,
  const arma::uvec& prior_types, const arma::mat& prior_pars);
template void is_correction_bsf<ng_bsm>(ng_bsm mod, const arma::mat& theta, const arma::vec& ll_store, 
  const arma::uvec& counts,
  unsigned int nsim_states, unsigned int n_threads, arma::vec& weights_store,
  arma::cube& alpha_store, bool const_m,
  const arma::uvec& prior_types, const arma::mat& prior_pars);
template void is_correction_bsf<svm>(svm mod, const arma::mat& theta, const arma::vec& ll_store, 
  const arma::uvec& counts,
  unsigned int nsim_states, unsigned int n_threads, arma::vec& weights_store,
  arma::cube& alpha_store, bool const_m,
  const arma::uvec& prior_types, const arma::mat& prior_pars);
