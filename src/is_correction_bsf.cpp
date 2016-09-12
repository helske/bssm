#include "bssm.h"
#include "ngssm.h"
#include "ng_bsm.h"
#include "svm.h"

// [[Rcpp::plugins(openmp)]]
template <typename T>
void is_correction_bsf(T mod, const arma::mat& theta, const arma::vec& ll_store, const arma::uvec& counts,
  unsigned int nsim_states, unsigned int n_threads, arma::vec& weights_store, 
  arma::cube& alpha_store, bool const_m) {
  
  unsigned n_iter = theta.n_cols;
  arma::uvec cum_counts = arma::cumsum(counts);
  
#pragma omp parallel num_threads(n_threads) default(none) \
  shared(n_threads, n_iter, nsim_states, theta, ll_store,            \
    weights_store, alpha_store, counts, cum_counts, const_m) firstprivate(mod)
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
        weights_store(i) = exp(logU - ll_store(i));
        
        std::discrete_distribution<> sample(V.begin(), V.end());
        
        alpha_store.slice(i) = alpha.slice(sample(mod.engine));
        
      }
    }
  
}

template void is_correction_bsf<ngssm>(ngssm mod, const arma::mat& theta, const arma::vec& ll_store, 
  const arma::uvec& counts, unsigned int nsim_states, unsigned int n_threads, arma::vec& weights_store,
  arma::cube& alpha_store, bool const_m);
template void is_correction_bsf<ng_bsm>(ng_bsm mod, const arma::mat& theta, const arma::vec& ll_store, 
  const arma::uvec& counts, unsigned int nsim_states, unsigned int n_threads, arma::vec& weights_store,
  arma::cube& alpha_store, bool const_m);
template void is_correction_bsf<svm>(svm mod, const arma::mat& theta, const arma::vec& ll_store, 
  const arma::uvec& counts, unsigned int nsim_states, unsigned int n_threads, arma::vec& weights_store,
  arma::cube& alpha_store, bool const_m);
