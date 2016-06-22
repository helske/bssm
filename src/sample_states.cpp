#include "bssm.h"
#include "gssm.h"
#include "bsm.h"

// [[Rcpp::plugins(openmp)]]
template <typename T> 
arma::cube sample_states(T mod, const arma::mat& theta,
  unsigned int nsim_states, unsigned int n_threads, arma::uvec seeds) {
  
  unsigned n_iter = theta.n_cols;
  arma::cube alpha_store(mod.m, mod.n, nsim_states * n_iter);
  
#pragma omp parallel num_threads(n_threads) default(none) shared(n_iter, \
  nsim_states, theta, alpha_store, seeds) firstprivate(mod)
  {
    if (seeds.n_elem == 1) {
      mod.engine = std::mt19937(seeds(0));
    } else {
      mod.engine = std::mt19937(seeds(omp_get_thread_num()));
    }
#pragma omp for schedule(static)
    for (int i = 0; i < n_iter; i++) {
      
      arma::vec theta_i = theta.col(i);
      mod.update_model(theta_i);
      
      alpha_store.slices(i * nsim_states, (i + 1) * nsim_states - 1) = mod.sim_smoother(nsim_states);
      
    }
  }
  return alpha_store;
}

template arma::cube sample_states<gssm>(gssm mod, const arma::mat& theta,
  unsigned int nsim_states, unsigned int n_threads, arma::uvec seeds);
template arma::cube sample_states<bsm>(bsm mod, const arma::mat& theta,
    unsigned int nsim_states, unsigned int n_threads, arma::uvec seeds);