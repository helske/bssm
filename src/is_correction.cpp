#include "bssm.h"
#include "ngssm.h"
#include "ng_bsm.h"

// [[Rcpp::plugins(openmp)]]
template <typename T> 
void is_correction(T mod, const arma::mat& theta, const arma::mat& y_store, const arma::mat& H_store,
  const arma::vec& ll_approx_u,  unsigned int nsim_states,
  unsigned int n_threads, arma::uvec seeds, arma::vec& weights_store, arma::cube& alpha_store) {
  
  unsigned n_iter = theta.n_cols;
  
#pragma omp parallel num_threads(n_threads) default(none) \
  shared(ll_approx_u, n_iter, nsim_states, y_store, H_store, theta, weights_store, alpha_store, seeds) firstprivate(mod)
  {
    if (seeds.n_elem == 1) {
      mod.engine = std::mt19937(seeds(0));
    } else {
      mod.engine = std::mt19937(seeds(omp_get_thread_num()));
    }
    
#pragma omp for schedule(static)
    for (int i = 0; i < n_iter; i++) {
      
      mod.y = y_store.col(i);
      mod.H = H_store.col(i);
      mod.HH = arma::square(H_store.col(i));
      arma::vec theta_i = theta.col(i);
      mod.update_model(theta_i);
      
      arma::cube alpha = mod.sim_smoother(nsim_states);
      arma::vec weights = exp(mod.importance_weights2(alpha) - ll_approx_u(i));
      weights_store(i) = arma::mean(weights);
      std::discrete_distribution<> sample(weights.begin(), weights.end());
      
      alpha_store.slice(i) = alpha.slice(sample(mod.engine));
      
    }
  }
  
}

template void is_correction<ngssm>(ngssm mod, const arma::mat& theta, const arma::mat& y_store, const arma::mat& H_store,
  const arma::vec& ll_approx_u,  unsigned int nsim_states,
  unsigned int n_threads, arma::uvec seeds, arma::vec& weights_store, arma::cube& alpha_store);
template void is_correction<ng_bsm>(ng_bsm mod, const arma::mat& theta, const arma::mat& y_store, const arma::mat& H_store,
  const arma::vec& ll_approx_u,  unsigned int nsim_states,
  unsigned int n_threads, arma::uvec seeds, arma::vec& weights_store, arma::cube& alpha_store);
