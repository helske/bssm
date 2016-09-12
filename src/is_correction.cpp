#include "bssm.h"
#include "ngssm.h"
#include "ng_bsm.h"
#include "svm.h"

// [[Rcpp::plugins(openmp)]]
template <typename T>
void is_correction(T mod, const arma::mat& theta, const arma::mat& y_store, const arma::mat& H_store,
  const arma::vec& ll_approx_u, const arma::uvec& counts, unsigned int nsim_states,
  unsigned int n_threads, arma::vec& weights_store, arma::cube& alpha_store, bool const_m) {

  unsigned n_iter = theta.n_cols;

  arma::uvec cum_counts = arma::cumsum(counts);
#pragma omp parallel num_threads(n_threads) default(none) \
  shared(n_threads, ll_approx_u, n_iter, nsim_states, y_store, H_store, theta, \
    weights_store, alpha_store, counts, cum_counts, const_m) firstprivate(mod)
  {
#ifdef _OPENMP
      if (n_threads > 1) {
        mod.engine = std::mt19937(omp_get_thread_num() + 1);
      }
#endif
#pragma omp for schedule(static)
    for (unsigned int i = 0; i < n_iter; i++) {

      mod.y = y_store.col(i);
      mod.H = H_store.col(i);
      mod.HH = arma::square(H_store.col(i));
      arma::vec theta_i = theta.col(i);
      mod.update_model(theta_i);

      unsigned int m = nsim_states;
      if (!const_m) {
        m *= counts(i);
      }
      arma::cube alpha = mod.sim_smoother(m, mod.distribution != 0);
      
      arma::vec weights = exp(mod.importance_weights(alpha) - ll_approx_u(i));
      weights_store(i) = arma::mean(weights);

      std::discrete_distribution<> sample(weights.begin(), weights.end());

      alpha_store.slice(i) = alpha.slice(sample(mod.engine));

    }
  }

}

template void is_correction<ngssm>(ngssm mod, const arma::mat& theta, const arma::mat& y_store, const arma::mat& H_store,
  const arma::vec& ll_approx_u, const arma::uvec& counts, unsigned int nsim_states,
  unsigned int n_threads, arma::vec& weights_store, arma::cube& alpha_store, bool const_m);
template void is_correction<ng_bsm>(ng_bsm mod, const arma::mat& theta, const arma::mat& y_store, const arma::mat& H_store,
  const arma::vec& ll_approx_u, const arma::uvec& counts, unsigned int nsim_states,
  unsigned int n_threads, arma::vec& weights_store, arma::cube& alpha_store, bool const_m);
template void is_correction<svm>(svm mod, const arma::mat& theta, const arma::mat& y_store, const arma::mat& H_store,
  const arma::vec& ll_approx_u, const arma::uvec& counts, unsigned int nsim_states,
  unsigned int n_threads, arma::vec& weights_store, arma::cube& alpha_store, bool const_m);
