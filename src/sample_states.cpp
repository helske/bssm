#include "gssm.h"
#include "bsm.h"

#ifdef _OPENMP
#include <omp.h>
#endif

template <typename T>
arma::cube sample_states(T mod, const arma::mat& theta, const arma::uvec& counts,
  unsigned int nsim_states, unsigned int n_threads) {

  unsigned n_iter = theta.n_cols;
  arma::cube alpha_store(mod.m, mod.n, nsim_states * arma::accu(counts));

  arma::uvec cum_counts = arma::cumsum(counts);

#pragma omp parallel num_threads(n_threads) default(none) shared(n_threads, n_iter, \
  nsim_states, theta, counts, cum_counts, alpha_store) firstprivate(mod)
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

      alpha_store.slices(nsim_states * (cum_counts(i)-counts(i)), nsim_states * cum_counts(i) - 1) =
        mod.sim_smoother(nsim_states * counts(i), true);

    }
  }
  return alpha_store;
}

template arma::cube sample_states<gssm>(gssm mod, const arma::mat& theta,
  const arma::uvec& counts, unsigned int nsim_states, unsigned int n_threads);
template arma::cube sample_states<bsm>(bsm mod, const arma::mat& theta,
  const arma::uvec& counts, unsigned int nsim_states, unsigned int n_threads);
