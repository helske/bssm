#include "bssm.h"
#include "ngssm.h"
#include "ng_bsm.h"
#include "svm.h"

// [[Rcpp::plugins(openmp)]]
template <typename T>
void is_correction_psif(T mod, const arma::mat& theta, const arma::mat& y_store, const arma::mat& H_store,
  const arma::vec& ll_store, const arma::uvec& counts, unsigned int nsim_states,
  unsigned int n_threads, arma::vec& weights_store, arma::cube& alpha_store, bool const_m) {

  unsigned n_iter = theta.n_cols;

  arma::uvec cum_counts = arma::cumsum(counts);
#pragma omp parallel num_threads(n_threads) default(none) \
  shared(n_threads, ll_store, n_iter, nsim_states, y_store, H_store, theta, \
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
      arma::cube alpha(mod.m, mod.n, m);
      arma::mat V(m, mod.n);
      arma::umat ind(m, mod.n - 1);
      // not optimal, computes the log-likelihood again even though we could save it in MCMC
      double ll = mod.psi_filter_precomp(m, alpha, V, ind, mod.log_likelihood(mod.distribution != 0));
      backtrack_pf(alpha, ind);
      weights_store(i) = exp(ll - ll_store(i)); //priors cancel out
      
      arma::vec tmp = V.col(mod.n - 1);
      std::discrete_distribution<> sample(tmp.begin(), tmp.end());

      alpha_store.slice(i) = alpha.slice(sample(mod.engine)).t();

    }
  }

}

template void is_correction_psif<ngssm>(ngssm mod, const arma::mat& theta, const arma::mat& y_store, const arma::mat& H_store,
  const arma::vec& ll_store, const arma::uvec& counts, unsigned int nsim_states,
  unsigned int n_threads, arma::vec& weights_store, arma::cube& alpha_store, bool const_m);
template void is_correction_psif<ng_bsm>(ng_bsm mod, const arma::mat& theta, const arma::mat& y_store, const arma::mat& H_store,
  const arma::vec& ll_store, const arma::uvec& counts, unsigned int nsim_states,
  unsigned int n_threads, arma::vec& weights_store, arma::cube& alpha_store, bool const_m);
template void is_correction_psif<svm>(svm mod, const arma::mat& theta, const arma::mat& y_store, const arma::mat& H_store,
  const arma::vec& ll_store, const arma::uvec& counts, unsigned int nsim_states,
  unsigned int n_threads, arma::vec& weights_store, arma::cube& alpha_store, bool const_m);
