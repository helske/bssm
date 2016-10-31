#include "bssm.h"
#include "ngssm.h"
#include "ng_bsm.h"
#include "svm.h"

// [[Rcpp::plugins(openmp)]]
template <typename T>
void is_correction_summary(T mod, const arma::mat& theta, const arma::mat& y_store, const arma::mat& H_store,
  const arma::vec& ll_approx_u, const arma::uvec& counts, unsigned int nsim_states,
  unsigned int n_threads,
  arma::vec& weights_store, arma::mat& alphahat, arma::cube& Vt, arma::mat& mu, arma::cube& Vmu,
  bool const_m) {
  
  unsigned n_iter = theta.n_cols;
  
  alphahat.zeros();
  Vt.zeros();
  mu.zeros();
  Vmu.zeros();
  arma::cube Valpha(mod.m, mod.m, mod.n, arma::fill::zeros);
  arma::cube Vmu2(1, 1, mod.n, arma::fill::zeros);
  double cumsumw = 0;
#pragma omp parallel num_threads(n_threads) default(none)           \
  shared(n_threads, ll_approx_u, n_iter, nsim_states, y_store, H_store, theta, \
    weights_store, counts, alphahat, Vt, Valpha, mu, Vmu, Vmu2, cumsumw, const_m) firstprivate(mod)
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
        arma::mat alphahat_i(mod.m, mod.n);
        arma::cube Vt_i(mod.m, mod.m, mod.n);
        running_weighted_summary(alpha, alphahat_i, Vt_i, weights);
        arma::mat mu_i(1, mod.n);
        arma::cube Vmu_i(1, 1, mod.n);
        running_weighted_summary(mod.invlink(alpha), mu_i, Vmu_i, weights);
        
#pragma omp critical
{
  double w = weights_store(i) * counts(i);
  
  
  arma::mat diff = alphahat_i - alphahat;
  arma::mat diff_mu = mu_i - mu;
  
  
  double tmp = w + cumsumw;
  alphahat += diff * w / tmp;
  for (unsigned int t = 0; t < diff.n_cols; t++) {
    Valpha.slice(t) +=  w * diff.col(t) * (alphahat_i.col(t) - alphahat.col(t)).t();
  }
  Vt += (Vt_i - Vt) * w / tmp;
  
  mu += diff_mu * w / tmp;
  for (unsigned int t = 0; t < diff_mu.n_cols; t++) {
    Vmu2.slice(t) +=  w * diff_mu.col(t) * (mu_i.col(t) - mu.col(t)).t();
  }
  Vmu += (Vmu_i - Vmu) * w / tmp;
  cumsumw = tmp;
}

      }
    }
  Vt = Vt + Valpha/cumsumw;
  Vmu = Vmu + Vmu2/cumsumw;
  
  
}
template void is_correction_summary<ngssm>(ngssm mod, const arma::mat& theta, const arma::mat& y_store, const arma::mat& H_store,
  const arma::vec& ll_approx_u, const arma::uvec& counts, unsigned int nsim_states,
  unsigned int n_threads, arma::vec& weights_store,
  arma::mat& alphahat, arma::cube& Vt, arma::mat& mu, arma::cube& Vmu, bool const_m);

template void is_correction_summary<ng_bsm>(ng_bsm mod, const arma::mat& theta, const arma::mat& y_store, const arma::mat& H_store,
  const arma::vec& ll_approx_u, const arma::uvec& counts, unsigned int nsim_states,
  unsigned int n_threads, arma::vec& weights_store,
  arma::mat& alphahat, arma::cube& Vt, arma::mat& mu, arma::cube& Vmu, bool const_m);
// 
// template void is_correction_summary<svm>(svm mod, const arma::mat& theta, const arma::mat& y_store, const arma::mat& H_store,
//   const arma::vec& ll_approx_u, const arma::uvec& counts, unsigned int nsim_states,
//   unsigned int n_threads, arma::vec& weights_store,
//   arma::mat& alphahat, arma::cube& Vt, arma::mat& mu, arma::cube& Vmu, bool const_m);
