#include "ngssm.h"
#include "ng_bsm.h"
#include "svm.h"
#include "backtrack.h"

#ifdef _OPENMP
#include <omp.h>
#endif

template <typename T>
void is_correction_psif(T model, const arma::mat& theta_store, const arma::mat& y_store, const arma::mat& H_store,
  const arma::mat& ll_approx_u, const arma::uvec& counts, unsigned int nsim_states,
  unsigned int n_threads, arma::vec& weights_store, arma::cube& alpha_store, bool const_m, 
  const arma::uvec& seeds) {
  
  if(n_threads > 1) {
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads) default(none)                      \
    shared(n_threads, ll_approx_u, nsim_states, y_store, H_store, theta_store, \
      weights_store, alpha_store, counts, const_m, seeds) firstprivate(model)
      {
        double n_iter = theta_store.n_cols;
       
        model.engine = std::mt19937(seeds(omp_get_thread_num()));
        unsigned int thread_size = std::floor(n_iter/n_threads);
        unsigned int start = omp_get_thread_num()*thread_size;
        unsigned int end = (omp_get_thread_num()+1)*thread_size - 1;
        if(omp_get_thread_num() == (int)(n_threads - 1)) {
          end = n_iter - 1;
        }
        arma::mat theta_piece = theta_store(arma::span::all, arma::span(start, end));
        arma::mat ll_approx_u_piece = ll_approx_u(arma::span::all, arma::span(start, end));
        arma::uvec counts_piece = counts(arma::span(start, end));
        arma::vec weights_piece = weights_store(arma::span(start, end));
        arma::cube alpha_piece = alpha_store.slices(start, end);
        arma::mat y_piece = y_store(arma::span::all, arma::span(start, end));
        arma::mat H_piece = H_store(arma::span::all, arma::span(start, end));
        if(const_m) {
          is_psi_cm(model, theta_piece, y_piece, H_piece, ll_approx_u_piece, counts_piece, 
            nsim_states, weights_piece, alpha_piece);
        } else {
          is_psi_ncm(model, theta_piece, y_piece, H_piece, ll_approx_u_piece, counts_piece, 
            nsim_states, weights_piece, alpha_piece);
        }
        weights_store(arma::span(start, end)) = weights_piece;
        alpha_store.slices(start, end) = alpha_piece;
      }
#else
    if(const_m) {
      is_psi_cm(model, theta_store, y_store, H_store, ll_approx_u, counts, nsim_states, 
        weights_store, alpha_store);
    } else {
      is_psi_ncm(model, theta_store, y_store, H_store, ll_approx_u, counts, nsim_states, 
        weights_store, alpha_store);
    }
#endif
  } else {
    if(const_m) {
      is_psi_cm(model, theta_store, y_store, H_store, ll_approx_u, counts, nsim_states, weights_store, 
        alpha_store);
    } else {
      is_psi_ncm(model, theta_store, y_store, H_store, ll_approx_u, counts, nsim_states, weights_store, 
        alpha_store);
    }
  }
}

template <typename T>
void is_psi_cm(T model, const arma::mat& theta_store, const arma::mat& y_store, const arma::mat& H_store,
  const arma::mat& ll_approx_u, const arma::uvec& counts, unsigned int nsim_states,
  arma::vec& weights_store, arma::cube& alpha_store) {
  
  arma::cube alpha(model.m, model.n, nsim_states);
  arma::mat V(nsim_states, model.n);
  arma::umat ind(nsim_states, model.n - 1);
  
  for (unsigned int i = 0; i < theta_store.n_cols; i++) {
    
    
    model.y = y_store.col(i);
    model.H = H_store.col(i);
    model.HH = arma::square(H_store.col(i));
    arma::vec theta_i = theta_store.col(i);
    model.update_model(theta_i);
    
    weights_store(i) = exp(model.psi_filter(nsim_states, alpha, V, ind, 0.0, ll_approx_u.col(i)));
    //backtrack all, could speed up a bit by tracking only one
    backtrack_pf(alpha, ind);
    
    arma::vec tmp = V.col(model.n - 1);
    std::discrete_distribution<> sample(tmp.begin(), tmp.end());
    
    alpha_store.slice(i) = alpha.slice(sample(model.engine)).t();
    
  }
}


template <typename T>
void is_psi_ncm(T model, const arma::mat& theta_store, const arma::mat& y_store, const arma::mat& H_store,
  const arma::mat& ll_approx_u, const arma::uvec& counts, unsigned int nsim_states,
   arma::vec& weights_store, arma::cube& alpha_store) {
  
  for (unsigned int i = 0; i < theta_store.n_cols; i++) {
    
    
    model.y = y_store.col(i);
    model.H = H_store.col(i);
    model.HH = arma::square(H_store.col(i));
    arma::vec theta_i = theta_store.col(i);
    model.update_model(theta_i);
    
    
    arma::cube alpha(model.m, model.n, counts(i) * nsim_states);
    arma::mat V(counts(i) * nsim_states, model.n);
    arma::umat ind(counts(i) * nsim_states, model.n - 1);
    
    weights_store(i) = exp(model.psi_filter(counts(i) * nsim_states, alpha, V, ind, 0.0, ll_approx_u.col(i)));
    //backtrack all, could speed up a bit by tracking only one
    backtrack_pf(alpha, ind);
    
    arma::vec tmp = V.col(model.n - 1);
    std::discrete_distribution<> sample(tmp.begin(), tmp.end());
    
    alpha_store.slice(i) = alpha.slice(sample(model.engine)).t();
    
  }
}

template void is_psi_cm<ngssm>(ngssm model, const arma::mat& theta_store, const arma::mat& y_store, 
  const arma::mat& H_store, const arma::mat& ll_approx_u, const arma::uvec& counts, 
  unsigned int nsim_states, arma::vec& weights_store, arma::cube& alpha_store);
template void is_psi_cm<ng_bsm>(ng_bsm model, const arma::mat& theta_store, const arma::mat& y_store, 
  const arma::mat& H_store, const arma::mat& ll_approx_u, const arma::uvec& counts, 
  unsigned int nsim_states, arma::vec& weights_store, arma::cube& alpha_store);
template void is_psi_cm<svm>(svm model, const arma::mat& theta_store, const arma::mat& y_store, 
  const arma::mat& H_store, const arma::mat& ll_approx_u, const arma::uvec& counts, 
  unsigned int nsim_states, arma::vec& weights_store, arma::cube& alpha_store);

template void is_psi_ncm<ngssm>(ngssm model, const arma::mat& theta_store, const arma::mat& y_store, 
  const arma::mat& H_store, const arma::mat& ll_approx_u, const arma::uvec& counts, 
  unsigned int nsim_states, arma::vec& weights_store, arma::cube& alpha_store);
template void is_psi_ncm<ng_bsm>(ng_bsm model, const arma::mat& theta_store, const arma::mat& y_store, 
  const arma::mat& H_store, const arma::mat& ll_approx_u, const arma::uvec& counts, 
  unsigned int nsim_states, arma::vec& weights_store, arma::cube& alpha_store);
template void is_psi_ncm<svm>(svm model, const arma::mat& theta_store, const arma::mat& y_store, 
  const arma::mat& H_store, const arma::mat& ll_approx_u, const arma::uvec& counts, 
  unsigned int nsim_states, arma::vec& weights_store, arma::cube& alpha_store);


template void is_correction_psif<ngssm>(ngssm model, const arma::mat& theta_store, 
  const arma::mat& y_store, const arma::mat& H_store,
  const arma::mat& ll_approx_u, const arma::uvec& counts, unsigned int nsim_states,
  unsigned int n_threads, arma::vec& weights_store, arma::cube& alpha_store, 
  bool const_m, const arma::uvec& seeds);
template void is_correction_psif<ng_bsm>(ng_bsm model, const arma::mat& theta_store, 
  const arma::mat& y_store, const arma::mat& H_store,
  const arma::mat& ll_approx_u, const arma::uvec& counts, unsigned int nsim_states,
  unsigned int n_threads, arma::vec& weights_store, arma::cube& alpha_store, 
  bool const_m, const arma::uvec& seeds);
template void is_correction_psif<svm>(svm model, const arma::mat& theta_store, 
  const arma::mat& y_store, const arma::mat& H_store,
  const arma::mat& ll_approx_u, const arma::uvec& counts, unsigned int nsim_states,
  unsigned int n_threads, arma::vec& weights_store, arma::cube& alpha_store, 
  bool const_m, const arma::uvec& seeds);
