#include "ngssm.h"
#include "ng_bsm.h"
#include "svm.h"
#include "backtrack.h"

#ifdef _OPENMP
#include <omp.h>
#endif

template <typename T>
void is_correction_bsf(T model, const arma::mat& theta_store, const arma::vec& ll_store, 
  const arma::uvec& counts, unsigned int nsim_states, unsigned int n_threads, 
  arma::vec& weights_store, arma::cube& alpha_store, bool const_m, const arma::uvec& seeds) {
  
  if(n_threads > 1) {
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads) default(none)                \
    shared(n_threads, nsim_states, theta_store, ll_store,                \
      weights_store, alpha_store, counts, seeds, const_m) firstprivate(model)
      {
        unsigned n_iter = theta_store.n_cols;
        
        model.engine = std::mt19937(seeds(omp_get_thread_num()));
        unsigned thread_size = floor(n_iter/n_threads);
        unsigned int start = omp_get_thread_num()*thread_size;
        unsigned int end = (omp_get_thread_num()+1)*thread_size - 1;
        if(omp_get_thread_num() == (n_threads - 1)) {
          end = n_iter - 1;
        }
        
        arma::mat theta_piece = theta_store(arma::span::all, arma::span(start, end));
        arma::vec ll_piece = ll_store(arma::span(start, end));
        arma::uvec counts_piece = counts(arma::span(start, end));
        arma::vec weights_piece = weights_store(arma::span(start, end));
        arma::cube alpha_piece = alpha_store.slices(start, end);
        if(const_m) {
          is_bsf_cm(model, theta_piece, ll_piece, counts_piece, nsim_states, weights_piece, 
            alpha_piece);
        } else {
          is_bsf_ncm(model, theta_piece, ll_piece, counts_piece, nsim_states, weights_piece, 
            alpha_piece);
        }
        weights_store(arma::span(start, end)) = weights_piece;
        alpha_store.slices(start, end) = alpha_piece;
      }
#else
    if(const_m) {
      is_bsf_cm(model, theta_store, ll_store, counts, nsim_states, weights_store, 
        alpha_store);
    } else {
      is_bsf_ncm(model, theta_store, ll_store, counts, nsim_states, weights_store, 
        alpha_store);
    }
#endif
  } else {
    if(const_m) {
      is_bsf_cm(model, theta_store, ll_store, counts, nsim_states, weights_store, 
        alpha_store);
    } else {
      is_bsf_ncm(model, theta_store, ll_store, counts, nsim_states, weights_store, 
        alpha_store);
    }
  }
}

template <typename T>
void is_bsf_cm(T model, const arma::mat& theta_store, const arma::vec& ll_store, 
  const arma::uvec& counts, unsigned int nsim_states, arma::vec& weights_store, 
  arma::cube& alpha_store) {
  
  arma::cube alpha(model.m, model.n, nsim_states);
  arma::mat V(nsim_states, model.n);
  arma::umat ind(nsim_states, model.n - 1);
  
  for (unsigned int i = 0; i < theta_store.n_cols; i++) {
    
    arma::vec theta_i = theta_store.col(i);
    model.update_model(theta_i);
    
    double ll = model.particle_filter(nsim_states, alpha, V, ind);
    backtrack_pf(alpha, ind);
    weights_store(i) = exp(ll - ll_store(i));
    
    arma::vec tmp = V.col(model.n - 1);
    std::discrete_distribution<> sample(tmp.begin(), tmp.end());
    
    alpha_store.slice(i) = alpha.slice(sample(model.engine)).t();
    
  }
  
}

template <typename T>
void is_bsf_ncm(T model, const arma::mat& theta_store, const arma::vec& ll_store, 
  const arma::uvec& counts, unsigned int nsim_states, arma::vec& weights_store, 
  arma::cube& alpha_store) {
  
 
  for (unsigned int i = 0; i < theta_store.n_cols; i++) {
    
    arma::vec theta_i = theta_store.col(i);
    model.update_model(theta_i);
    
     arma::cube alpha(model.m, model.n, counts(i) * nsim_states);
  arma::mat V(counts(i) * nsim_states, model.n);
  arma::umat ind(counts(i) * nsim_states, model.n - 1);
  
    double ll = model.particle_filter(counts(i) * nsim_states, alpha, V, ind);
    backtrack_pf(alpha, ind);
    weights_store(i) = exp(ll - ll_store(i));
    
    arma::vec tmp = V.col(model.n - 1);
    std::discrete_distribution<> sample(tmp.begin(), tmp.end());
    
    alpha_store.slice(i) = alpha.slice(sample(model.engine)).t();
    
  }
  
}


template void is_bsf_cm<svm>(svm model, const arma::mat& theta_store, const arma::vec& ll_store, 
  const arma::uvec& counts, unsigned int nsim_states, arma::vec& weights_store, 
  arma::cube& alpha_store);
template void is_bsf_cm<ng_bsm>(ng_bsm model, const arma::mat& theta_store, const arma::vec& ll_store, 
  const arma::uvec& counts, unsigned int nsim_states, arma::vec& weights_store, 
  arma::cube& alpha_store);
template void is_bsf_cm<ngssm>(ngssm model, const arma::mat& theta_store, const arma::vec& ll_store, 
  const arma::uvec& counts, unsigned int nsim_states, arma::vec& weights_store, 
  arma::cube& alpha_store);
template void is_bsf_ncm<svm>(svm model, const arma::mat& theta_store, const arma::vec& ll_store, 
  const arma::uvec& counts, unsigned int nsim_states, arma::vec& weights_store, 
  arma::cube& alpha_store);
template void is_bsf_ncm<ng_bsm>(ng_bsm model, const arma::mat& theta_store, const arma::vec& ll_store, 
  const arma::uvec& counts, unsigned int nsim_states, arma::vec& weights_store, 
  arma::cube& alpha_store);
template void is_bsf_ncm<ngssm>(ngssm model, const arma::mat& theta_store, const arma::vec& ll_store, 
  const arma::uvec& counts, unsigned int nsim_states, arma::vec& weights_store, 
  arma::cube& alpha_store);

template void is_correction_bsf<ngssm>(ngssm model, const arma::mat& theta_store, const arma::vec& ll_store, 
  const arma::uvec& counts, unsigned int nsim_states, unsigned int n_threads, arma::vec& weights_store,
  arma::cube& alpha_store, bool const_m, const arma::uvec& seeds);
template void is_correction_bsf<ng_bsm>(ng_bsm model, const arma::mat& theta_store, const arma::vec& ll_store, 
  const arma::uvec& counts, unsigned int nsim_states, unsigned int n_threads, arma::vec& weights_store,
  arma::cube& alpha_store, bool const_m, const arma::uvec& seeds);
template void is_correction_bsf<svm>(svm model, const arma::mat& theta_store, const arma::vec& ll_store, 
  const arma::uvec& counts, unsigned int nsim_states, unsigned int n_threads, arma::vec& weights_store,
  arma::cube& alpha_store, bool const_m, const arma::uvec& seeds);
