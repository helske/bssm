#ifndef ISC_H
#define ISC_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

template <typename T>
void is_correction(T mod, const arma::mat& theta, const arma::mat& y_store, const arma::mat& H_store,
  const arma::vec& ll_approx_u, const arma::uvec& counts, unsigned int nsim_states,
  unsigned int n_threads, arma::vec& weights_store, arma::cube& alpha_store, bool const_m);

template <typename T>
void is_correction_summary(T mod, const arma::mat& theta, const arma::mat& y_store, const arma::mat& H_store,
  const arma::vec& ll_approx_u, const arma::uvec& counts, unsigned int nsim_states,
  unsigned int n_threads, arma::vec& weights_store, 
  arma::mat& alphahat, arma::cube& Vt, arma::mat& mu, arma::cube& Vmu, bool const_m);


template <typename T>
void is_correction_bsf(T mod, const arma::mat& theta, const arma::vec& ll_store, 
  const arma::uvec& counts, unsigned int nsim_states,
  unsigned int n_threads, arma::vec& weights_store, arma::cube& alpha_store, bool const_m);


template <typename T>
void is_correction_psif(T mod, const arma::mat& theta, const arma::mat& y_store, const arma::mat& H_store,
  const arma::mat& ll_approx_u, const arma::uvec& counts, unsigned int nsim_states,
  unsigned int n_threads, arma::vec& weights_store, arma::cube& alpha_store, bool const_m);

template <typename T>
void is_correction_psif_summary(T mod, const arma::mat& theta, const arma::mat& y_store, const arma::mat& H_store,
  const arma::mat& ll_approx_u, const arma::uvec& counts, unsigned int nsim_states,
  unsigned int n_threads, arma::vec& weights_store, arma::mat& alphahat, arma::cube& Vt, arma::mat& mu, arma::cube& Vmu, 
  bool const_m);

#endif