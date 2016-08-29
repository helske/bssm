#ifndef BSSM_H
#define BSSM_H

#include <omp.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]
#define ARMA_NO_DEBUG

const double LOG2PI = std::log(2.0 * M_PI);

using namespace Rcpp;

arma::mat cholupdate(arma::mat L, arma::vec u);

arma::mat choldowndate(arma::mat L, arma::vec u);

arma::vec dmvnorm1(const arma::mat& x, const arma::vec& mean,  
  arma::mat sigma, bool lwr = false, bool logd = false);

arma::vec dmvnorm2(const arma::mat& x, const arma::mat& mean,  
  arma::mat sigma, bool lwr, bool logd, const arma::mat& A);

void adjust_S(arma::mat& S, arma::vec& u, double current, double target, unsigned int n, double gamma);

void running_summary(const arma::mat& x, arma::mat& mean_x, arma::cube& cov_x, const unsigned int n);
void running_weighted_summary(const arma::cube& x, arma::mat& mean_x, arma::cube& cov_x, const arma::vec& weights);
  
arma::mat intervals(arma::mat& means, const arma::mat& sds, const arma::vec& probs, unsigned int n_ahead);

double uv_filter_update(const double y, arma::subview_col<double> Z, const double HH,
  arma::subview_col<double> at, arma::mat& Pt,
  arma::subview_col<double> att, arma::mat& Ptt, const double zero_tol);

void uv_filter_predict(const arma::mat& T, const arma::mat& RR,
  arma::subview_col<double> att, arma::mat& Ptt,
  arma::subview_col<double> at, arma::mat& Pt);

double uv_filter(const double y, const arma::vec& Z, const double HH,
  const arma::mat& T, const arma::mat& RR, arma::vec& at, arma::mat& Pt, const double zero_tol);

template <typename T>
arma::cube sample_states(T mod, const arma::mat& theta, const arma::uvec& counts,
  unsigned int nsim_states, unsigned int n_threads, arma::uvec seeds);

template <typename T>
void is_correction(T mod, const arma::mat& theta, const arma::mat& y_store, const arma::mat& H_store,
  const arma::vec& ll_approx_u, const arma::uvec& counts, unsigned int nsim_states,
  unsigned int n_threads, arma::uvec seeds, arma::vec& weights_store, arma::cube& alpha_store);

template <typename T>
void is_correction_param(T mod, const arma::mat& theta, const arma::mat& y_store, const arma::mat& H_store,
  const arma::vec& ll_approx_u, const arma::uvec& counts, unsigned int nsim_states,
  unsigned int n_threads, arma::uvec seeds, arma::vec& weights_store);


template <typename T>
void is_correction_summary(T mod, const arma::mat& theta, const arma::mat& y_store, const arma::mat& H_store,
  const arma::vec& ll_approx_u, const arma::uvec& counts, unsigned int nsim_states,
  unsigned int n_threads, arma::uvec seeds, arma::vec& weights_store, 
  arma::mat& alphahat, arma::cube& Vt, arma::mat& mu, arma::cube& Vmu, bool const_nsim);


template <typename T>
void is_correction_bsf(T mod, const arma::mat& theta, const arma::vec& ll_store, const arma::uvec& counts, unsigned int nsim_states,
  unsigned int n_threads, arma::uvec seeds, arma::vec& weights_store, arma::cube& alpha_store);

#endif
