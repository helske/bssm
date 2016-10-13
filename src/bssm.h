#ifndef BSSM_H
#define BSSM_H

#ifdef _OPENMP
#include <omp.h>
#endif
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]
//#define ARMA_NO_DEBUG

const double LOG2PI = std::log(2.0 * M_PI);

using namespace Rcpp;

void backtrack_pf(arma::cube& alpha, arma::umat& ind);

arma::mat cholupdate(arma::mat L, arma::vec u);

arma::mat choldowndate(arma::mat L, arma::vec u);

double dmvnorm1(const arma::vec& x, const arma::vec& mean,  
  const arma::mat& sigma, bool lwr = false, bool logd = false);

void adjust_S(arma::mat& S, arma::vec& u, double current, double target, unsigned int n, double gamma);

void running_summary(const arma::mat& x, arma::mat& mean_x, arma::cube& cov_x, const unsigned int n);
void running_weighted_summary(const arma::cube& x, arma::mat& mean_x, arma::cube& cov_x, const arma::vec& weights);

arma::mat intervals(arma::mat& means, const arma::mat& sds, const arma::vec& probs, unsigned int n_ahead);

double uv_filter_update(const double y, const arma::vec& Z, const double HH,
  arma::subview_col<double> at, arma::mat& Pt,
  arma::subview_col<double> att, arma::mat& Ptt, const double zero_tol);

void uv_filter_predict(const arma::mat& T, const arma::mat& RR, const arma::vec& C,
  arma::subview_col<double> att, arma::mat& Ptt,
  arma::subview_col<double> at, arma::mat& Pt);

double uv_filter(const double y, const arma::vec& Z, const double HH,
  const arma::mat& T, const arma::mat& RR, const arma::vec& C, arma::vec& at, arma::mat& Pt, 
  const double zero_tol);

template <typename T>
arma::cube sample_states(T mod, const arma::mat& theta, const arma::uvec& counts,
  unsigned int nsim_states, unsigned int n_threads);

void conditional_dist_helper(arma::cube& V, arma::cube& C);

arma::uvec stratified_sample(arma::vec p, arma::vec& r, unsigned int N);

#endif
