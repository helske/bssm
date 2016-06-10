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

arma::mat intervals(arma::mat& means, arma::mat& sds, arma::vec& probs, unsigned int n_ahead);

double uv_filter_update(const double y, arma::subview_col<double> Z, const double HH,
  const double xbeta, arma::subview_col<double> at, arma::mat& Pt,
  arma::subview_col<double> att, arma::mat& Ptt);

void uv_filter_predict(const arma::mat& T, const arma::mat& RR,
  arma::subview_col<double> att, arma::mat& Ptt,
  arma::subview_col<double> at, arma::mat& Pt);

double uv_filter(const double y, arma::subview_col<double> Z, const double HH, const double xbeta,
  const arma::mat& T, const arma::mat& RR, arma::vec& at, arma::mat& Pt);

#endif
