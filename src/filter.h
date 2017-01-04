#ifndef FILTER_H
#define FILTER_H

#include <RcppArmadillo.h>

double uv_filter_update(const double y, const arma::vec& Z, const double HH,
  arma::subview_col<double> at, arma::mat& Pt,
  arma::subview_col<double> att, arma::mat& Ptt, const double zero_tol);

void uv_filter_predict(const arma::mat& T, const arma::mat& RR, const arma::vec& C,
  arma::subview_col<double> att, arma::mat& Ptt,
  arma::subview_col<double> at, arma::mat& Pt);

double uv_filter(const double y, const arma::vec& Z, const double HH,
  const arma::mat& T, const arma::mat& RR, const arma::vec& C, arma::vec& at, arma::mat& Pt, 
  const double zero_tol);

#endif
