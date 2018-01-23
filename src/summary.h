#ifndef SUMMARY_H
#define SUMMARY_H

#include "bssm.h"

void running_summary(const arma::cube& x, arma::mat& mean_x, arma::cube& cov_x);
void running_weighted_summary(const arma::cube& x, arma::mat& mean_x,
  arma::cube& cov_x, const arma::vec& weights);
void filter_summary(const arma::cube& alpha, arma::mat& at, arma::mat& att, 
  arma::cube& Pt, arma::cube& Ptt, arma::mat weights);
#endif
