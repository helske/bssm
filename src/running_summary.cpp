#include "bssm.h"

void running_summary(const arma::mat& x, arma::mat& mean_x, arma::cube& cov_x,
  unsigned int n) {
  
  cov_x *= n;
  
  arma::mat diff = x - mean_x;
  mean_x += diff / (n + 1);
  for (unsigned int t = 0; t < x.n_cols; t++) {
    cov_x.slice(t) += diff.col(t) * (x.col(t) - mean_x.col(t)).t();
  }
  cov_x /= (n + 1);
  
}

void running_weighted_summary(const arma::cube& x, arma::mat& mean_x, arma::cube& cov_x, 
  const arma::vec& weights) {
  
  cov_x.zeros();
  mean_x.zeros();
  double cumsumw = 0;
  for(unsigned int i = 0; i < x.n_slices; i++) {
    double tmp = weights(i) + cumsumw;
    arma::mat diff = x.slice(i) - mean_x;
    mean_x += diff * weights(i) / tmp;
    for (unsigned int t = 0; t < x.n_cols; t++) {
      cov_x.slice(t) +=  weights(i) * diff.col(t) * (x.slice(i).col(t) - mean_x.col(t)).t();
    }
    cumsumw = tmp;
  }
  cov_x = cov_x / cumsumw;
  
}


void filter_summary(const arma::cube& alpha, arma::mat& at, arma::mat& att, 
  arma::cube& Pt, arma::cube& Ptt, arma::mat& weights) {
  
  at.zeros();
  att.zeros();
  
  for (unsigned int t = 0; t < alpha.n_cols; t++) {
    weights.col(t) /= arma::sum(weights.col(t));
    for (unsigned int i = 0; i < alpha.n_slices; i++) {
      att.col(t) += alpha.slice(i).col(t) * weights(i, t);
      at.col(t) += alpha.slice(i).col(t);
    }
  }
  
  at /= alpha.n_slices;
  
  Pt.zeros();
  Ptt.zeros();
  for (unsigned int t = 0; t < alpha.n_cols; t++) {
    for(unsigned int i = 0; i < alpha.n_slices; i++) {
      Pt.slice(t) += 
        (alpha.slice(i).col(t) - at.col(t)) * (alpha.slice(i).col(t) - at.col(t)).t();
      Ptt.slice(t) += weights(i, t) * 
        (alpha.slice(i).col(t) - att.col(t)) * (alpha.slice(i).col(t) - att.col(t)).t();
      
    }
  }
  Pt /= alpha.n_slices;
}

